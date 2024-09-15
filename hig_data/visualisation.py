
import wandb
import torch
import matplotlib.pyplot as plt
import networkx as nx
import dnnlib
import numpy as np

# handles batch or single image, n sets number to display
def plot_array_images(images, n=8, save_path=None):
    
    if not isinstance(images, np.ndarray):
        images = images.numpy()

    if images.ndim == 2:
        images = images[np.newaxis]
    if images.ndim == 3:
        images = images[np.newaxis]
    if images.shape[0] < n:
        n = images.shape[0]

    if n == 1:
        img = images[0].transpose(1, 2, 0)
        plt.figure(figsize=(2, 5), dpi=300)
        plt.imshow(img,)
        plt.axis('off')

    else:
        _, axes = plt.subplots(1, n, figsize=(10, 10), dpi=300)
        for i in range(n):
            img = images[i]
            if img.shape[0] in [1, 3]: # if image is in CHW format
                img = img.transpose(1, 2, 0)

            axes[i].imshow(img)
            axes[i].axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()

@torch.no_grad()
def logging_generate_sample_vis(
        batch,
        sampled_images,
        n=8,
        labels=['HIG', 'Ground Truth Decoded Latents', 'Sampled Image'],
        title="Sampled Images - Training Graphs",
        latent_images=True,
        **kwargs,
    ):

    n = n if n < batch.image.shape[0] else batch.image.shape[0]

    # create image with ground truth graph overlay
    vae = None if not latent_images else dnnlib.util.construct_class_by_name(class_name='training.encoders.StabilityVAEEncoder')


    sampled_images = sampled_images[:n] # take first n
    sampled_images_pixels = sampled_images if vae is None else batch_convert_sampled_to_pixels(sampled_images, vae) # convert to pix
    graph_on_image_tensor, images = visualise_het_graph_on_image_batch(batch, n=n, vae=vae, **kwargs)
    
    

    # save to wandb
    save_image_batch_list([graph_on_image_tensor,
                           images,
                           sampled_images_pixels,],
                            row_labels=labels,
                            title=title,
                            sample_batch_size=n)


"""
Input: list of image batches -> [[B, C, H, W], ...]
with matching list of labels, save row wise to wandb
"""
def save_image_batch_list(
        image_batch_list,
        row_labels,
        sample_batch_size = 8,
        vis_size_factor = 3,
        title = "Image batch"
    ):

    
    cols = sample_batch_size if sample_batch_size < len(image_batch_list[0]) else len(image_batch_list[0])
    rows = len(image_batch_list)


    fig = plt.figure(figsize = (cols*vis_size_factor, rows*vis_size_factor), constrained_layout=True, dpi=300)
    fig.suptitle(title,fontsize=10*vis_size_factor)
    fig.patch.set_facecolor('white')

    # create n * 1 subfigs
    subfigs = fig.subfigures(nrows=rows, ncols=1)

    for row, (subfig, image_set) in enumerate(zip(subfigs, image_batch_list)):
        subfig.suptitle(row_labels[row], fontsize = 5*vis_size_factor)

        # create 1 * batchsize subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=cols)
        for col, (ax, ax_img) in enumerate(zip(axs, image_set)):
            ax.imshow(ax_img)
            ax.axis('off')

    if wandb.run is not None:
        wandb.log({f"{title}": fig})
    else:
        plt.show()


def visualise_het_graph_on_image_batch(graph_batch, n=8, vae=None, **kwargs): # unpack graph batch and return list of images
    images = []
    decoded_images = []
    for i, graph in enumerate(graph_batch.to_data_list()):
        if i >= n:
            break
        decoded_img = graph.image if vae is None else convert_latents_to_pixels(graph.image, vae)
        decoded_images.append(decoded_img[np.newaxis, ...])
        graph_on_image = visualise_het_graph_on_image(graph, images=decoded_img, return_image=True, **kwargs)[np.newaxis, ...]
        images.append(graph_on_image)
    return np.concatenate(np.array(images), axis=0).transpose(0,2,3,1), np.concatenate(np.array(decoded_images), axis=0)

def batch_convert_sampled_to_pixels(batch, vae):

    decoded_images = []
    for i in range(batch.shape[0]):
        sampled_image = batch[i].unsqueeze(0)
        pix = vae.decode(sampled_image)[0].permute(1, 2, 0).cpu().numpy()
        decoded_images.append(pix[np.newaxis, ...])
    return np.concatenate(np.array(decoded_images), axis=0)

def convert_latents_to_pixels(std_mean, vae=None):
    if vae is None:
        vae = dnnlib.util.construct_class_by_name(class_name='training.encoders.StabilityVAEEncoder')
    if std_mean.dim() == 3:
        std_mean = std_mean.unsqueeze(0)
    latents = vae.encode_latents(std_mean)
    pix = vae.decode(latents)[0].permute(1, 2, 0).cpu().numpy()
    return pix

"""Visualise HIG representation, display both underlying image and graph nodes on top
Note: hetero_data must contain node positions for each node type"""
def visualise_het_graph_on_image(
        hetero_data,
        images,
        image_alpha=0.8,
        node_types=['image_node', 'class_node'],
        edge_types=[('class_node', 'class_to_image', 'image_node'), ('class_node', 'class_edge', 'class_node')],
        linewidth=1.5,
        return_image=False,
        **kwargs
    ):

    assert isinstance(images, np.ndarray), "Images must be a numpy array"

    # Set up the matplotlib figure
    dpi = 300
    image_size = images.shape[0]
    fig, ax = plt.subplots(figsize=(image_size/dpi, image_size/dpi), dpi=dpi)

    ax.imshow(images, alpha=image_alpha)

    if hasattr(hetero_data, 'mask_path'):
        mask = np.load(hetero_data.mask_path, allow_pickle=True) 
        ax.imshow(mask.squeeze(), alpha=0.45)
    
    # Create a NetworkX graph
    G = nx.Graph()
    # Get node positions and add them to the graph
    for node_type in node_types:
        assert hetero_data[node_type].pos is not None, f"Node type {node_type} does not have positions"
        node_positions = hetero_data[node_type].pos

        class_labels = None
        if hasattr(hetero_data[node_type], 'label'):
            class_labels = torch.argmax(hetero_data[node_type].x, dim=1)
        for i in range(node_positions.size(0)):
            cls = class_labels[i] if class_labels is not None else 0
            G.add_node(f'{node_type}_{i}', type=node_type, class_type=int(cls), pos=(node_positions[i, 0].item(), node_positions[i, 1].item()))

    # Add edges to the graph
    for edge_type in edge_types:
        assert hetero_data[edge_type].edge_index is not None
        edge_index = hetero_data[edge_type].edge_index
        for i in range(edge_index.size(1)):

            src_node = f'{edge_type[0]}_{edge_index[0, i].item()}'
            tgt_node = f'{edge_type[2]}_{edge_index[1, i].item()}'

            G.add_edge(src_node, tgt_node, type=edge_type)

    # Get positions from node attributes
    pos = nx.get_node_attributes(G, 'pos')
    # Example usage within your main visualization function
    node_colors = get_node_colors(G)

    nodes_alphas = [0.9 if 'class_node' in n else 0. for n in G.nodes()]
    nodes_size = [10 if 'class_node' in n else 0.0 for n in G.nodes()]
    # Draw the graph on top of the image
    nx.draw(G, pos, with_labels=False, node_size=nodes_size, alpha=nodes_alphas, node_color=node_colors, width=linewidth, ax=ax, edgelist=[])
    
    edges = G.edges(data=True)
    edge_alphas = [0.25 if 'class_edge' in e[2]['type'] else 0.1 for e in edges]
    edge_widths = [0.75 if 'class_edge' in e[2]['type'] else 0.5 for e in edges]
    edge_styles = ['dashed' if 'class_edge' in e[2]['type'] else 'solid' for e in edges]
    edge_colors = ['black' if 'class_edge' in e[2]['type'] else 'white' for e in edges]
    
    for i, (u, v, data) in enumerate(edges):        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=edge_colors[i],
            alpha=edge_alphas[i],
            width=edge_widths[i],
            style=edge_styles[i],
        )

    ax.set_aspect("equal")
    ax.grid(False)

    ax.set_xlim([0, image_size])
    ax.set_ylim([0, image_size])
    ax.axis('off') 
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=0)
    if return_image:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(h, w, 4)[:,:,:3].transpose(2, 0, 1)
        plt.close()
        return np.flip(image, 1)
    else:
        plt.show()
        plt.axis('off')


# utility function to generate a class node color
def generate_color_for_type(class_type, num_colors=185):
    colormap = plt.get_cmap('gist_rainbow', num_colors)  # Using tab10 colormap for 10 distinct colors
    return colormap(class_type)

# Utility function to get node colors based on node types
def get_node_colors(G, color_nodes = ['class_node']):
    node_colors = []
    for node in G.nodes(data=True):
        if node[1] and node[1]['type'] in color_nodes:
            node_type = node[1]['class_type']
            node_colors.append(generate_color_for_type(node_type,))
        else:
            node_colors.append('black')
    return node_colors


