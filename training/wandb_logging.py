
import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL
import networkx as nx

@torch.no_grad()
def logging_generate_sample_vis(
        batch,
        n=8,
        labels=['HIG', 'Ground Truth', 'Sampled Image'],
        title="Sampled Images - Validation Graphs"
    ):

    n = n if n < batch.image.shape[0] else batch.image.shape[0]

    # sample images with graphs
    # sampled_images = self.sample(init_graph_batch=batch, n=n)
    sampled_images = torch.zeros_like(batch.image, device=batch.image.device)

    # create image with ground truth graph overlay
    graph_on_image_tensor = visualise_het_graph_on_image_batch(batch, n=n,)
    print('graph_on_image_tensor', graph_on_image_tensor.shape)

    # save to wandb
    save_image_batch_list([graph_on_image_tensor,
                           batch.image.numpy(),
                           sampled_images.numpy(),],
                        row_labels=labels,
                        use_wandb=False,
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
        use_wandb = True,
        vis_size_factor = 8,
        title = "Image batch"
    ):

    cols = sample_batch_size if sample_batch_size < len(image_batch_list[0]) else len(image_batch_list[0])
    rows = len(image_batch_list)

    fig = plt.figure(figsize = (cols*vis_size_factor, rows*vis_size_factor), constrained_layout=True)
    fig.suptitle(title,fontsize=10*vis_size_factor)
    fig.patch.set_facecolor('white')

    # create n * 1 subfigs
    subfigs = fig.subfigures(nrows=rows, ncols=1)

    for row, (subfig, image_set) in enumerate(zip(subfigs, image_batch_list)):
        subfig.suptitle(row_labels[row], fontsize = 5*vis_size_factor)

        # create 1 * batchsize subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=cols)
        for col, (ax, ax_img) in enumerate(zip(axs, image_set)):
            img = ax_img.transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis('off')

    if use_wandb:
        wandb.log({f"{title}": fig})
    else:
        plt.show()


def visualise_het_graph_on_image_batch(graph_batch, n=8): # unpack graph batch and return list of images
    images = []
    for i, graph in enumerate(graph_batch.to_data_list()):
        if i >= n:
            break
        graph_on_image = visualise_het_graph_on_image(graph, return_image=True)[np.newaxis, ...]
        images.append(graph_on_image)
    return np.concatenate(np.array(images), axis=0)

"""Visualise HIG representation, display both underlying image and graph nodes on top
Note: hetero_data must contain node positions for each node type"""
def visualise_het_graph_on_image(
        hetero_data,
        image_alpha=0.8,
        node_types=['image_node', 'class_node'],
        edge_types=[('class_node', 'class_to_image', 'image_node'), ('class_node', 'class_edge', 'class_node')],
        image_size=256,
        linewidth=1.5,
        resize_mask=False,
        return_image=False,
    ):
    # Set up the matplotlib figure
    dpi = 1000
    fig, ax = plt.subplots(figsize=(image_size/dpi, image_size/dpi), dpi=dpi)

    # Display the image with alpha transparency
    ax.imshow(hetero_data.image.squeeze().permute(1,2,0), alpha=image_alpha)

    mask = hetero_data.mask
    if resize_mask:
        resized_mask = torch.nn.functional.interpolate(mask.float(), scale_factor=1/8, mode='nearest')
        mask = torch.nn.functional.interpolate(resized_mask, scale_factor=8, mode='nearest')
        
    ax.imshow(mask.squeeze(), alpha=0.45)
    
    # Create a NetworkX graph
    G = nx.Graph()

    # Get node positions and add them to the graph
    for node_type in node_types:
        assert hetero_data[node_type].pos is not None, f"Node type {node_type} does not have positions"
        node_positions = hetero_data[node_type].pos 
        node_features = hetero_data[node_type].x
        for i in range(node_positions.size(0)):

            G.add_node(f'{node_type}_{i}', type=node_type, class_type=int(node_features[i][0]), pos=(node_positions[i, 0].item(), node_positions[i, 1].item()))

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
    nodes_size = [0.35 if 'class_node' in n else 0.0 for n in G.nodes()]
    # Draw the graph on top of the image
    nx.draw(G, pos, with_labels=False, node_size=nodes_size, alpha=nodes_alphas, node_color=node_colors, width=linewidth, ax=ax, edgelist=[])
    
    edges = G.edges(data=True)
    edge_alphas = [0.5 if 'class_edge' in e[2]['type'] else 0.1 for e in edges]
    edge_widths = [0.25 if 'class_edge' in e[2]['type'] else 0.1 for e in edges]
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
    ax.set_xlim([0, hetero_data.image.shape[-1]])
    ax.set_ylim([0, hetero_data.image.shape[-1]])
    ax.axis('off') 
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=0)
    if return_image:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(h, w, 4)
        pil_image = np.array(PIL.Image.fromarray(image, mode='RGBA').convert('RGB')).transpose(2, 0, 1)
        plt.close()
        return np.flip(pil_image, 1)
    else:
        plt.show()
        plt.axis('off')


# utility function to generate a class node color
def generate_color_for_type(class_type, num_colors=185):
    colormap = plt.get_cmap('gist_rainbow', num_colors)  # Using tab10 colormap for 10 distinct colors
    return colormap(class_type)

# Utility function to get node colors based on node types
def get_node_colors(G,):
    node_colors = []
    for node in G.nodes(data=True):
        if node[1]:
            node_type = node[1]['class_type']
            node_colors.append(generate_color_for_type(node_type,))
    return node_colors


