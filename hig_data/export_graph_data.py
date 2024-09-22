import numpy as np
import os

def create_coco_graph_dataset_and_export(dataset_root='/home/rfsm2/rds/hpc-work/coco/', output_dir = 'coco_graph', suffix='val'):
    
    output_dir=f'{output_dir}_{suffix}', 
    img_path = os.path.join(dataset_root, f'coco_{suffix}2017_256-sdxl.zip')
    mask_path = os.path.join(dataset_root, f'coco_{suffix}2017_masks_256.zip')
    output_dir = os.path.join(dataset_root, output_dir)

    if not os.path.exists(output_dir): # make if doesn't exist
        os.makedirs(output_dir) 
    
    coco_graph = CocoStuffGraphDataset(img_path, mask_path, latent_images=True)

    latent_means = []
    latent_stds = []

    for idx in range(len(coco_graph)):
        image, mask = coco_graph.dataset[idx]

        # store stats for encoder from first 4 channels
        latent_means.append(np.mean(image[:4], axis=(1,2)))
        latent_stds.append(np.std(image[:4], axis=(1,2)))

        # get graph
        graph = coco_graph[idx]

        name = coco_graph.dataset._fnames['image'][coco_graph.dataset._raw_idx[idx]]
        raw_id = os.path.splitext(os.path.basename(name))[0].split('-')[-1]

        # save image/mask/graph
        np.save(os.path.join(output_dir, f'{raw_id}_image.npy'), image)
        np.save(os.path.join(output_dir, f'{raw_id}_mask.npy'), mask)
        np.savez(os.path.join(output_dir, f'{raw_id}_graph'),
                 class_node=graph['class_node'].x,
                 class_pos=graph['class_node'].pos,
                 class_edge=graph['class_node', 'class_edge', 'class_node'].edge_index,
                 class_to_image=graph['class_node', 'class_to_image', 'image_node'].edge_index,
                 )
        
    latent_raw_mean = np.mean(latent_means, axis=0)
    latent_raw_std = np.mean(latent_stds, axis=0)
    np.savez(os.path.join(dataset_root, f'raw_latent_statistics'), latent_raw_mean=latent_raw_mean, latent_raw_std=latent_raw_std)
        
if __name__ == "__main__":
    from hig_data.coco import CocoStuffGraphDataset
    create_coco_graph_dataset_and_export()