import numpy as np
import os

def create_graph_dataset_and_export(output_dir='/home/rfsm2/rds/hpc-work/coco/coco_graph_train'):

    dataset_root = '/home/rfsm2/rds/hpc-work/coco/'
    
    img_path = dataset_root+'coco_train2017_256-sd.zip'
    mask_path = dataset_root+'coco_train2017_masks_256.zip'
    coco_graph = CocoStuffGraphDataset(img_path, mask_path, latent_images=True)

    for idx in range(len(coco_graph)):
        image, mask = coco_graph.dataset[idx]
        graph = coco_graph[idx]

        name = coco_graph.dataset._fnames['image'][coco_graph.dataset._raw_idx[idx]]
        raw_id = os.path.splitext(os.path.basename(name))[0].split('-')[-1]

        np.save(os.path.join(output_dir, f'{raw_id}_image.npy'), image)
        np.save(os.path.join(output_dir, f'{raw_id}_mask.npy'), mask)
        np.savez(os.path.join(output_dir, f'{raw_id}_graph'),
                 class_node=graph['class_node'].x,
                 class_pos=graph['class_node'].pos,
                 class_edge=graph['class_node', 'class_edge', 'class_node'].edge_index,
                 class_to_image=graph['class_node', 'class_to_image', 'image_node'].edge_index,
                 )
    
if __name__ == "__main__":
    from hig_data.coco import CocoStuffGraphDataset
    create_graph_dataset_and_export()