import numpy as np
from hig_data.coco import COCOStuffGraphPrecomputedDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
import dnnlib
from tqdm import tqdm

def get_latent_statistics(dataset_root='/home/rfsm2/rds/hpc-work/coco/coco_graph_train'):

    coco_graph = COCOStuffGraphPrecomputedDataset(dataset_root)
    dls = GeoDataLoader(coco_graph, batch_size=1024, num_workers=4, prefetch_factor=2, drop_last=True)
   
    latent_means = []
    latent_stds = []

    for idx, batch in tqdm(enumerate(dls), total=len(dls)):
        graph_batch = batch.to('cuda')

        # store stats for encoder from first 4 channels
        latent_means.append(np.mean(graph_batch.image.cpu().numpy(), axis=(0, 2,3)))
        latent_stds.append(np.std(graph_batch.image.cpu().numpy(), axis=(0, 2,3)))
        
    latent_raw_mean = np.mean(np.array(latent_means), axis=0)
    latent_raw_std = np.mean(np.array(latent_stds), axis=0)
    np.savez(f'./raw_latent_statistics', latent_raw_mean=latent_raw_mean, latent_raw_std=latent_raw_std)
        
if __name__ == "__main__":
    get_latent_statistics()