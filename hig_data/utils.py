
import torch
import torch_geometric
from torch_geometric.data import HeteroData
from hig_data.augmentations import BatchedHIGAugmentation

def random_subgraph_collate(graph: HeteroData, subsampling_keys=['class_node', 'instance_node']):
    node_dict = {}

    for key in graph.node_types:
        num_nodes = graph[key].num_nodes
        if key in subsampling_keys and num_nodes >= 3:
            # Ensure at least one node is selected (no zero-length batches)
            num_to_sample = torch.randint(2, num_nodes, (1,)).item()
            sample_indices = torch.randperm(num_nodes)[:num_to_sample]
            node_dict[key] = sample_indices
        else:  # Include all nodes for keys not in subsampling_keys
            node_dict[key] = torch.arange(num_nodes)

    return graph.subgraph(node_dict)

    
class HIGCollator(torch_geometric.loader.dataloader.Collater):
    def __init__(self, dataset, subsample=False, subsampling_keys=None, augmentation=False):
        super().__init__(dataset)
        self.subsample = subsample
        self.subsampling_keys = subsampling_keys if subsampling_keys is not None else ['class_node', 'instance_node']
        self.augmentation = None

        self.augmentation = BatchedHIGAugmentation(train=augmentation)
    
    def __call__(self, batch):
        # Apply subsampling before collating into a batch

        batch = self.augmentation(batch)
        if self.subsample:
            batch = [random_subgraph_collate(graph, self.subsampling_keys) for graph in batch]
        batched_graph = super().__call__(batch)  # Call base class Collater to get the batched graph
        return batched_graph
        # return batch

class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        subsample: bool = False,
        augmentation: bool = False,
        **kwargs,
        ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=HIGCollator(dataset, subsample=subsample, augmentation=augmentation),
            **kwargs,
        )    