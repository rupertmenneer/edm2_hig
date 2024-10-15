
import torch
import torch_geometric
from torch_geometric.data import HeteroData

def random_subgraph_collate(graph: HeteroData, subsampling_keys=['class_node', 'instance_node']):
    node_dict = {}
    
    for key in graph.node_types:
        num_nodes = graph[key].num_nodes
        if key in subsampling_keys:
            num_to_sample = torch.randint(1, num_nodes, (1,)).item()
            sample_indices = torch.randperm(num_nodes)[:num_to_sample]
            node_dict[key] = sample_indices
        else: # Include all nodes for keys not in subsampling_keys
            node_dict[key] = torch.arange(num_nodes)

    return graph.subgraph(node_dict)

    
class HIGCollator(torch_geometric.loader.dataloader.Collater):
    def __init__(self, dataset, subsample=False, subsampling_keys=None):
        super().__init__(dataset)
        self.subsample = subsample
        self.subsampling_keys = subsampling_keys if subsampling_keys is not None else ['class_node', 'instance_node']
    
    def __call__(self, batch):
        batched_graph = super().__call__(batch) # call base class Collater to get the batched graph
        if isinstance(batched_graph, torch_geometric.data.Batch) and self.subsample: # if the batched graph is a Batch object and subsample is True, then subsample the graph
            return random_subgraph_collate(batched_graph, self.subsampling_keys)
        return batched_graph

class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        subsample: bool = False,
        **kwargs,
        ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=HIGCollator(dataset, subsample=subsample),
            **kwargs,
        )    