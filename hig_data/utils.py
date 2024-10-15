
import torch
from torch import Tensor
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
