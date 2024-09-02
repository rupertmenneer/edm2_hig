
import torch
import numpy as np
from torch_utils import persistence

from torch_geometric.nn import SAGEConv, to_hetero, Linear

#----------------------------------------------------------------------------
# Interface class that creates the GNN, and handles representation switching

@persistence.persistent_class
class DualGNNInterface(torch.nn.Module):
    def __init__(self,
        metadata,
        gnn_channels,
        num_gnn_layers=2,
    ):
        super().__init__()
        gnn = MP_GNN(gnn_channels, gnn_channels, num_gnn_layers)
        self.gnn = to_hetero(gnn, metadata, aggr='sum')

    def update_graph_image_nodes(self, x, graph):
        # reshape image into image nodes [B, C, H, W] -> [B * H * W, C]
        graph['image_node'].x.copy_(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))
        return graph
        
    def extract_image_nodes(self, graph):
        # reshape image nodes back into image [B * H * W, C] -> [B, C, H, W]
        b,_,h,w = graph.target_image.shape
        return graph['image_node'].x.reshape(b, h, w, -1).permute(0, 3, 1, 2) 

    def update_graph_embeddings(self, x, graph):
        # updates HeteroData object with updated GNN features   
        for key, emb in x.items():
            graph[key].x = emb
        return graph

    def forward(self, x, graph):
        graph = self.update_graph_image_nodes(x, graph) # update and resize image nodes on graph with current feature map

        y = self.gnn(graph.x_dict, graph.edge_index_dict) # pass dual graph through GNN

        graph = self.update_graph_embeddings(y, graph) # update graph with new embeddings
        
        out = self.extract_image_nodes(graph) # extract and resize image nodes back to image

        return out, graph


#----------------------------------------------------------------------------
# Simple lazy initialised GNN using SAGE Conv with l2 feature normalisation 

class MP_GNN(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels, num_gnn_layers=2,):
        super().__init__()
        self.gnn_layers = torch.nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(MP_SAGEConv((-1,-1), hidden_channels, normalize=True))
        self.gnn_layers.append(Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index,):
        for block in self.gnn_layers:
            x = block(x) if isinstance(block, Linear) else heterogenous_mp_silu(block(x, edge_index))
        return x
    
#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

def heterogenous_mp_silu(x):
    return {key: mp_silu(x[key]) for key in x.keys()} if isinstance(x, torch.Tensor) else x

#----------------------------------------------------------------------------
# Magnitude-preserving Graph SAGE Conv
# with force weight normalization (Equation 66 karras).
    
class MP_SAGEConv(SAGEConv):
    def forward(self, x, edge_index):
        # access and normalize the weight matrices if lazy initialisation complete
        if not has_uninitialized_params(self): 
            self.normalise_weights('lin_l', x[0].dtype)
            self.normalise_weights('lin_r', x[0].dtype)
        
        # call forward method as usual
        return super().forward(x, edge_index)
        
    # modified norm function, normalises weights based off attr name so can be re-used if class has multiple weights to norm
    def normalise_weights(self, weight_name, dtype):
        w = getattr(self, weight_name).weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                getattr(self, weight_name).weight.copy_(normalize(w)) # Forced weight normalization
        w = normalize(w) # Traditional weight normalization
        w = w * (1 / np.sqrt(w[0].numel())) # Magnitude-preserving scaling
        getattr(self, weight_name).weight.data = w.to(dtype)


# check if param is UninitializedParameter - from pytorch geometric linear.py
def is_uninitialized_parameter(x) -> bool:
    if not hasattr(torch.nn.parameter, 'UninitializedParameter'):
        return False
    return isinstance(x, torch.nn.parameter.UninitializedParameter)

# find if model has UninitializedParameter(s)
def has_uninitialized_params(model):
    for param in model.parameters():
        if is_uninitialized_parameter(param):
            return True
    return False