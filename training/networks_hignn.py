
import torch
import numpy as np
from typing import Union, Tuple
import torch_geometric

from torch_geometric.nn import MessagePassing

from torch_utils import persistence

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Interface class that creates the GNN, and handles representation switching

@persistence.persistent_class
class HIGnnInterface(torch.nn.Module):
    def __init__(self,
        metadata,
        gnn_channels,
        num_gnn_layers=2,
    ):
        super().__init__()
        gnn = MP_GNN(gnn_channels, num_gnn_layers)
        self.gnn = torch_geometric.nn.to_hetero(gnn, metadata, aggr="mean")

    def update_graph_image_nodes(self, x, graph):
        # reshape image into image nodes [B, C, H, W] -> [B * H * W, C]
        reshape_x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        if graph['image_node'].x.ndim == 1:
            graph['image_node'].x = reshape_x # overwrites placeholder
        else:
            graph['image_node'].x.copy_(reshape_x) # updates existing image node
        return graph
        
    def extract_image_nodes(self, graph, shape):
        # reshape image nodes back into image [B * H * W, C] -> [B, C, H, W]
        b,_,h,w = shape
        return graph['image_node'].x.reshape(b, h, w, -1).permute(0, 3, 1, 2) 

    def update_graph_embeddings(self, x, graph):
        # updates HeteroData object with updated GNN features   
        for key, emb in x.items():
            graph[key].x = emb
        return graph

    def forward(self, x, graph):

        graph = self.update_graph_image_nodes(x, graph) # update and resize image nodes on graph with current feature map
        
        y = self.gnn(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict) # pass dual graph through GNN

        graph = self.update_graph_embeddings(y, graph) # update graph with new embeddings
        
        out = self.extract_image_nodes(graph, x.shape) # extract and resize image nodes back to image

        return out, graph


#----------------------------------------------------------------------------
# Simple lazy initialised GNN using SAGE Conv with l2 feature normalisation 

class MP_GNN(torch.nn.Module):

    def __init__(self, hidden_channels, num_gnn_layers=2,):
        super().__init__()
        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(MP_GeoLinear(-1, hidden_channels,)) # projection layer
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(MP_HIPGnnConv((-1,-1), hidden_channels, normalize=True)) # gnn layers
        self.gnn_layers.append(MP_GeoLinear(hidden_channels, hidden_channels,)) # output layer

    def forward(self, x, edge_index, edge_attr):
        for block in self.gnn_layers:            
            x = block(x) if isinstance(block, (MP_GeoLinear)) else block(heterogenous_mp_silu(x), edge_index=edge_index, edge_attr=edge_attr)
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
    
class MP_HIPGnnConv(torch_geometric.nn.MessagePassing):

    def __init__(self,
                 in_channels        = Union[int, Tuple[int, int]], # input channels for L and R branches
                 out_channels       = int,                         # output channels
                 normalize          = False,                       # normalise output features
                 aggr               = "mean",                      # aggregation scheme
                 **kwargs,
        ):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(aggr, **kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        if isinstance(self.aggr_module, torch_geometric.nn.aggr.MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = MP_GeoLinear(aggr_out_channels, out_channels,)
        self.lin_r = MP_GeoLinear(in_channels[1], out_channels,)
        self.normalize = normalize
        self.reset_parameters()

    # modified norm function, normalises weights based off attr name so can be re-used if class has multiple weights to norm
    def normalise_weights(self, weight_name, dtype):
        w = getattr(self, weight_name).weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                getattr(self, weight_name).weight.copy_(normalize(w)) # Forced weight normalization
        w = normalize(w) # Traditional weight normalization
        w = w * (1 / np.sqrt(w[0].numel())) # Magnitude-preserving scaling
        getattr(self, weight_name).weight.data = w.to(dtype)

    def forward(
        self,
        x: Union[torch.Tensor, torch_geometric.typing.OptPairTensor],
        edge_index: torch_geometric.typing.Adj,
        size: torch_geometric.typing.Size = None,
        edge_attr: torch_geometric.typing.OptPairTensor = None,
    ) -> torch.Tensor:
        
        # access and normalize the weight matrices if lazy initialisation complete
        if not has_uninitialized_params(self): 
            self.normalise_weights('lin_l', x[0].dtype)
            self.normalise_weights('lin_r', x[0].dtype)

        if isinstance(x, torch.Tensor):
            x = (x, x) # split into two branches

        # custom MP cat propagate function to support edge attr if required
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size) # propagate (with MP-cat if edge attribute exists)
        out = self.lin_l(out) # left weight matrix

        x_r = x[1]
        if x_r is not None:
            out = mp_sum(out, self.lin_r(x_r)) # apply right weight matrix and MP sum to connect branches

        if self.normalize: # optional norm on output features
            out = torch.nn.functional.normalize(out, p=2., dim=-1)

        return out

    def propagate(self, edge_index, size=None, **kwargs):
        return super().propagate(edge_index, size=size, **kwargs)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            out = mp_cat(x_j, edge_attr)
            return out
        return x_j
    
    def message_and_aggregate(self, adj_t: torch_geometric.typing.Adj, x: torch_geometric.typing.OptPairTensor) -> torch.Tensor:
        if isinstance(adj_t, torch_geometric.typing.SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return torch_geometric.utils.spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


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


import copy

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


@persistence.persistent_class
class MP_GeoLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=[]):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        # handle lazy initialisation
        if in_channels > 0:
            self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
        else:
            self.weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)
        self.reset_parameters()

    # modifed from MPConv Karras et al. 2024
    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))
    
    def reset_parameters(self,): # added reset parameters method for lazy hetero initialisation
        if self.in_channels > 0:
            torch.nn.init._no_grad_normal_(self.weight, 0, 1)

    # -----------
    # Following methods add lazy initialisation support - adapted from torch_geometric.nn.Linear.

    def __deepcopy__(self, memo):
        # PyTorch<1.13 cannot handle deep copies of uninitialized parameters
        out = MP_GeoLinear(self.in_channels, self.out_channels,).to(self.weight.device)
        if self.in_channels > 0:
            self.weight = copy.deepcopy(self.weight, memo)
        return out

    @torch.no_grad()
    def initialize_parameters(self, _, input):
        if is_uninitialized_parameter(self.weight):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if (is_uninitialized_parameter(self.weight)
                or torch.onnx.is_in_onnx_export() or keep_vars):
            destination[prefix + 'weight'] = self.weight
        else:
            destination[prefix + 'weight'] = self.weight.detach()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = state_dict.get(prefix + 'weight', None)

        if weight is not None and is_uninitialized_parameter(weight):
            self.in_channels = -1
            self.weight = torch.nn.parameter.UninitializedParameter()
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

        elif weight is not None and is_uninitialized_parameter(self.weight):
            self.in_channels = weight.size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels},)')