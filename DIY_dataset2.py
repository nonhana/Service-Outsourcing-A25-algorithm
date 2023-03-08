import torch
from torch_geometric.data import Data
import networkx as nx 
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

edge_index=torch.tensor(
    [[0,0,0,1,2,2,3,3],
    [1,2,3,0,0,3,0,2]],dtype=torch.long
)
node_attr=torch.tensor(
    [[-1,1,2],[1,1,1],[0,1,2],[3,1,2]]
)

edge_attr=torch.tensor(
    [[0,0,0],[0,1,2],[0,3,3],[0,0,0],[0,0,2],[0,2,3],[0,3,3],[0,2,3]]
)

data=Data(x=node_attr,edge_index=edge_index,edge_attr=edge_attr)

# data.num_classes = torch.max(data.y).item() + 1

print(data.y)