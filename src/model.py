
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, Linear, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops


class SHGNN(nn.Module):
    def __init__(self, num_layers, dim, num_class, dp, convs:bool=False):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.self_loop = True
        self.convs = convs
        
        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        for _ in range(num_layers):
            # self.N2E_covs.append(SAGEConv(-1, dim))
            # self.E2N_covs.append(SAGEConv(-1, dim))
            # self.N2E_covs.append(GATConv(-1, dim))
            # self.E2N_covs.append(GATConv(-1, dim))
            self.N2E_covs.append(GIN(-1, 2*dim, 1, dim, dropout=dp))
            self.E2N_covs.append(GIN(-1, 2*dim, 1, dim, dropout=dp))
            # self.N2E_covs.append(GAT(-1, dim, 1, dim, dropout=dp))
            # self.E2N_covs.append(GAT(-1, dim, 1, dim, dropout=dp))
        
        self.out = nn.Sequential(Linear(-1, dim),
                                 nn.ReLU(),
                                 Linear(-1, num_class),
                                 nn.LogSoftmax(dim=-1))
        
    def forward(self, edge_sub_batch, node_sub_batch, node_x):
        if self.self_loop:
            edge_1 = add_self_loops(edge_sub_batch.edge_index)[0]
            edge_2 = add_self_loops(node_sub_batch.edge_index)[0]
        else:
            edge_1 = edge_sub_batch.edge_index
            edge_2 = node_sub_batch.edge_index
        
        node_x = F.dropout(node_x, p=self.dp, training=self.training) # Input dropout
        for i in range(self.num_layers):
            xs = node_x
            copyed_node_x = node_x[edge_sub_batch.nodes_map]
            if self.convs:
                copyed_node_x = self.N2E_covs[i](copyed_node_x, edge_1)
            edge_x = global_mean_pool(copyed_node_x, edge_sub_batch.batch)
            edge_x = F.dropout(F.relu(edge_x), p=self.dp, training=self.training) 
            
            copyed_edge_x = edge_x[node_sub_batch.edges_map]
            if self.convs:
                copyed_edge_x = self.E2N_covs[i](copyed_edge_x, edge_2)
            node_x = global_mean_pool(copyed_edge_x, node_sub_batch.batch)
            node_x = F.dropout(F.relu(node_x), p=self.dp, training=self.training) 
            
            node_x = torch.cat([xs, node_x], dim=1)
            
        return self.out(node_x)
    
    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss  