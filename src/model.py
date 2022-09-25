
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, GCN, SAGEConv, GATConv
from torch_geometric.nn import Linear, MLP
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops


class SHGNN(nn.Module):
    def __init__(self, num_layers, dim, num_class, dp, convs:bool=False):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.if_convs = convs
        
        self.in_emb = nn.Sequential(Linear(-1, 2*dim),
                                 nn.ReLU(),
                                 Linear(-1, dim))
        
        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        for _ in range(num_layers):
            # self.N2E_covs.append(GIN(-1, dim, 1, dim, dropout=dp))
            # self.E2N_covs.append(GIN(-1, dim, 1, dim, dropout=dp))
            self.N2E_covs.append(GAT(-1, dim, 1, dim, dropout=dp, v2=True))
            self.E2N_covs.append(GAT(-1, dim, 1, dim, dropout=dp, v2=True))
            # self.N2E_covs.append(GCN(-1, dim, 1, dim, dropout=dp))
            # self.E2N_covs.append(GCN(-1, dim, 1, dim, dropout=dp))
        
        self.out = nn.Sequential(Linear(-1, 2*dim),
                                 nn.ReLU(),
                                 Linear(-1, num_class),
                                 nn.LogSoftmax(dim=-1))
        
    def forward(self, edge_sub_batch, node_sub_batch, node_x):
        node_x = F.dropout(node_x, p=self.dp, training=self.training) # Input dropout
        
        for i in range(self.num_layers):
            xs = node_x
            
            edge_x = []
            for eb in edge_sub_batch:
                copyed_node_x = node_x[eb.nodes_map]
                if self.if_convs:
                    try:
                        copyed_node_x = self.N2E_covs[i](copyed_node_x, eb.edge_index)
                    except Exception as e:
                        print(e)
                        print(eb.edge_index.max(), eb.edge_index.min())
                        print(copyed_node_x.shape)
                        exit()
                sub_ex = global_mean_pool(copyed_node_x, eb.batch)
                sub_ex = F.dropout(F.relu(sub_ex), p=self.dp, training=self.training)
                edge_x.append(sub_ex)
            edge_x = torch.cat(edge_x, dim=0)
            
            node_x = []
            for nb in node_sub_batch:
                copyed_edge_x = edge_x[nb.edges_map]
                if self.if_convs:
                    copyed_edge_x = self.E2N_covs[i](copyed_edge_x, nb.edge_index)
                sub_nx = global_mean_pool(copyed_edge_x, nb.batch)
                sub_nx = F.dropout(F.relu(sub_nx), p=self.dp, training=self.training) 
                node_x.append(sub_nx)
            node_x = torch.cat(node_x, dim=0)
            
            node_x = torch.cat([xs, node_x], dim=1)
            
        return self.out(node_x)
    
    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss  