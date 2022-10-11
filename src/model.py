import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, GCN, SAGEConv, GATConv, GATv2Conv, GINConv, GCNConv
from torch_geometric.nn import Linear, MLP, LayerNorm, Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops
from torch.nn import Parameter
from src.layers import PMA


class SHGNN(nn.Module):
    def __init__(self, heads, pool, num_layers, dim, num_features, num_class, dp, convs: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.convs = convs
        self.pool = pool
        self.in_emb = nn.Sequential(nn.Dropout(0.2))
        
        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        self.N2E_pooling = nn.ModuleList()
        self.E2N_pooling = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()
        
        for i in range(num_layers):
            self.N2E_covs.append(GCN(-1, dim, 1, dim, dp, "relu", add_self_loops=True))
            self.E2N_covs.append(GCN(-1, dim, 1, dim, dp, "relu", add_self_loops=True))
            
            pma_in_dim = num_features if i==0 else dim
            self.N2E_pooling.append(PMA(pma_in_dim, dim, dim, heads=heads))
            self.E2N_pooling.append(PMA(dim*2, dim, dim, heads=heads))
            
            self.EdgeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 nn.Dropout(dp)))
            self.NodeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 nn.Dropout(dp)))
            
            
        self.classifier = nn.Sequential(Linear(-1, num_class),
                                        nn.LogSoftmax(dim=-1))
        
    def forward(self, edge_sub_batch, node_sub_batch, node_x):
        node_x = self.in_emb(node_x)
        for i in range(self.num_layers):
            edge_x = []
            for eb in edge_sub_batch:
                copyed_node_x = node_x[eb.nodes_map]
                sub_ex_convs = global_mean_pool(self.N2E_covs[i](copyed_node_x,  eb.edge_index),
                                                eb.batch)   # change into SetGNN.PMA
                
                node2edge_map = torch.stack([torch.LongTensor(range(len(eb.batch))).to(eb.batch.device), eb.batch], dim=0)
                sub_ex_pma = self.N2E_pooling[i](copyed_node_x, node2edge_map)
                
                sub_ex = torch.cat([sub_ex_convs, sub_ex_pma], dim=1)
                edge_x.append(sub_ex)
                
            edge_x = torch.cat(edge_x, dim=0)
            edge_x = self.EdgeUpdate[i](edge_x)

            node_x = []
            for nb in node_sub_batch:
                copyed_edge_x = edge_x[nb.edges_map]
                sub_nx_convs = global_mean_pool(self.E2N_covs[i](copyed_edge_x, nb.edge_index),
                                          nb.batch)
                
                edge2node_map = torch.stack([torch.LongTensor(range(len(nb.batch))).to(nb.batch.device), nb.batch], dim=0)
                sub_nx_pma = self.E2N_pooling[i](copyed_edge_x, edge2node_map)
                
                sub_nx = torch.cat([sub_nx_convs, sub_nx_pma], dim=1)
                node_x.append(sub_nx)
                
            node_x = torch.cat(node_x, dim=0)
            node_x = self.NodeUpdate[i](node_x)

        return self.classifier(node_x)
    
    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss 
