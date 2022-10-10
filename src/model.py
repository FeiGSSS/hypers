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
    def __init__(self, heads, num_layers, dim, num_features, num_class, dp, convs: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.convs = convs
        self.in_emb = nn.Dropout(0.2)
        
        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        self.N2E_pooling = nn.ModuleList()
        self.E2N_pooling = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()
        
        for i in range(num_layers):
            self.N2E_covs.append(GCNConv(-1, dim))
            self.E2N_covs.append(GCNConv(-1, dim))
            if not self.convs and i==0:
                self.N2E_pooling.append(PMA(num_features, dim, dim, 1, heads=heads))
            else:
                self.N2E_pooling.append(PMA(dim, dim, dim, 1, heads=heads))
            self.E2N_pooling.append(PMA(dim, dim, dim, 1, heads=heads))  # feanture=dim=dim
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
                # eb_edge_index = add_self_loops(eb.edge_index, num_nodes=eb.num_nodes)[0]
                eb_edge_index = eb.edge_index
                if self.convs:
                    copyed_node_x = self.N2E_covs[i](copyed_node_x, eb_edge_index)  # subgraph-GNN
                # sub_ex = global_mean_pool(copyed_node_x, eb.batch)   # change into SetGNN.PMA
                node2edge_map = torch.stack([torch.LongTensor(range(len(eb.batch))).to(eb.batch.device), eb.batch], dim=0)
                sub_ex = self.N2E_pooling[i](copyed_node_x, node2edge_map)
                edge_x.append(sub_ex)
            edge_x = torch.cat(edge_x, dim=0)
            edge_x = self.EdgeUpdate[i](edge_x)

            node_x = []
            for nb in node_sub_batch:
                copyed_edge_x = edge_x[nb.edges_map]
                # nb_edge_index = add_self_loops(nb.edge_index, num_nodes=nb.num_nodes)[0]  # add self_loop
                nb_edge_index = nb.edge_index
                if self.convs:
                    copyed_edge_x = self.E2N_covs[i](copyed_edge_x, nb_edge_index)
                # sub_nx = global_mean_pool(copyed_edge_x, nb.batch)
                edge2node_map = torch.stack([torch.LongTensor(range(len(nb.batch))).to(nb.batch.device), nb.batch], dim=0)
                sub_nx = self.E2N_pooling[i](copyed_edge_x, edge2node_map)
                node_x.append(sub_nx)
            node_x = torch.cat(node_x, dim=0)
            node_x = self.NodeUpdate[i](node_x)

        return self.classifier(node_x)
    
    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss 
