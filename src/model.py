import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, GCN, SAGEConv, GATConv, GATv2Conv, GINConv, GCNConv
from torch_geometric.nn import Linear, BatchNorm
from torch_geometric.nn import global_mean_pool
from src.layers import PMA


class SHGNN(nn.Module):
    def __init__(self, heads, pool, num_layers, dim, num_features, num_class, dp, type_gnn, convs):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.convs = convs
        self.pool = pool
        self.NormLayer = 'ln'
        self.in_emb = nn.Sequential(BatchNorm(num_features),
                                    nn.Dropout(dp))

        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        self.N2E_pooling1 = nn.ModuleList()
        self.E2N_pooling1 = nn.ModuleList()
        self.N2E_pooling = nn.ModuleList()
        self.E2N_pooling = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()
        self.type_gnn = type_gnn
        self.convs = convs

        for i in range(num_layers):
            if self.type_gnn == 'GIN':
                self.N2E_covs.append(GIN(-1, dim, 1, dim, dropout=dp))
                self.E2N_covs.append(GIN(-1, dim, 1, dim, dropout=dp))
            elif self.type_gnn == 'GAT':
                self.N2E_covs.append(GATConv(-1, dim, 1, dropout=dp))
                self.E2N_covs.append(GATConv(-1, dim, 1, dropout=dp))
            else:
                self.N2E_covs.append(GCNConv(-1, dim, add_self_loops=True))
                self.E2N_covs.append(GCNConv(-1, dim, add_self_loops=True))
                

            dim1 = num_features if i == 0 else dim
            self.N2E_pooling.append(PMA(dim1, dim, dim, heads=heads))
            dim2 = dim * 2 if self.convs else dim
            self.E2N_pooling.append(PMA(dim2, dim, dim, heads=heads))

            self.EdgeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 nn.Dropout(dp)))
            self.NodeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 nn.Dropout(dp)))

        self.classifier = nn.Sequential(Linear(-1, num_class),
                                        nn.LogSoftmax(dim=-1))

    def reset_parameters(self):
        for layer in self.N2E_pooling:
            layer.reset_parameters()
        for layer in self.E2N_pooling:
            layer.reset_parameters()

    def forward(self, edge_sub_batch, node_sub_batch, node_x):
        node_x = self.in_emb(node_x)
        node_x_res = []
        for i in range(self.num_layers):
            edge_x = []
            for eb in edge_sub_batch:
                node2edge_map = torch.stack([eb.nodes_map, eb.batch], dim=0)
                sub_ex_pma = self.N2E_pooling[i](node_x, node2edge_map)

                if self.convs:  # whether use gnn
                    copyed_node_x = node_x[eb.nodes_map]
                    copyed_node_x = self.N2E_covs[i](copyed_node_x, eb.edge_index)
                    sub_ex_conv = global_mean_pool(copyed_node_x, eb.batch)
                    sub_ex = torch.cat([sub_ex_pma, sub_ex_conv], dim=1)
                    edge_x.append(sub_ex)
                else:
                    edge_x.append(sub_ex_pma)

            edge_x = torch.cat(edge_x, dim=0)
            edge_x = self.EdgeUpdate[i](edge_x)

            node_x = []
            for nb in node_sub_batch:
                edge2node_map = torch.stack([nb.edges_map, nb.batch], dim=0)
                sub_nx_pma = self.E2N_pooling[i](edge_x, edge2node_map)

                if self.convs:  # whether use gnn
                    copyed_edge_x = edge_x[nb.edges_map]
                    copyed_edge_x = self.E2N_covs[i](copyed_edge_x, nb.edge_index)
                    sub_nx_conv = global_mean_pool(copyed_edge_x, nb.batch)
                    sub_nx = torch.cat([sub_nx_pma, sub_nx_conv], dim=1)
                    node_x.append(sub_nx)
                else:
                    node_x.append(sub_nx_pma)

            node_x = torch.cat(node_x, dim=0)
            node_x = self.NodeUpdate[i](node_x)
            node_x_res.append(node_x)
        node_x = torch.cat(node_x_res, dim=1)

        return self.classifier(node_x)

    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss
