import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, GCN, SAGEConv, GATConv, GATv2Conv, GINConv, GCNConv
from torch_geometric.nn import Linear, MLP, LayerNorm, Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops
from torch.nn import Parameter
from src.layers import PMA, MLP2


class SHGNN(nn.Module):
    def __init__(self, heads, pool, num_layers, dim, num_features, num_class, dp, type_gnn, convs):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.convs = convs
        self.pool = pool
        self.NormLayer = 'ln'
        self.in_emb = nn.Sequential(nn.Dropout(0.2))

        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        self.N2E_pooling = nn.ModuleList()
        self.E2N_pooling = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()
        self.type_gnn = type_gnn
        self.convs = convs

        for i in range(num_layers):
            if self.type_gnn == 'GIN':
                self.N2E_covs.append(GIN(-1, dim, 1, dim, dropout=0))
                self.E2N_covs.append(GIN(-1, dim, 1, dim, dropout=0))
            elif self.type_gnn == 'GAT':
                self.N2E_covs.append(GAT(-1, dim, 1, dim, dropout=0, v2=True))
                self.E2N_covs.append(GAT(-1, dim, 1, dim, dropout=0, v2=True))
            else:
                self.N2E_covs.append(GCN(-1, dim, 1, dim, dp, "relu", add_self_loops=True))
                self.E2N_covs.append(GCN(-1, dim, 1, dim, dp, "relu", add_self_loops=True))

            pma_in_dim = num_features if i == 0 else dim
            dim2 = dim * 2 if self.convs else dim
            self.N2E_pooling.append(PMA(pma_in_dim, dim, dim, heads=heads))
            self.E2N_pooling.append(PMA(dim2, dim, dim, heads=heads))  # * 2

            self.EdgeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 nn.Dropout(dp)))
            self.NodeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 nn.Dropout(dp)))

        # self.classifier = Linear(-1, num_class)  # F , nn.LogSoftmax(dim=-1)
        self.classifier = MLP2(in_channels=dim2,
                               hidden_channels=256,
                               out_channels=num_class,
                               num_layers=1,
                               dropout=self.dp,
                               Normalization=self.NormLayer,
                               InputNorm=False)  #  * 2

    def reset_parameters(self):
        for layer in self.N2E_pooling:
            layer.reset_parameters()
        for layer in self.E2N_pooling:
            layer.reset_parameters()
        # self.classifier.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, edge_sub_batch, node_sub_batch, node_x):
        # node_x = self.in_emb(node_x)
        node_x = F.dropout(node_x, p=0.2, training=self.training)
        for i in range(self.num_layers):
            edge_x = []
            for eb in edge_sub_batch:
                node2edge_map = torch.stack([eb.nodes_map, eb.batch], dim=0)
                sub_ex_pma = F.relu(self.N2E_pooling[i](node_x, node2edge_map))

                if self.convs:  # whether use gnn
                    copyed_node_x = node_x[eb.nodes_map]
                    sub_ex_convs = global_mean_pool(self.N2E_covs[i](copyed_node_x, eb.edge_index), eb.batch)
                    sub_ex = torch.cat([sub_ex_convs, sub_ex_pma], dim=1)
                    edge_x.append(sub_ex)
                else:
                    edge_x.append(sub_ex_pma)

            edge_x = torch.cat(edge_x, dim=0)
            # edge_x = self.EdgeUpdate[i](edge_x)
            edge_x = F.dropout(edge_x, p=self.dp, training=self.training)

            node_x = []
            for nb in node_sub_batch:
                edge2node_map = torch.stack([nb.edges_map, nb.batch], dim=0)
                sub_nx_pma = F.relu(self.E2N_pooling[i](edge_x, edge2node_map))

                if self.convs:  # whether use gnn
                    copyed_edge_x = edge_x[nb.edges_map]
                    sub_nx_convs = global_mean_pool(self.E2N_covs[i](copyed_edge_x, nb.edge_index), nb.batch)
                    sub_nx = torch.cat([sub_nx_convs, sub_nx_pma], dim=1)
                    node_x.append(sub_nx)
                else:
                    node_x.append(sub_nx_pma)

            node_x = torch.cat(node_x, dim=0)
            # node_x = self.NodeUpdate[i](node_x)
            node_x = F.dropout(node_x, p=self.dp, training=self.training)

        return self.classifier(node_x)

    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss
