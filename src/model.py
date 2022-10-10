import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, GCN, SAGEConv, GATConv
from torch_geometric.nn import Linear, MLP
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops
from torch.nn import Parameter
from src.layers import PMA


class SHGNN(nn.Module):
    def __init__(self, num_layers, dim, num_features, num_class, dp, convs: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.if_convs = convs

        self.in_emb = nn.Sequential(Linear(-1, 2 * dim),
                                    nn.ReLU(),
                                    Linear(-1, dim))

        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()

        self.N2E_pooling = nn.ModuleList()
        self.E2N_pooling = nn.ModuleList()
        for _ in range(num_layers):
            # self.N2E_covs.append(GIN(-1, dim, 1, dim, dropout=0))
            # self.E2N_covs.append(GIN(-1, dim, 1, dim, dropout=0))
            # self.N2E_covs.append(GAT(-1, dim, 1, dim, dropout=0, v2=True))
            # self.E2N_covs.append(GAT(-1, dim, 1, dim, dropout=0, v2=True))
            self.N2E_covs.append(GCN(-1, dim, 1, dim, dropout=0))
            self.E2N_covs.append(GCN(-1, dim, 1, dim, dropout=0))

            self.EdgeUpdate.append(MLP(in_channels=-1, hidden_channels=2 * dim, out_channels=dim, num_layers=1))
            self.NodeUpdate.append(MLP(in_channels=-1, hidden_channels=dim, out_channels=dim, num_layers=1))

            self.N2E_pooling.append(PMA(num_features + _*64, 64, 64, 2, heads=1))
            self.E2N_pooling.append(PMA(64, 64, 64, 2, heads=1))  # feanture=dim=64

        self.out = nn.Sequential(Linear(-1, 2 * dim),
                                 nn.ReLU(),
                                 Linear(-1, num_class),
                                 nn.LogSoftmax(dim=-1))

    def forward(self, edge_sub_batch, node_sub_batch, node_x):
        node_x = F.dropout(node_x, p=self.dp, training=self.training)  # Input dropout
        for i in range(self.num_layers):
            xs = node_x

            edge_x = []
            for eb in edge_sub_batch:
                copyed_node_x = node_x[eb.nodes_map]
                eb_edge_index = add_self_loops(eb.edge_index, num_nodes=eb.num_nodes)[0]
                if self.if_convs:
                    copyed_node_x = self.N2E_covs[i](copyed_node_x, eb_edge_index)  # subgraph-GNN
                # sub_ex = global_mean_pool(copyed_node_x, eb.batch)   # change into SetGNN.PMA
                ind_batch_eb = eb_edge_index[0].clone()
                for j in range(len(ind_batch_eb)):
                    ind_batch_eb[j] = eb.batch[ind_batch_eb[j]]
                edge_index_N2E = torch.stack([eb_edge_index[0], ind_batch_eb])
                sub_ex = self.N2E_pooling[i](copyed_node_x, edge_index_N2E)
                edge_x.append(sub_ex)
            edge_x = F.dropout(F.relu(torch.cat(edge_x, dim=0)),
                               p=self.dp,
                               training=self.training)
            # edge_x = self.EdgeUpdate[i](edge_x)

            node_x = []
            for nb in node_sub_batch:
                copyed_edge_x = edge_x[nb.edges_map]
                nb_edge_index = add_self_loops(nb.edge_index, num_nodes=nb.num_nodes)[0]  # add self_loop
                if self.if_convs:
                    copyed_edge_x = self.E2N_covs[i](copyed_edge_x, nb_edge_index)
                # sub_nx = global_mean_pool(copyed_edge_x, nb.batch)
                ind_batch_nb = nb_edge_index[0].clone()
                for j in range(len(ind_batch_nb)):
                    ind_batch_nb[j] = nb.batch[ind_batch_nb[j]]
                edge_index_E2N = torch.stack([nb_edge_index[0], ind_batch_nb])
                sub_nx = self.E2N_pooling[i](copyed_edge_x, edge_index_E2N)
                node_x.append(sub_nx)
            node_x = F.dropout(F.relu(torch.cat(node_x, dim=0)),
                               p=self.dp,
                               training=self.training)
            node_x = torch.cat([xs, node_x], dim=1)
            # node_x = self.NodeUpdate[i](node_x)

        return self.out(node_x)

    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss


