
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, GCN, SAGEConv, GATConv, GATv2Conv, GINConv, GCNConv
from torch_geometric.nn import Linear, MLP, LayerNorm, Sequential
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.aggr import GraphMultisetTransformer


class SHGNN(nn.Module):
    def __init__(self, heads, num_layers, dim, num_class, dp, convs:bool=False):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.if_convs = convs
        
        self.in_drp = nn.Dropout(0.)
        
        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()
        for i in range(num_layers):
            self.N2E_covs.append(GATConv(-1, dim, heads=heads))
            self.E2N_covs.append(GATConv(-1, dim, heads=heads))
            
            # self.N2E_covs.append(GAT(-1, dim, 1, dim, 0))
            # self.E2N_covs.append(GAT(-1, dim, 1, dim, 0))
            
            # self.N2E_covs.append(GINConv(Linear(-1, dim), train_eps=True))
            # self.E2N_covs.append(GINConv(Linear(-1, dim), train_eps=True))
            
            # hid_dim = 4*dim
            # self.EdgeUpdate.append(Sequential('x',[(LayerNorm(hid_dim), 'x->x'),
            #                                        (MLP([hid_dim, hid_dim, hid_dim]), 'x->mlp_x'),
            #                                        (lambda x,y: x+y, 'x, mlp_x->x'),
            #                                         (LayerNorm(hid_dim), 'x->x')]))
            
            # self.NodeUpdate.append(Sequential('x',[(LayerNorm(hid_dim), 'x->x'),
            #                                        (MLP([hid_dim, hid_dim, hid_dim]), 'x->mlp_x'),
            #                                        (lambda x,y: x+y, 'x, mlp_x->x'),
            #                                         (LayerNorm(hid_dim), 'x->x')]))
            
            self.EdgeUpdate.append(nn.Sequential(Linear(-1, dim),
                                                 LayerNorm(dim),
                                                 nn.ReLU(),
                                                 nn.Dropout(dp)))
            self.NodeUpdate.append(nn.Sequential(Linear(-1, dim),
                                                 LayerNorm(dim),
                                                 nn.ReLU(),
                                                 nn.Dropout(dp)))
            
            
        self.classifier = nn.Sequential(Linear(-1, num_class),
                                        nn.LogSoftmax(dim=-1))
        
    def forward(self, edge_sub_batch, node_sub_batch, node_x):
        node_x = self.in_drp(node_x)
        for i in range(self.num_layers):
            edge_x = []
            for eb in edge_sub_batch:
                copyed_node_x = node_x[eb.nodes_map]
                if self.if_convs:
                    copyed_node_x = self.N2E_covs[i](copyed_node_x, eb.edge_index)
                edge_x.append(global_mean_pool(copyed_node_x, eb.batch))
            edge_x = torch.cat(edge_x, dim=0)
            edge_x = self.EdgeUpdate[i](edge_x)
            
            node_x = []
            for nb in node_sub_batch:
                copyed_edge_x = edge_x[nb.edges_map]
                if self.if_convs:
                    copyed_edge_x = self.E2N_covs[i](copyed_edge_x, nb.edge_index)
                node_x.append(global_mean_pool(copyed_edge_x, nb.batch))
            node_x = torch.cat(node_x, dim=0)
            node_x = self.NodeUpdate[i](node_x)
        
        return self.classifier(node_x)
    
    def loss_fun(self, pred, labels):
        loss = F.nll_loss(pred, labels)
        return loss 
