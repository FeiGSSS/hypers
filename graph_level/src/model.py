import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GIN, GAT, GCN, SAGEConv, GATConv, GATv2Conv, GINConv, GCNConv
from torch_geometric.nn import Linear, BatchNorm, LayerNorm
from torch_geometric.nn import global_mean_pool
from src.layers import PMA


class SHGNN(nn.Module):
    def __init__(self, num_layers, num_features, dim, dp, type_gnn, convs, heads, num_classes):
        super().__init__()
        self.num_layers = num_layers
        self.dp = dp
        self.convs = convs

        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        self.N2E_pooling = nn.ModuleList()
        self.E2N_pooling = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()
        self.type_gnn = type_gnn
        self.convs = convs
        
        self.ln_e = LayerNorm(dim)
        self.ln_n = LayerNorm(dim)

        for i in range(num_layers):
            if self.type_gnn == 'gat':
                assert dim % heads == 0
                self.N2E_covs.append(GATConv(-1, dim//heads, heads, concat=True))
                self.E2N_covs.append(GATConv(-1, dim//heads, heads, concat=True))
            elif self.type_gnn == 'gin':
                self.N2E_covs.append(GIN(-1, dim, 2, dim))
                self.E2N_covs.append(GIN(-1, dim, 2, dim))
            elif self.type_gnn == "gcn":
                self.N2E_covs.append(GCNConv(-1, dim, add_self_loops=True))
                self.E2N_covs.append(GCNConv(-1, dim, add_self_loops=True))
                
            if i==0:
                dim1 = num_features
            elif self.convs == "both":
                dim1 = 2*dim
            else:
                dim1 = dim
            self.N2E_pooling.append(PMA(dim1, dim, dim, heads=heads))
            
            dim2 = dim*2 if self.convs == "both" else dim
            self.E2N_pooling.append(PMA(dim2, dim, dim, heads=heads))

            self.EdgeUpdate.append(nn.Sequential(nn.ELU(),
                                                 nn.Dropout(dp)))
            self.NodeUpdate.append(nn.Sequential(nn.ELU(),
                                                 nn.Dropout(dp)))
        self.lin1 = Linear(-1, dim)
        self.lin2 = Linear(-1, num_classes)
        self.reset_parameters()
        

    def reset_parameters(self):
        layers = []
        for layer in self.N2E_pooling:
            layers.append(layer)
        for layer in self.E2N_pooling:
            layers.append(layer)
        for layer in self.N2E_covs:
            layers.append(layer)
        for layer in self.E2N_covs:
            layers.append(layer)
        for la in layers:
            if not isinstance(la, nn.Dropout):
                la.reset_parameters()
        self.ln_e.reset_parameters()
        self.ln_n.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def layer_forward(self, layer_ind, edge_graph, node_graph, node_x):

        if self.convs == "gnn":  # whether use gnn
            _node_x = self.N2E_covs[layer_ind](node_x[edge_graph.orig_node_idx],
                                                edge_graph.edge_index)
            # print(node_x.shape, edge_graph.orig_node_idx.max())
            edge_x = global_mean_pool(_node_x, edge_graph.node_to_hedge)
            edge_x = self.ln_e(edge_x)
            
        elif self.convs == "pma":
            node2edge_map = torch.stack([edge_graph.orig_node_idx,
                                         edge_graph.node_to_hedge], dim=0)
            edge_x = self.N2E_pooling[layer_ind](node_x, node2edge_map)
            
        elif self.convs == "both":
            node2edge_map = torch.stack([edge_graph.orig_node_idx,
                                         edge_graph.node_to_hedge], dim=0)
            edge_x_pma = self.N2E_pooling[layer_ind](node_x, node2edge_map)
            
            _node_x = self.N2E_covs[layer_ind](node_x[edge_graph.orig_node_idx],
                                              edge_graph.edge_index)
            edge_x_gnn = global_mean_pool(_node_x, edge_graph.node_to_hedge)
            edge_x_gnn = self.ln_e(edge_x_gnn)
            
            edge_x = torch.cat([edge_x_pma, edge_x_gnn], dim=1)
            
        edge_x = self.EdgeUpdate[layer_ind](edge_x)

        if self.convs == "gnn":  # whether use gnn
            _edge_x = edge_x[node_graph.orig_edge_idx]
            _edge_x = self.E2N_covs[layer_ind](_edge_x,
                                               node_graph.edge_index)
            node_x = global_mean_pool(_edge_x, node_graph.hedge_to_node)
            node_x = self.ln_n(node_x)
            
        elif self.convs == "pma":
            edge2node_map = torch.stack([node_graph.orig_edge_idx,
                                         node_graph.hedge_to_node], dim=0)
            node_x = self.E2N_pooling[layer_ind](edge_x, edge2node_map)
            
        elif self.convs == "both":
            _edge_x = edge_x[node_graph.orig_edge_idx]
            _edge_x = self.E2N_covs[layer_ind](_edge_x,
                                               node_graph.edge_index)
            node_x_gnn = global_mean_pool(_edge_x, node_graph.hedge_to_node)
            node_x_gnn = self.ln_n(node_x_gnn)
            
            edge2node_map = torch.stack([node_graph.orig_edge_idx,
                                         node_graph.hedge_to_node], dim=0)
            node_x_pma = self.E2N_pooling[layer_ind](edge_x, edge2node_map)
            
            node_x = torch.cat([node_x_gnn, node_x_pma], dim=1)

        node_x = self.NodeUpdate[layer_ind](node_x)
        
        return node_x

    def forward(self, data):
        node_x = data.x
        edge_graph, node_graph = data.edge_graph, data.node_graph
        node_x = self.layer_forward(0, edge_graph, node_graph, node_x)
        node_xs = [node_x]
        for i in range(1, self.num_layers):
            node_x = self.layer_forward(i, edge_graph, node_graph, node_x)
            node_xs += [node_x]
        node_xs = torch.cat(node_xs, dim=1)
        # pooling to graphs
        graph_xs = global_mean_pool(node_xs, data.node_to_graph)
        graph_xs = F.relu(self.lin1(graph_xs))
        graph_xs = F.dropout(graph_xs, p=0.5, training=self.training)
        graph_xs = self.lin2(graph_xs)
        return F.log_softmax(graph_xs, dim=-1)

    
    def __repr__(self):
        return self.__class__.__name__