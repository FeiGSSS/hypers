# -*- encoding: utf-8 -*-
'''
@File    :   hgnn_net.py
@Time    :   2022/11/11 16:41:00
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import torch
import torch.nn as nn
from torch.nn import Dropout
from torch_geometric.nn import GIN
from torch_geometric.nn import Linear, BatchNorm
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from src.nets.layers import PMA


class SHGNN(nn.Module):
    def __init__(self, in_dim:int, hid_dim:int, dropout:float,
                 num_layers:int, inner_num_layers:int, hyper_heads:int,
                 inner_gnn:str, hyper_convs:str, num_classes:int,
                 readout:str, inner_readout:str, batch_norm:bool):
        super().__init__()
        assert hid_dim % hyper_heads == 0, "hid_dim must be multiple of hyper_heads"
        self.num_layers = num_layers
        self.dropout = Dropout(dropout)
        self.hyper_convs = hyper_convs

        self.embedding_x = nn.Linear(in_dim, hid_dim)
        
        self.N2E_pooling = nn.ModuleList()
        self.E2N_pooling = nn.ModuleList()
        self.EdgeUpdate = nn.ModuleList()
        self.NodeUpdate = nn.ModuleList()
        
        self.N2E_covs = nn.ModuleList()
        self.E2N_covs = nn.ModuleList()
        for i in range(num_layers):
            # inner gnn layers
            if inner_gnn == 'GIN':
                self.N2E_covs.append(GIN(-1, hid_dim, inner_num_layers, hid_dim, dropout, norm="batch_norm"))
                self.E2N_covs.append(GIN(-1, hid_dim, inner_num_layers, hid_dim, dropout, norm="batch_norm"))
            else:
                raise NotImplementedError
            
            # pma pooling layers
            dim1 = 2*hid_dim if self.hyper_convs == "both" and i!=0 else hid_dim
            dim2 = 2*hid_dim if self.hyper_convs == "both" else hid_dim
            
            self.N2E_pooling.append(PMA(dim1, hid_dim, hid_dim, heads=hyper_heads))
            self.E2N_pooling.append(PMA(dim2, hid_dim, hid_dim, heads=hyper_heads))

            # updating layers
            if batch_norm:
                bn_dim = hid_dim if self.hyper_convs != "both" else 2*hid_dim
                self.bn = BatchNorm(bn_dim)
                
            self.EdgeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 Dropout(dropout)))
            self.NodeUpdate.append(nn.Sequential(nn.ReLU(),
                                                 Dropout(dropout)))
        if inner_readout == 'sum':
            self.inner_pool = global_add_pool
        elif inner_readout == 'mean':
            self.inner_pool = global_mean_pool
        elif inner_readout == 'max':
            self.inner_pool = global_max_pool
        else:
            raise NotImplementedError
        
        if readout == 'sum':
            self.pool = global_add_pool
        elif readout == 'mean':
            self.pool = global_mean_pool
        elif readout == 'max':
            self.pool = global_max_pool
        else:
            raise NotImplementedError
        
        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for _ in range(self.num_layers+1):
            self.linears_prediction.append(Linear(-1, num_classes))
    
    def layer_forward(self, layer_ind, data, node_x):

        if self.hyper_convs == "gnn":  # whether use gnn
            _node_x = node_x[data.ori_node_idx]
            _node_x = self.N2E_covs[layer_ind](_node_x, data.edge_index_N)
            edge_x = self.inner_pool(_node_x, data.node2edge)
            
        elif self.hyper_convs == "pma":
            node2edge_index = torch.stack([data.ori_node_idx, data.node2edge], dim=0)
            edge_x = self.N2E_pooling[layer_ind](node_x, node2edge_index)
            
        elif self.hyper_convs == "both":
            node2edge_index = torch.stack([data.ori_node_idx, data.node2edge], dim=0)
            edge_x_pma = self.N2E_pooling[layer_ind](node_x, node2edge_index)
            
            _node_x = node_x[data.ori_node_idx]
            _node_x = self.N2E_covs[layer_ind](_node_x, data.edge_index_N)
            edge_x_gnn = self.inner_pool(_node_x, data.node2edge)
            
            
            edge_x = torch.cat([edge_x_pma, edge_x_gnn], dim=1)
            
        edge_x = self.EdgeUpdate[layer_ind](edge_x)

        if self.hyper_convs == "gnn":  # whether use gnn
            _edge_x = edge_x[data.ori_edge_idx]
            _edge_x = self.E2N_covs[layer_ind](_edge_x, data.edge_index_E)
            node_x = self.inner_pool(_edge_x, data.edge2node)
            
        elif self.hyper_convs == "pma":
            edge2node_index = torch.stack([data.ori_edge_idx, data.edge2node], dim=0)
            node_x = self.E2N_pooling[layer_ind](edge_x, edge2node_index)
            
        elif self.hyper_convs == "both":
            _edge_x = edge_x[data.ori_edge_idx]
            _edge_x = self.E2N_covs[layer_ind](_edge_x, data.edge_index_E)
            node_x_gnn = self.inner_pool(_edge_x, data.edge2node)
            
            edge2node_index = torch.stack([data.ori_edge_idx, data.edge2node], dim=0)
            node_x_pma = self.E2N_pooling[layer_ind](edge_x, edge2node_index)
            
            node_x = torch.cat([node_x_gnn, node_x_pma], dim=1)

        node_x = self.NodeUpdate[layer_ind](node_x)
        
        return node_x

    def forward(self, data):
        node_x = self.embedding_x(data.x_N)
        xs = [node_x]
        for i in range(self.num_layers):
            node_x = self.layer_forward(i, data, node_x)
            xs += [node_x]
        
        score_over_layer = 0
        # perform pooling over all nodes in each graph in every layer
        for i, x in enumerate(xs):
            pooled_x = self.pool(x[data.ori_node_idx], data.batch)
            score_over_layer += self.linears_prediction[i](pooled_x)
        
        return score_over_layer
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss