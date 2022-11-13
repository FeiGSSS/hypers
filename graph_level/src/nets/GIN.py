# -*- encoding: utf-8 -*-
'''
@File    :   gin_net.py
@Time    :   2022/11/11 10:23:39
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

# adopted from:
# https://github.com/graphdeeplearning/benchmarking-gnns/blob/b6c407712fa576e9699555e1e035d1e327ccae6c/nets/superpixels_graph_classification/gin_net.py#L16

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MLP, GINConv, BatchNorm
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

class GINLayer(nn.Module):
    def __init__(self, mlp, in_dim, out_dim, dropout, batch_norm,
                 residule=False, init_eps=0, learn_eps=False):
        super().__init__()
        self.conv = GINConv(mlp,
                            eps=init_eps,
                            train_eps=learn_eps)
        
        self.batch_norm = batch_norm
        self.residule = residule
        self.dropout = nn.Dropout(p=dropout)
        
        if in_dim != out_dim:
            self.residule = False
        
        self.bn_node_x = BatchNorm(out_dim)

    def forward(self, x, edge_index):
        x_in = x
        x = self.conv(x, edge_index)
        if self.batch_norm:
            x = self.bn_node_x(x)
        x = F.relu(x)
        if self.residule:
            x = x_in + x
        x = self.dropout(x)
        return x

class GINNet(nn.Module):
    
    def __init__(self, in_dim:int, hid_dim:int, num_classes:int, dropout:float,
                       num_layers:int=4, eps_train:bool=True, readout:str="mean", 
                       batch_norm:bool=True, residule:bool=True):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_x = nn.Linear(in_dim, hid_dim)
        
        self.ginlayers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            mlp = MLP(in_channels=hid_dim,
                      hidden_channels=hid_dim,
                      out_channels=hid_dim,
                      num_layers=2)
            self.ginlayers.append(GINLayer(mlp=mlp,
                                           in_dim=hid_dim,
                                           out_dim=hid_dim,
                                           dropout=dropout,
                                           batch_norm=batch_norm,
                                           residule=residule,
                                           init_eps=0,
                                           learn_eps=eps_train))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for _ in range(self.num_layers+1):
            self.linears_prediction.append(nn.Linear(hid_dim, num_classes))
        
        if readout == 'sum':
            self.pool = global_add_pool
        elif readout == 'mean':
            self.pool = global_mean_pool
        elif readout == 'max':
            self.pool = global_max_pool
        else:
            raise NotImplementedError
        
    def forward(self, data, **kwargs):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else None
        x = self.embedding_x(x)
        # list of hidden representation at each layer (including input)
        hidden_rep = [x]
        for i in range(self.num_layers):
            x = self.ginlayers[i](x, edge_index)
            hidden_rep.append(x)

        score_over_layer = 0
        # perform pooling over all nodes in each graph in every layer
        for i, x in enumerate(hidden_rep):
            pooled_x = self.pool(x, batch)
            score_over_layer += self.linears_prediction[i](pooled_x)

        return score_over_layer
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss