from copy import copy
import os.path as osp
import pickle as pkl
from typing import List
from unittest import result
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite

import torch
from torch_scatter import scatter
import torch_geometric as tg

def build_dual_subgraph(num_nodes:int, edges_list, add_self_loop:bool=True):
    """build the sequence of subgraphs of nodes and edges

    Args:
        hyperedges (List): _description_
    """
    assert isinstance(num_nodes, int) and num_nodes>0
    if add_self_loop:
        for node in range(num_nodes):
            edges_list.append(set([node]))
    num_edges = len(edges_list)
    
    nodes_name = list(range(num_nodes))
    edges_name = [str(e) for e in range(num_edges)]
    
    B = nx.Graph()
    B.add_nodes_from(edges_name, bipartite=0)
    B.add_nodes_from(nodes_name, bipartite=1)
    for eid, edge in enumerate(edges_list):
        for n in edge:
            B.add_edge(str(eid), n)
    
    G_nodes = bipartite.weighted_projected_graph(B, nodes_name)
    G_edges = bipartite.weighted_projected_graph(B, edges_name)
    
    edge_subgraph_list = []
    for e_name in edges_name:
        nodes_in_e = list(nx.neighbors(B, e_name))
        nodes_induced_graph = nx.subgraph(G_nodes, nodes_in_e)
        
        result_graph = nx.Graph()
        result_graph.add_nodes_from(nodes_induced_graph.nodes())
        
        for (u,v) in nodes_induced_graph.edges():
            if nodes_induced_graph.edges[u,v]["weight"] > 1:
                result_graph.add_edge(u,v)
        data = tg.utils.from_networkx(result_graph)
        data.nodes_map = torch.LongTensor(list(result_graph.nodes()))
        data.edge_name = int(e_name)
        # nodes_idx maps the data back to original graph
        # edge_name stores the target of this subgraph
        edge_subgraph_list.append(data)
    
    node_subgraph_list = []
    for n_name in nodes_name:
        edges_to_n = list(nx.neighbors(B, n_name))
        edges_induced_graph = nx.subgraph(G_edges, edges_to_n)
        
        result_graph = nx.Graph()
        result_graph.add_nodes_from(edges_induced_graph.nodes())
        
        for (s,t) in edges_induced_graph.edges():
            if edges_induced_graph.edges[s,t]["weight"] > 1:
                result_graph.add_edge(s, t)
        data = tg.utils.from_networkx(result_graph)
        data.edges_map = torch.LongTensor([int(e) for e in result_graph.nodes()])
        data.node_name = int(n_name)
        node_subgraph_list.append(data)
    
    edge_sub_batch = tg.data.Batch.from_data_list(edge_subgraph_list)
    node_sub_batch = tg.data.Batch.from_data_list(node_subgraph_list)
    
    return edge_sub_batch, node_sub_batch
    

def load_citation_dataset(path:str,
                          dataset:str='cora'):

    # first load node features:
    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pkl.load(f)
        features = features.todense()
        features = torch.Tensor(features).float()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pkl.load(f)
        labels = torch.Tensor(labels).long()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pkl.load(f)

    hyperedges = list(hypergraph.values())
    edge_sub_batch, node_sub_batch = build_dual_subgraph(num_nodes,
                                                         hyperedges,
                                                         True)
    
    return edge_sub_batch, node_sub_batch, features, labels

class MyDataset():
    def __init__(self, path:str, dataset:str):
        data = load_citation_dataset(path, dataset)
        self.edge_sub_batch = data[0]
        self.node_sub_batch = data[1]
        self.features = data[2]
        self.labels = data[3]
        
        self.num_class = int(torch.max(self.labels) + 1)
        self.split = self.rand_split_labels(self.labels)
    
    def to(self, device):
        self.edge_sub_batch = self.edge_sub_batch.to(device)
        self.node_sub_batch = self.node_sub_batch.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        return self
        
        
    
    def rand_split_labels(self, label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
        """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
        """ randomly splits label into train/valid/test splits """
        if not balance:
            if ignore_negative:
                labeled_nodes = torch.where(label != -1)[0]
            else:
                labeled_nodes = label

            n = labeled_nodes.shape[0]
            train_num = int(n * train_prop)
            valid_num = int(n * valid_prop)

            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            val_indices = perm[train_num:train_num + valid_num]
            test_indices = perm[train_num + valid_num:]

            if not ignore_negative:
                return train_indices, val_indices, test_indices

            train_idx = labeled_nodes[train_indices]
            valid_idx = labeled_nodes[val_indices]
            test_idx = labeled_nodes[test_indices]

            split_idx = {'train': train_idx,
                        'valid': valid_idx,
                        'test': test_idx}
        else:
            #         ipdb.set_trace()
            indices = []
            for i in range(label.max()+1):
                index = torch.where((label == i))[0].view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

            percls_trn = int(train_prop/(label.max()+1)*len(label))
            val_lb = int(valid_prop*len(label))
            train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
            rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
            rest_index = rest_index[torch.randperm(rest_index.size(0))]
            valid_idx = rest_index[:val_lb]
            test_idx = rest_index[val_lb:]
            split_idx = {'train': train_idx,
                        'valid': valid_idx,
                        'test': test_idx}
        return split_idx
        