# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/10/31 13:08:17
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import networkx as nx
from networkx.algorithms import bipartite
from collections import namedtuple
from copy import deepcopy

import torch
import torch_geometric as tg
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Batch, Dataset
from torch_geometric.utils import k_hop_subgraph

def build_dual_subgraph(num_nodes:int, edges_list):
    """build the sequence of subgraphs of nodes and edges

    Args:
        hyperedges (List): _description_
    """
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
            weight = nodes_induced_graph.edges[u,v]["weight"]
            if weight>1: result_graph.add_edge(u,v)
        data = tg.utils.from_networkx(result_graph)
        data.orig_node_idx = torch.LongTensor(list(result_graph.nodes()))
        data.orig_edge_idx = int(e_name)
        edge_subgraph_list.append(data)
    
    node_subgraph_list = []
    for n_name in nodes_name:
        edges_to_n = list(nx.neighbors(B, n_name))
        edges_induced_graph = nx.subgraph(G_edges, edges_to_n)
        
        result_graph = nx.Graph()
        result_graph.add_nodes_from(edges_induced_graph.nodes())
        
        for (s,t) in edges_induced_graph.edges():
            weight = edges_induced_graph.edges[s,t]["weight"]
            if weight > 1:result_graph.add_edge(s, t)
        data = tg.utils.from_networkx(result_graph)
        data.orig_edge_idx = torch.LongTensor([int(e) for e in result_graph.nodes()])
        data.orig_node_idx = int(n_name)
        node_subgraph_list.append(data)
    
    return edge_subgraph_list, node_subgraph_list

######## load tu dataset ###########
def load_tu_dataset(data_name:str):
    dataset = TUDataset("../data/TU", name=data_name)
    edge_graph = []
    node_graph = []
    xs = []
    ys = []
    for graph in dataset:
        num_nodes = graph.num_nodes
        hyper_edge_list = []
        for ind in range(num_nodes):
            hyper_edge, _, _, _ = k_hop_subgraph(ind,
                                                 num_hops=1,
                                                 edge_index=graph.edge_index)
            hyper_edge = set(hyper_edge.tolist())
            hyper_edge_list.append(hyper_edge)
        edge_subgraph_list, node_subgraph_list = build_dual_subgraph(num_nodes,
                                                                     hyper_edge_list)
        edge_subgraph = Batch.from_data_list(edge_subgraph_list)
        edge_subgraph.orig_num_hedges = len(edge_subgraph_list)
        edge_subgraph.node_to_hedge = edge_subgraph.batch
        del edge_subgraph.batch
        
        node_subgraph = Batch.from_data_list(node_subgraph_list)
        node_subgraph.orig_num_nodes = len(node_subgraph_list)
        node_subgraph.hedge_to_node = node_subgraph.batch
        del node_subgraph.batch
        
        edge_graph.append(edge_subgraph)
        node_graph.append(node_subgraph)
        
        xs.append(graph.x)
        ys.append(graph.y)
    return edge_graph, node_graph, xs, ys

class GraphDataset(Dataset):
    def __init__(self, data_name:str):
        super(GraphDataset, self).__init__()
        edge_graph, node_graph, xs, ys = load_tu_dataset(data_name)
        self.edge_graph = edge_graph
        self.node_graph = node_graph
        self.xs = xs
        self.y = torch.cat(ys)
        self.size = len(edge_graph)
    
    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return self._infer_num_classes(self.y)
    
    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        x = torch.cat(self.xs, dim=0)
        return x.shape[1]
    
    def len(self):
        return self.size
    
    def get(self, index:int):
        eg = self.edge_graph[index]
        ng = self.node_graph[index]
        x = self.xs[index]
        y = self.y[index]
        return (eg, ng, x, y)
    

def collate(bunch):
    edge_graph, node_graph, xs, ys, batch = [], [], [], [], []
    cum_nodes = 0
    cum_edges = 0
    for cnt, item in enumerate(bunch):
        # collate is in-place function, therefore, we 
        # have to copy the data
        eg, ng, x, y = [deepcopy(i) for i in item]
        
        eg.orig_node_idx += cum_nodes
        eg.orig_edge_idx += cum_edges
        eg.node_to_hedge += cum_edges
        
        ng.orig_edge_idx += cum_edges
        ng.orig_node_idx += cum_nodes
        ng.hedge_to_node += cum_nodes
        
        cum_edges += eg.orig_num_hedges
        cum_nodes += ng.orig_num_nodes
        
        edge_graph.append(eg)
        node_graph.append(ng)
        xs.append(x)
        ys.append(y)
        
        batch.extend([cnt]*ng.orig_num_nodes)
    
    batch = (Batch.from_data_list(edge_graph),
                 Batch.from_data_list(node_graph),
                 torch.cat(xs, dim=0),
                 torch.tensor(ys),
                 torch.LongTensor(batch))
    return batch