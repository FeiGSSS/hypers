# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/10/31 13:08:17
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import os
import os.path as osp
import time
import networkx as nx
import pickle as pkl
from itertools import combinations
from networkx.algorithms import bipartite
from sklearn.model_selection import StratifiedKFold

import torch
import torch_geometric as tg
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph

def build_dual_subgraph(num_nodes:int, edges_list):
    """build the sequence of subgraphs of nodes and edges
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
        result_graph = nx.Graph()
        result_graph.add_nodes_from(B.neighbors(e_name))
        
        adj_es = G_edges.neighbors(e_name)
        for adj_e_name in adj_es:
            cross_nodes = list(nx.common_neighbors(B, e_name, adj_e_name))
            result_graph.add_edges_from(combinations(cross_nodes, 2))

        data = tg.utils.from_networkx(result_graph)
        data.ori_node_idx = torch.LongTensor(list(result_graph.nodes()))
        data.ori_edge_idx = int(e_name)
        edge_subgraph_list.append(data)
    
    node_subgraph_list = []
    for n_name in nodes_name:
        result_graph = nx.Graph()
        result_graph.add_nodes_from(B.neighbors(n_name))
        
        adj_ns = G_nodes.neighbors(n_name)
        for adj_n_name in adj_ns:
            cross_edges = list(nx.common_neighbors(B, n_name, adj_n_name))
            result_graph.add_edges_from(combinations(cross_edges, 2))

        data = tg.utils.from_networkx(result_graph)
        data.ori_edge_idx = torch.LongTensor([int(e) for e in result_graph.nodes()])
        data.ori_node_idx = int(n_name)
        node_subgraph_list.append(data)
    return edge_subgraph_list, node_subgraph_list

class PairData(Data):
    def __init__(self, x_N=None, 
                       edge_index_N=None,
                       node2edge=None,
                       edge_index_E=None,
                       edge2node=None,
                       ori_node_idx=None,
                       ori_edge_idx=None,
                       num_ori_nodes=None,
                       num_ori_edges=None,
                       num_dup_nodes=None,
                       num_dup_edges=None,
                       y=None):
        """_summary_

        Args:
            x_N (_type_): is the original node features
            edge_index_N (_type_): is the edge_index of batched hyper-edge-graphs, i.e., nodes are lifted 
                                original nodes, and a graph/data indicates one original hyperedge.
            node2edge (_type_): is the batch_index, as the pooling mapping from node to hyperedges 
                                e.g., [0,0,0,1,1,1,1] indicates the first three lifted nodes belong to
                                hyperedge 0, and the next four lifted nodes belongs to hyeredge 1.
            edge_index_E (_type_): is the edge_index of batched hyper-node-graphs, i.e., nodes are the dual
                                hyperedges in the original hypergraph, and the graph/data indixates the
                                dual nodes.
            edge2node (_type_): is the batch-index, as the pooling mapping from hyperedges to node.
                                Similar to node2edge.
            ori_node_idx (_type_): the original node indexes of the lifted nodes, used to genetate the node
                                features of the lifted nodes.
            ori_edge_idx (_type_): similar to ori_node_idx
            num_ori_nodes (_type_): the original number of nodes of each graph/data, corresponding to the
                                number of pseudo hyperedges.
            num_ori_edges (_type_): the original number of edges of each graph/data, corresponding to the
                                number of nodes.
            num_dup_nodes (_type_): the number of lifted nodes
            num_dup_edges (_type_): the number of hyperedges
            y (_type_): _description_
        """
        
        super().__init__()
        self.x_N = x_N
        self.edge_index_N = edge_index_N
        self.edge_index_E = edge_index_E
        self.ori_node_idx = ori_node_idx
        self.ori_edge_idx = ori_edge_idx
        self.node2edge = node2edge
        self.edge2node = edge2node
        self.num_ori_nodes = num_ori_nodes
        self.num_ori_edges = num_ori_edges
        self.num_dup_nodes = num_dup_nodes
        self.num_dup_edges = num_dup_edges
        self.y = y
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_N':
            return self.num_dup_nodes
        if key == 'edge_index_E':
            return self.num_dup_edges
        if key == "node2edge":
            return self.num_ori_edges
        if key == "edge2node":
            return self.num_ori_nodes
        if key == "ori_node_idx":
            return self.num_ori_nodes
        if key == "ori_edge_idx":
            return self.num_ori_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
######## transform pair-wise graph to hypergraph ###########
def to_hyper(graph):
    t0 = time.time()
    num_nodes = graph.num_nodes
    hyper_edge_list = []
    for ind in range(num_nodes):
        hyper_edge, _, _, _ = k_hop_subgraph(ind,
                                             num_hops=1,
                                             num_nodes=num_nodes,
                                             edge_index=graph.edge_index)
        hyper_edge = set(hyper_edge.tolist())
        hyper_edge_list.append(hyper_edge)
    edge_subgraph_list, node_subgraph_list = build_dual_subgraph(num_nodes,
                                                                 hyper_edge_list)
    # edge_subgraph is the batched graph by combining
    # sub-structure/graph in all hyperedges
    edge_subgraph = Batch.from_data_list(edge_subgraph_list)
    # edge_subgraph is the dual batched graph of edge_subgraph
    node_subgraph = Batch.from_data_list(node_subgraph_list)
    
    ori_node_idx = edge_subgraph.ori_node_idx
    edge_index_N = edge_subgraph.edge_index
    node2edge = edge_subgraph.batch
    num_ori_edges = len(edge_subgraph_list)
    num_dup_nodes = edge_subgraph.num_nodes
    
    ori_edge_idx = node_subgraph.ori_edge_idx
    edge_index_E = node_subgraph.edge_index
    edge2node = node_subgraph.batch
    num_ori_nodes = len(node_subgraph_list)
    num_dup_edges = node_subgraph.num_nodes
    
    pairgraph = PairData(graph.x, edge_index_N, node2edge, edge_index_E, edge2node,
                         ori_node_idx, ori_edge_idx, num_ori_nodes, num_ori_edges,
                         num_dup_nodes, num_dup_edges, graph.y)
    return pairgraph

def superpixel_pre_transform(data):
    data.x = torch.cat([data.x, data.pos], dim=1)
    del data.pos
    if hasattr(data, "edge_attr") and data.edge_attr.dim() == 1:
        data.edge_attr = data.edge_attr.unsqueeze(1)
    return data
    
class Simple2HyperDataset(InMemoryDataset):
    tu_dataset = [ 'DD', 'MUTAG', 'PROTEINS', 'PTC_MR', 'ENZYMES']
    gnn_benchmark = ['MNIST', "CIFAR10"]
    
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        root = osp.join(root, name)
        self.name = name
        self.kwargs = kwargs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def num_features(self):
        return self.data.x_N.shape[1]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        sub_root = osp.join(self.root, "pair_wise")
        if self.name in self.tu_dataset:
            pair_wise_graphs = TUDataset(sub_root, name=self.name)
        elif self.name in self.gnn_benchmark:
            pair_wise_graphs = GNNBenchmarkDataset(sub_root, name=self.name, 
                                                   split=self.kwargs["split"],
                                                   pre_transform=superpixel_pre_transform)
        else:
            raise ValueError
        self.pair_wise_graphs = pair_wise_graphs
    
    @property
    def raw_file_names(self):
        return "None"

    def process(self):
        data_list = self.pair_wise_graphs

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            # import multiprocessing as mp
            # with mp.Pool(30) as pool:
            #     data_list = pool.starmap(self.pre_transform, [(data) for data in data_list])
            from tqdm import tqdm
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class MyTUDataset(TUDataset):
    def __init__(self, root: str, name: str, transform = None, pre_transform = None, pre_filter = None, use_node_attr: bool = False, use_edge_attr: bool = False, cleaned: bool = False):
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)
        self.folds = 10
        self.root_idx_dir = osp(root, name)
        self.k_fold()
        
        
    def k_fold(self):
        # adopted from NestedGNN:
        # https://github.com/muhanzhang/NestedGNN/blob/0c386032f86feb22f5c910011e6ff29e22aae043/kernel/train_eval.py
        """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 10 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 10 fold have unique test set.
        """
        if os.path.exists(osp(self.root_idx_dir, "train_index.pkl")):
            print("loading spliting from ", self.root_idx_dir)
            with open(osp(self.root_idx_dir, "train_index.pkl"), "rb") as f:
                self.train_indices = pkl.load(f)
            with open(osp(self.root_idx_dir, "val_index.pkl"), "rb") as f:
                self.val_indices = pkl.load(f)
            with open(osp(self.root_idx_dir, "test_index.pkl"), "rb") as f:
                self.test_indices = pkl.load(f)
        else:
            folds = self.folds
            skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
            test_indices, train_indices = [], []
            for _, idx in skf.split(torch.zeros(len(self)), self.data.y[self.indices()]):
                test_indices.append(torch.from_numpy(idx))

            val_indices = [test_indices[i - 1] for i in range(folds)]

            for i in range(folds):
                train_mask = torch.ones(len(self), dtype=torch.uint8)
                train_mask[test_indices[i]] = 0
                train_mask[val_indices[i]] = 0
                train_indices.append(train_mask.nonzero().view(-1))

            with open(osp(self.root_idx_dir, "train_index.pkl"), "wb") as f:
                pkl.dump(train_indices, f)
            with open(osp(self.root_idx_dir, "val_index.pkl"), "wb") as f:
                pkl.dump(val_indices, f)
            with open(osp(self.root_idx_dir, "test_index.pkl"), "wb") as f:
                pkl.dump(test_indices, f)
                
            self.train_indices = train_indices
            self.val_indices = val_indices
            self.test_indices = test_indices
            
            print("saving spliting to ", self.root_idx_dir)
        
    

def load_dataset(valliaGNN, data_name):
    tu_data_names = ['DD', 'MUTAG', 'PROTEINS', 'PTC_MR', 'ENZYMES']
    superpixel_data_names = ["MNIST", "CIFAR10"]
    if valliaGNN:
        root = "./data/valliaGNN/"
        if data_name in tu_data_names:
            dataset = MyTUDataset(root, name=data_name)
            return [dataset]
            
        elif data_name in superpixel_data_names:
            train_set = GNNBenchmarkDataset(root, name=data_name, split="train", pre_transform=superpixel_pre_transform)
            val_set   = GNNBenchmarkDataset(root, name=data_name, split="val", pre_transform=superpixel_pre_transform)
            test_set  = GNNBenchmarkDataset(root, name=data_name, split="test", pre_transform=superpixel_pre_transform)
            return [train_set, val_set, test_set]
        else:
            raise ValueError
        
    else:
        root = "./data/hyperGNN/"
        if data_name in tu_data_names:
            dataset = Simple2HyperDataset(root, data_name, pre_transform=to_hyper)
            return [dataset]
        elif data_name in superpixel_data_names:
            train_set = Simple2HyperDataset(root, data_name, pre_transform=to_hyper, split="train")
            val_set   = Simple2HyperDataset(root, data_name, pre_transform=to_hyper, split="val")
            test_set  = Simple2HyperDataset(root, data_name, pre_transform=to_hyper, split="test")
            return [train_set, val_set, test_set]
        else:
            raise ValueError