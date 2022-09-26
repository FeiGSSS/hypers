
from collections import defaultdict
import os.path as osp
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from networkx.algorithms import bipartite

import torch
import torch_geometric as tg

def build_dual_subgraph(num_nodes:int, edges_list, add_self_loop:bool=True):
    """build the sequence of subgraphs of nodes and edges

    Args:
        hyperedges (List): _description_
    """
    import time
    t0 = time.time()
    print("building sub graph")
    assert isinstance(num_nodes, int) and num_nodes>0
    if add_self_loop:
        for node in range(num_nodes):
            edges_list.append(set([node]))
    num_edges = len(edges_list)
    
    nodes_name = list(range(num_nodes))
    edges_name = [str(e) for e in range(num_edges)]
    # print("{:.1}s".format(time.time()-t0))
    B = nx.Graph()
    B.add_nodes_from(edges_name, bipartite=0)
    B.add_nodes_from(nodes_name, bipartite=1)
    for eid, edge in enumerate(edges_list):
        for n in edge:
            B.add_edge(str(eid), n)
    # print("{:.1}s".format(time.time()-t0))
            
    
    G_nodes = bipartite.weighted_projected_graph(B, nodes_name)
    # print("{:.1}s".format(time.time()-t0))
    G_edges = bipartite.weighted_projected_graph(B, edges_name)
    # print("{:.1}s".format(time.time()-t0))
    
    edge_subgraph_list = []
    for e_name in tqdm(edges_name):
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
    for n_name in tqdm(nodes_name):
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
    
    return edge_subgraph_list, node_subgraph_list
    
#####################################################################
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
        if labels.min() == 1: labels -= 1

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pkl.load(f)

    hyperedges = list(hypergraph.values())
    edge_sub_list, node_sub_list = build_dual_subgraph(num_nodes,hyperedges)
    
    return edge_sub_list, node_sub_list, features, labels

#####################################################################

def load_LE_dataset(path=None, 
                    dataset="ModelNet40"):
    file_name = f'{dataset}.content'
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)#n*f
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape).T
    
    num_nodes = int(edges[0].max() + 1)
    features = features[:num_nodes]
    labels = labels[:num_nodes]
    if labels.min() == 1: labels -= 1
    
    hyperedges = defaultdict(set)
    for (v,e) in edges.T:
        hyperedges[e].add(v)
    hyperedges = list(hyperedges.values())
    edge_sub_list, node_sub_list = build_dual_subgraph(num_nodes,hyperedges)
    
    return edge_sub_list, node_sub_list, features, labels

###########################################################################


def load_yelp_dataset(path,
                      dataset = 'yelp'):
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features = 1000, stop_words = 'english', strip_accents = 'ascii')
    res_name = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    if labels.min() == 1: labels -= 1

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(osp.join(path, dataset, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes
    
    hyperedges = defaultdict(set)
    for (v,e) in zip(node_list, edge_list):
        hyperedges[e].add(v)
    hyperedges = list(hyperedges.values())
    
    edge_sub_list, node_sub_list = build_dual_subgraph(num_nodes,hyperedges)
    
    return edge_sub_list, node_sub_list, features, labels