# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2022/09/25 23:43:22
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import numpy as np

import torch
import torch_geometric as tg

from src.load_data import load_citation_dataset, load_LE_dataset, load_yelp_dataset


class MyDataset():
    def __init__(self, path:str, dataset:str, mini_batch:int=10):
        if "cocitation" in path:
            data = load_citation_dataset(path, dataset)
        elif "coauthorship" in path:
            data = load_citation_dataset(path, dataset)
        elif dataset in ["zoo", "ModelNet40", "NTU2012", "20newsW100"]:
            data = load_LE_dataset(path, dataset)
        elif dataset == "yelp":
            data = load_yelp_dataset(path, dataset)
        else:
            raise ValueError
        self.features = data[2]
        self.labels = data[3]
        self.edge_sub_batch, self.node_sub_batch = self.mini_batch(data, mini_batch)
        
        self.num_class = int(torch.max(self.labels) + 1)
        self.split = self.rand_split_labels(self.labels)
    
    def to(self, device):
        self.edge_sub_batch = [x.to(device) for x in self.edge_sub_batch]
        self.node_sub_batch = [x.to(device) for x in self.node_sub_batch]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        return self
    
    def mini_batch(self, data, mini_batch:int):
        edge_sub_list = data[0]
        node_sub_list = data[1]
        num_edges = len(edge_sub_list)
        num_nodes = len(node_sub_list)
        
        assert isinstance(mini_batch, int) and mini_batch >= 1
        
        edge_bs = num_edges // mini_batch
        node_bs = num_nodes // mini_batch
        
        edge_sub_batch = []
        for cnt in range(mini_batch+1):
            start = cnt * edge_bs
            end = (cnt+1) * edge_bs
            edge_sub_batch.append(tg.data.Batch.from_data_list(edge_sub_list[start:end]))
            if end >= num_edges:
                break
        
        node_sub_batch = []
        for cnt in range(mini_batch+1):
            start = cnt * node_bs
            end = (cnt+1) * node_bs
            node_sub_batch.append(tg.data.Batch.from_data_list(node_sub_list[start:end]))
            if end >= num_nodes:
                break
        
        return edge_sub_batch, node_sub_batch
            
        
        
        
    
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
        