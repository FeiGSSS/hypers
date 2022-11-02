# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/11/01 00:18:31
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import torch

from sklearn.model_selection import StratifiedKFold

def k_fold(dataset, folds:int):
    # adopted from Nested GNN
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)),
                            dataset.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices