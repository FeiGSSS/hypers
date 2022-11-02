# -*- encoding: utf-8 -*-
'''
@File    :   main_tu.py
@Time    :   2022/10/31 15:47:40
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import argparse

from src.train import run
from src.dataset import GraphDataset
from src.model import SHGNN

import torch
import numpy as np
import random
        
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Graph Classification for TU dataset")
    # Data configuration
    parser.add_argument("-dn", "--data_name", type=str, default="MUTAG", choices=["MUTAG", "DD"])
    
    # model configurations
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--type_gnn", type=str, default="gat",
                        choices=["gat", "gin", "gcn"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--convs", type=str, default="gnn",
                        choices=["gnn", "pma", "both"])
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1E-3)
    parser.add_argument('--wd', type=float, default=0, help="weight decay")
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    
    # Others
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--folds", type=int, default=10,
                        help="The number of cross validation")
    parser.add_argument("--cuda", type=int, default=1)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # load dataset
    dataset = GraphDataset(data_name = args.data_name)
    
    # define model
    model = SHGNN(num_layers=args.num_layers,
                  num_features=dataset.num_features,
                  dim=args.hid_dim,
                  dp=args.dropout,
                  type_gnn=args.type_gnn,
                  convs=args.convs,
                  heads=args.heads,
                  num_classes=dataset.num_classes)
    
    # train
    run(args, dataset, model)