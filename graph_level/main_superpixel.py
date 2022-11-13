# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/11/13 08:50:16
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import argparse
import torch
from pytorch_lightning import seed_everything

## import custom module
from src.dataset import load_dataset
from src.model import load_model
from src.train.train_gnnbenchmark import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## dataset
    parser.add_argument("--data_name", type=str, choices=['MNIST', "CIFAR10"])
    ## model setting
    parser.add_argument("--model_name", type=str, default="GIN",
                        choices=["GIN", "HyperSGIN","HyperSGAT", "HyperSGCN"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--residule", type=bool, default=True)
    parser.add_argument("--batch_norm", type=bool, default=True)
    parser.add_argument("--eps_train", type=bool, default=True,
                        help="eps for GIN")
    parser.add_argument("--readout", type=str, default="sum")
    
    
    ## hyperGNN model setting
    parser.add_argument("--inner_num_layers", type=int, default=2,
                        help="The number layers of GNN used for sub-structures")
    parser.add_argument("--hyper_heads", type=int, default=4,
                        help="the number of attention heads used in PMA and inner GAT")
    parser.add_argument("--hyper_convs", type=str, default="both",
                        choices=["gnn", "pma", "both"],
                        help="The hyperGNN variants")
    parser.add_argument("--inner_readout", type=str, default="sum",
                        help="The readout of inner GNN")
    
    ## Training configuration
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1E-3)
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
    parser.add_argument('--lr_schedule_patience', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.,
                        help="weight decay")
    parser.add_argument("--cuda_id", type=int, default=0)
    
    ## Others
    parser.add_argument('--seed', type=int, default=41)
    
    args = parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    
    # device
    device = torch.device("cuda:{}".format(args.cuda_id)) if args.cuda_id >=0 else torch.device("cpu")
    args.device = device
    
    # load data
    args.valliaGNN = False if "Hyper" in args.model_name else True
    dataset = load_dataset(args.valliaGNN, args.data_name)
    args.in_dim = dataset[0].num_features
    args.num_classes = dataset[0].num_classes
    
    # build model
    model = load_model(args)
    
    # train and eval
    run(args, dataset, model)