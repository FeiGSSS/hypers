# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/11/13 09:52:05
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

## import all models
from src.nets.GIN import GINNet
from src.nets.HyperSGNN import SHGNN

def load_model(args):
    model_name = args.model_name
    if not "Hyper" in model_name:
        if model_name == "GIN":
            model = GINNet(in_dim=args.in_dim,
                        hid_dim=args.hid_dim,
                        num_classes=args.num_classes,
                        dropout=args.dropout,
                        num_layers=args.num_layers,
                        eps_train=args.eps_train,
                        readout=args.readout,
                        batch_norm=args.batch_norm,
                        residule=args.residule)
    else:
        if "GIN" in model_name:
            inner_gnn = "GIN"
        else:
            raise NotImplementedError
        model = SHGNN(in_dim=args.in_dim,
                      hid_dim=args.hid_dim,
                      num_classes=args.num_classes,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      inner_num_layers=args.inner_num_layers,
                      hyper_heads=args.hyper_heads,
                      inner_gnn=inner_gnn,
                      hyper_convs=args.hyper_convs,
                      readout=args.readout,
                      inner_readout=args.inner_readout,
                      batch_norm=args.batch_norm)
    return model