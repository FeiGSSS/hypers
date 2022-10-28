from copy import deepcopy
import os
import argparse
from tqdm import trange
import numpy as np

from src.dataset import MyDataset, rand_split_labels
from src.model import SHGNN

import torch
from torch.optim import Adam
from torch_geometric.utils import add_self_loops

import nni
from nni.utils import merge_parameter

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    edge_sub_batch = deepcopy(data.edge_sub_batch)
    node_sub_batch = deepcopy(data.node_sub_batch)
    for b in edge_sub_batch:
        b.edge_index = add_self_loops(b.edge_index)[0]
    for b in node_sub_batch:
        b.edge_index = add_self_loops(b.edge_index)[0]
    features = data.features
    out = model(edge_sub_batch,
                node_sub_batch,
                features)

    train_acc = eval_acc(data.labels[split_idx_lst[run]['train']],
                         out[split_idx_lst[run]['train']])
    valid_acc = eval_acc(data.labels[split_idx_lst[run]['valid']],
                         out[split_idx_lst[run]['valid']])
    test_acc = eval_acc(data.labels[split_idx_lst[run]['test']],
                        out[split_idx_lst[run]['test']])
    result = [x if isinstance(x, float) else x.cpu().numpy() for x in [train_acc, valid_acc, test_acc, out]]
    return result


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def train(args, data, device):
    # model
    model = SHGNN(heads=args.heads,
                  pool=args.pool,
                  num_layers=args.num_layers,
                  dim=args.dim,
                  num_features=data.features.size()[1],
                  num_class=data.num_class,
                  dp=args.dp,
                  type_gnn=args.type_gnn,
                  convs=args.convs)
    model = model.to(device)

    # optimizer
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = 0
    best_test = 0
    patience_cnt = 0

    edge_sub_batch = deepcopy(data.edge_sub_batch)
    node_sub_batch = deepcopy(data.node_sub_batch)
    for b in edge_sub_batch:
        b.edge_index = add_self_loops(b.edge_index)[0]
    for b in node_sub_batch:
        b.edge_index = add_self_loops(b.edge_index)[0]
    features = data.features

    process = trange(args.epochs)
    model.reset_parameters()
    for epoch in process:
        model.train()
        opt.zero_grad()
        pred = model(edge_sub_batch,
                     node_sub_batch,
                     features, )
        loss = model.loss_fun(pred[split_idx_lst[run]["train"]],
                              data.labels[split_idx_lst[run]["train"]])
        loss.backward()
        opt.step()

        result = evaluate(model, data)

        if result[1] > best_val:
            patience_cnt = 0
            best_val = result[1]
            best_test = result[2]
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                break

        process.set_description('Runs {:<1}/{} Epoch {:<2}'.format(run + 1, runs, epoch))
        process.set_postfix(valid_acc=result[1] * 100, test_acc=result[2] * 100, best_test_acc=best_test * 100)

    return best_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HGNN with sub-structure awared")
    parser.add_argument("--data_path", type=str, default="./data/raw_data/cocitation/")
    parser.add_argument("--data_name", type=str, default="citeseer")

    # model parameters
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dp", type=float, default=0.5, help="dropout")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)

    parser.add_argument("--type_gnn", type=str, default='GCN', choices=['GAT', 'GIN', 'GCN'])

    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--convs", action="store_true", help="whether use GNN")
    parser.add_argument("--pool", type=str, default="mean", help="pooling methods")

    args = parser.parse_args()
    print(args)

    '''tuner_params = nni.get_next_parameter()
    args = vars(merge_parameter(args, tuner_params))'''

    print("Exps. for ", os.path.join(args.data_path, args.data_name))
    device = torch.device('cuda:{}'.format(args.device)) if args.device >= 0 else torch.device('cpu')

    # load dataset
    dataset = MyDataset(args.data_path, args.data_name)
    dataset = dataset.to(device)

    all_test_score = []
    runs = 10
    split_idx_lst = []
    for run in range(runs):
        split_idx = rand_split_labels(dataset.labels)
        split_idx_lst.append(split_idx)
        
    for run in range(runs):
        '''dataset = MyDataset(args.data_path, args.data_name)
        dataset = dataset.to(device)'''
        best_test = train(args, dataset, device)
        all_test_score.append(best_test)
    print("Results for {} runs:".format(runs))
    print("mean : {:.2f}% std : {:.2f}".format(np.mean(all_test_score) * 100, np.std(all_test_score) * 100))