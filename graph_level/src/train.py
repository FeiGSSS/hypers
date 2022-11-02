# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/11/02 08:53:03
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import time
from tqdm import tqdm
from collections import namedtuple

from src.dataset import collate
from src.utils import k_fold
from src.model import SHGNN

import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import DataLoader
from torch.optim import Adam

def data2device(data, device):
    out = namedtuple("out", ["edge_graph", "node_graph", "x", "y","node_to_graph"])
    return out(*[x.to(device) for x in data])

def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data2device(data, device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * len(data.y)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data2device(data, device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device):
    model.eval()
    loss = 0
    for data in loader:
        data = data2device(data, device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def run(args, dataset):
    
    # device
    device = torch.device("cuda:{}".format(args.cuda)) if args.cuda>=0 else torch.device("cpu")
    
    final_train_losses, val_losses, accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds))):
        # define datasets
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_loader = DataLoader(train_dataset, args.batch_size, collate_fn=collate, num_workers=2, shuffle=True)
        val_loader   = DataLoader(val_dataset,   args.batch_size, collate_fn=collate, num_workers=2, shuffle=False)
        test_loader  = DataLoader(test_dataset,  args.batch_size, collate_fn=collate, num_workers=2, shuffle=False)
        
        # define model
        model = SHGNN(num_layers=args.num_layers,
                    num_features=dataset.num_features,
                    inner_num_layers=args.inner_num_layers,
                    dim=args.hid_dim,
                    dp=args.dropout,
                    type_gnn=args.type_gnn,
                    convs=args.convs,
                    heads=args.heads,
                    num_classes=dataset.num_classes)
        model = model.to(device)
        
        # optimizer
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        # train epoches
        t_start = time.perf_counter()
        pbar = tqdm(range(1, args.epochs + 1), ncols=70)
        cur_val_losses = []
        cur_accs = []
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device)
            cur_val_losses.append(eval_loss(model, val_loader, device))
            cur_accs.append(eval_acc(model, test_loader, device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': cur_val_losses[-1],
                'test_acc': cur_accs[-1],
            }
            log = 'Fold: %d, train_loss: %0.4f, val_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["val_loss"], eval_info["test_acc"]
            )
            pbar.set_description(log)

            if epoch % args.lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr_decay_factor * param_group['lr']
        val_losses += cur_val_losses
        accs += cur_accs

        loss, argmin = tensor(cur_val_losses).min(dim=0)
        acc = cur_accs[argmin.item()]
        final_train_losses.append(eval_info["train_loss"])
        log = 'Fold: %d, final train_loss: %0.4f, best val_loss: %0.4f, test_acc: %0.4f' % (
            fold, eval_info["train_loss"], loss, acc
        )
        print(log)

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
    
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(args.folds, args.epochs), acc.view(args.folds, args.epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(args.folds, dtype=torch.long), argmin]

    log = 'Val Loss: {:.4f}, Test Accuracy: {:.1f}% ± {:.1f}, Duration: {:.3f}s'.format(
        loss.mean().item(),
        acc.mean().item()*100,
        acc.std().item()*100,
        duration.mean().item()
    ) #+ ', Avg Train Loss: {:.4f}'.format(average_train_loss)
    print(log)

    return loss.mean().item(), acc.mean().item(), acc.std().item()