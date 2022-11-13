# -*- encoding: utf-8 -*-
'''
@File    :   train_gnnbenchmark.py
@Time    :   2022/11/13 12:16:40
@Author  :   Fei Gao
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler

from torch_geometric.loader import DataLoader

from src.train.metrics import accuracy_MNIST_CIFAR as accuracy

def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    for iter, batch_graphs in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_graphs.y
        optimizer.zero_grad()
        batch_scores = model(batch_graphs)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_labels = batch_graphs.y
            
            batch_scores = model.forward(batch_graphs)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc


def run(args, dataset, model):
    t0 = time.time()
    per_epoch_time = []
    
    train_dataset, val_dataset, test_dataset = dataset
    device = args.device
    model = model.to(device)
    
    follow_batch = "x" if args.valliaGNN else "x_N"
    
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=2, follow_batch=follow_batch, shuffle=True)
    val_loader   = DataLoader(val_dataset,   args.batch_size, num_workers=2, follow_batch=follow_batch, shuffle=False)
    test_loader  = DataLoader(test_dataset,  args.batch_size, num_workers=2, follow_batch=follow_batch, shuffle=False)
        
    # optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                               factor=args.lr_reduce_factor,
                                               patience=args.lr_schedule_patience,
                                               verbose=True)
    
    epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs = [], [], [], []
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(args.epochs)) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader)

                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader)
                _, epoch_test_acc = evaluate_network(model, device, test_loader)                
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)    

                per_epoch_time.append(time.time()-start)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < args.min_lr:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_acc = evaluate_network(model, device, test_loader)
    _, train_acc = evaluate_network(model, device, train_loader)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
