import argparse
import random
from tqdm import trange
import numpy as np

from src.load_data import MyDataset
from src.model import SHGNN

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(2022)
random.seed(2022)
np.random.seed(2022)

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.edge_sub_batch,
                data.node_sub_batch,
                data.features)

    train_acc = eval_acc(data.labels[data.split['train']],
                                 out[data.split['train']])
    valid_acc = eval_acc(data.labels[data.split['valid']],
                                 out[data.split['valid']])
    test_acc = eval_acc(data.labels[data.split['test']],
                                out[data.split['test']])
    result = [x if isinstance(x, float) else x.cpu().numpy() for x in [train_acc, valid_acc, test_acc, out]]
    return result

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def train(args, data, device):
    # model
    model = SHGNN(num_layers=args.num_layers,
                  dim=args.dim,
                  num_class=data.num_class,
                  dp=args.dp,
                  convs=args.convs)
    model = model.to(device)
    
    # optimizer
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = ReduceLROnPlateau(opt, 'max', factor=0.8, patience=5)
    
    best_val = 0
    best_test = 0
    patience_cnt = 0
    
    process = trange(args.epochs)
    for epoch in process:
        process.set_description('Runs {:<1}/{} Epoch {:<2}'.format(run+1, runs, epoch))
        model.train()
        opt.zero_grad()
        pred = pred = model(data.edge_sub_batch,
                            data.node_sub_batch,
                            data.features)
        loss = model.loss_fun(pred[data.split["train"]], data.labels[data.split["train"]])
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
        
        # scheduler.step(result[1])
        
        process.set_postfix(valid_acc=result[1], test_acc=result[2], best_test_acc=best_test)
    
    return best_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="HGNN with sub-structure awared")
    parser.add_argument("--data_path", type=str, default="./data/raw_data/cocitation/")
    parser.add_argument("--data_name", type=str, default="cora")
    
    # model parameters
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dp", type=float, default=0.4, help="dropout")
    parser.add_argument("--convs", action="store_true", help="whether use GNN")
    
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-4)
    args = parser.parse_args()
    
    print("Exps. for ", args.data_name)
    device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')
    
    # load dataset
    dataset = MyDataset(args.data_path, args.data_name)
    dataset = dataset.to(device)

    
    all_test_score = []
    runs = 10
    for run in range(runs):
        best_test = train(args, dataset, device)
        all_test_score.append(best_test)
    print("Results for {} runs:".format(runs))
    print("mean : {:.2f}% std : {:.2f}".format(np.mean(all_test_score)*100, np.std(all_test_score)*100))