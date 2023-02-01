"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

from numpy.lib.function_base import append
from model import BaseRGCN
import json
from sklearn.metrics import f1_score
from scipy import sparse
from dgl.nn.pytorch import RelGraphConv
from model import RelGraphConv_s
import dgl
import torch.nn.functional as F
import torch
import time
import numpy as np
from os import link
import argparse
import torch.nn as nn
import pynvml
import os
import gc
import psutil
import sys
sys.path.append('../../')
from utils.data import load_data_semi, load_data
from utils.pytorchtools import EarlyStopping
import ipdb
import optuna
pynvml.nvmlInit()
import joblib

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def evaluate(model_pred, labels):

    pred_result = model_pred.argmax(dim=1)
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro


def multi_evaluate(model_pred, labels):
    model_pred = torch.sigmoid(model_pred)
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    labels = labels.cpu()
    pred_result = pred_result.cpu()

    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')

    return micro, macro


class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

    def build_hidden_layer(self, idx):
        if self.simplify:
            return RelGraphConv_s(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.elu, self_loop=self.use_self_loop,
                            dropout=self.dropout, layer_norm=self.layer_norm)
        elif self.original:
            return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=F.elu, 
                            self_loop=self.use_self_loop,
                            dropout=self.dropout, layer_norm=self.layer_norm)
        else:
            return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.elu, self_loop=self.use_self_loop,
                            dropout=self.dropout, layer_norm=self.layer_norm)

    def build_output_layer(self):
        if self.simplify:
            return RelGraphConv_s(self.h_dim, self.out_dim, self.num_rels, "basis",
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop, layer_norm=False)
        elif self.original:
            return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, activation=None, 
                                self_loop=self.use_self_loop, layer_norm=False)
        else:
            return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop, layer_norm=False)
 

def main(args):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if args.gpu >= 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_begin = meminfo.used
    if args.dataset in ['imdb']:
        LOSS = F.binary_cross_entropy_with_logits
    else:
        LOSS = F.cross_entropy
    
    if args.dataset in ['imdb']:
        EVALUATE = multi_evaluate
    else:
        EVALUATE = evaluate
    
    print("begin:", gpu_begin)
    features_list, adjM, labels, train_val_test_idx, links, num_nodes = load_data_semi(args.dataset, args.run, args.path, train_num=args.train_num, valid_num=args.valid_num)#load_data_semi(args.dataset, args.run)
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    feats_type = args.feats_type
    in_dims = []
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    num_classes = labels.max() + 1
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    edge2type = {}
    for k in links:
        for u,v in links[k]:
            edge2type[(u,v)] = k
    for i in range(sum(num_nodes)):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(links)
    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    src_node_list = []
    dst_node_list = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        src_node_list.append(u)
        dst_node_list.append(v)
        e_feat.append(edge2type[(u,v)])
    dst_node_np = np.asarray(dst_node_list) 
    src_node_np = np.asarray(src_node_list)      
    e_feat_np = np.asarray(e_feat)    
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    etype_num = np.max(e_feat_np) + 1
    edge_norm = np.ones(e_feat.shape[0])    
    for e_t in np.arange(etype_num):
        e_idx = (e_feat_np == e_t)
        dst_node_e = dst_node_np[e_idx]
        src_node_e = src_node_np[e_idx]
        _, inverse_idx, count = np.unique(dst_node_e, return_inverse=True, return_counts=True)
        edge_norm[e_idx] = 1./count[inverse_idx]
    edge_norm = torch.FloatTensor(edge_norm).unsqueeze(1).to(device)
    edge_type = e_feat
    num_rels = etype_num
    # num_classes = dl.labels_train['num_classes']
    model = EntityClassify(in_dims,
                           args.hidden_dim,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.num_layers,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop, 
                           simplify = args.simplify,
                           original = args.original,
                           layer_norm = args.layer_norm)

    model.to(device)
    g = g.to('cuda:%d' % args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    # print("start training...")
    forward_time = []
    backward_time = []
    save_dict_micro = {}
    save_dict_macro = {}
    best_result_micro = 0
    best_result_macro = 0
    best_epoch_micro = 0
    best_epoch_macro = 0
    print(edge_type.shape)
    print(edge_type.max)
    model.train()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}_{}_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.simplify, args.n_bases, args.original, args.train_num))
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, features_list, edge_type, edge_norm)
        # print('logits', logits)
        #logits = logits[train_idx]
        loss = LOSS(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        model.eval()
        with torch.no_grad():
            logits = model(g, features_list, edge_type, edge_norm)
            val_loss = LOSS(logits[val_idx], labels[val_idx])
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}_{}_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.simplify, args.n_bases, args.original, args.train_num)))
    model.eval()
    result = [save_dict_micro, save_dict_macro]
    torch.cuda.empty_cache()
    with torch.no_grad():
        #model.load_state_dict(result[i])
        t0 = time.time()
        logits = model.forward(g, features_list, edge_type, edge_norm)
        t1 = time.time()
        print("test time:"+str(t1-t0))
        # logits = logits[target_idx]
        test_loss = LOSS(logits[test_idx], labels[test_idx])
        valid_loss = LOSS(logits[val_idx], labels[val_idx])
        test_micro, test_macro = EVALUATE(
            logits[test_idx], labels[test_idx])
        # ipdb.set_trace()
        valid_micro, valid_macro = EVALUATE(
            logits[val_idx], labels[val_idx])
        print("Test micro: {:.4f} | Test macro: {:.4f} | Test loss: {:.4f}".format(
            test_micro, test_macro, test_loss.item()))
        print("Valid micro: {:.4f} | Valid macro: {:.4f} | Valid loss: {:.4f}".format(
            valid_micro, valid_macro, valid_loss.item()))
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_end = meminfo.used
    print("test end:", gpu_end)
    print("net gpu usage:", (gpu_end-gpu_begin)/1024/1024, 'MB')
    torch.cuda.empty_cache() 
    return {'micro-f1': valid_micro, 'macro-f1': valid_macro}, {'micro-f1': test_micro, 'macro-f1': test_macro}



def opt_objective(trial, args):
    valid_res_list = []
    test_res_list = []
    args.hidden_dim =  trial.suggest_int('hidden_dim', 0, 1000)
    args.num_layers = trial.suggest_int('num_layers', 0, 10)
    args.dropout = trial.suggest_float('dropout', 0, 1)
    for i in range(args.run_num):
        args.run = i + 1 
        for seed in range(args.run_num_seed):
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            valid_result, test_result = main(args)
            valid_res_list.append(valid_result)
            test_res_list.append(test_result)
    print(valid_res_list)
    #print(test_res_list)
    avg_res_micro_f1 = [res['micro-f1'] for res in valid_res_list]
    avg_res_macro_f1 = [res['macro-f1'] for res in valid_res_list]
    print(f'valid avg micro f1: {np.mean(avg_res_micro_f1)}, {np.std(avg_res_micro_f1)}')   
    print(f'valid avg macro f1: {np.mean(avg_res_macro_f1)}, {np.std(avg_res_macro_f1)}')   
    return -(np.mean(avg_res_micro_f1) + np.mean(avg_res_macro_f1))


def run_best(args, study):
    args.hidden_dim = study.best_trial.params['hidden_dim']
    args.num_layers = study.best_trial.params['num_layers']
    args.dropout = study.best_trial.params['dropout']
    valid_res_list = []
    test_res_list = []
    for i in range(args.run_num):
        args.run = i + 1 
        for seed in range(args.run_num_seed):
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            valid_result, test_result = main(args)
            valid_res_list.append(valid_result)
            test_res_list.append(test_result)
    print(valid_res_list)
    print(test_res_list)
    avg_res_micro_f1 = [res['micro-f1'] for res in valid_res_list]
    avg_res_macro_f1 = [res['macro-f1'] for res in valid_res_list]
    avg_res_micro_f1_test = [res['micro-f1'] for res in test_res_list]
    avg_res_macro_f1_test = [res['macro-f1'] for res in test_res_list]
    print(f'best valid avg micro f1: {np.mean(avg_res_micro_f1)}, {np.std(avg_res_micro_f1)}')   
    print(f'best valid avg macro f1: {np.mean(avg_res_macro_f1)}, {np.std(avg_res_macro_f1)}')   
    print(f'best test avg micro f1: {np.mean(avg_res_micro_f1_test)}, {np.std(avg_res_micro_f1_test)}')   
    print(f'best test avg macro f1: {np.mean(avg_res_macro_f1_test)}, {np.std(avg_res_macro_f1_test)}')   
    print(study.best_trial.params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument('--feats-type', type=int, default=3,
                        help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--hidden_dim", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--cuda_device", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument("--n_bases", type=int, default=1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=150,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
                        help="l2 norm coef")
    parser.add_argument("--patience", type=int, default=30,
                        help="patience")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument('--run_num', type=int, default=1, help='run num')
    parser.add_argument('--run_num_seed', type=int, default=1, help='run num')
    parser.add_argument('--train_num', type=int, default=20, help='train num')
    parser.add_argument('--valid_num', type=int, default=30, help='valid num')
    parser.add_argument('--path', type=str, default="/data/wenzheng/Heter_data_clean/", help='run num')
    parser.add_argument('--simplify', default=False, action='store_true', help='simplifying')
    parser.add_argument('--original', default=False, action='store_true', help='original R-GCN')
    parser.add_argument('--layer_norm', default=False, action='store_true', help='run num')
    parser.add_argument('--search', default=False, action='store_true', help='hyperparam search')
    parser.add_argument('--trials', type=int, default=30, help='hyperparam search')
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    args.gpu = args.cuda_device
    if args.search:
        search_space = {'hidden_dim':[32, 64, 128], 'num_layers':[2, 3, 4], 'dropout': [0.3, 0.5, 0.7]}
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(lambda trial: opt_objective(trial, args), n_trials=args.trials)
        joblib.dump(study, f'{args.dataset}_{args.simplify}_{args.n_bases}_study.pkl')
        run_best(args, study)
    else:
        valid_res_list = []
        test_res_list = []
        for i in range(args.run_num):
            args.run = i + 1 
            for seed in range(args.run_num_seed):
                torch.cuda.manual_seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                valid_result, test_result = main(args)
                valid_res_list.append(valid_result)
                test_res_list.append(test_result)
        print(valid_res_list)
        print(test_res_list)
        avg_res_micro_f1 = [res['micro-f1'] for res in valid_res_list]
        avg_res_macro_f1 = [res['macro-f1'] for res in valid_res_list]
        avg_res_micro_f1_test = [res['micro-f1'] for res in test_res_list]
        avg_res_macro_f1_test = [res['macro-f1'] for res in test_res_list]
        print(f'valid avg micro f1: {np.mean(avg_res_micro_f1)}, {np.std(avg_res_micro_f1)}')   
        print(f'valid avg macro f1: {np.mean(avg_res_macro_f1)}, {np.std(avg_res_macro_f1)}')   
        print(f'test avg micro f1: {np.mean(avg_res_micro_f1_test)}, {np.std(avg_res_micro_f1_test)}')   
        print(f'test avg macro f1: {np.mean(avg_res_macro_f1_test)}, {np.std(avg_res_macro_f1_test)}')   
        print(args)
