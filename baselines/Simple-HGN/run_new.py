import sys
sys.path.append('../../')
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data_semi
from utils.tools import evaluate
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT
import dgl
import ipdb
import optuna

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

def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, links, num_nodes = load_data_semi(args.dataset, args.run, args.path, train_num=args.train_num, valid_num=args.valid_num)
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
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
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    num_classes = labels.max() + 1
    labels = torch.LongTensor(labels).to(device)
    print(train_val_test_idx)
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
    #print(dl.nodes) 
    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    for _ in range(args.repeat):
        # num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        net = myGAT(g, args.edge_feats, len(links)+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.relu, args.dropout, args.dropout, args.slope, True, args.alpha, args.simplify)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.simplify, args.num_heads))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()

            logits = net(features_list, e_feat)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end - t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list, e_feat)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}_{}_{}.pt'.format(args.dataset, args.num_layers, args.simplify, args.num_heads)))
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(features_list, e_feat)
            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            pred = onehot[pred]
            test_y_true = onehot[labels[test_idx].cpu().numpy()]
            test_result = evaluate(pred, test_y_true)
            print(test_result)
            val_logits = logits[val_idx]
            val_pred = val_logits.cpu().numpy().argmax(axis=1)            
            val_pred = onehot[val_pred]
            val_y_true = onehot[labels[val_idx].cpu().numpy()]
            valid_result = evaluate(val_pred, val_y_true)
            print('validation:', valid_result)
            return valid_result, test_result
    torch.cuda.empty_cache() 

def opt_objective(trial, args):
    valid_res_list = []
    test_res_list = []
    args.hidden_dim =  trial.suggest_int('hidden_dim', 0, 1000)
    args.num_heads = trial.suggest_int('num_heads', 0, 100)
    args.num_layers = trial.suggest_int('num_layers', 0, 10)
    args.edge_feats = trial.suggest_int('edge_feats', 0, 1000)
    args.dropout = trial.suggest_float('dropout', 0, 1)
    for i in range(args.run_num):
        args.run = i + 1 
        for seed in range(args.run_num_seed):
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            valid_result, test_result = run_model_DBLP(args)
            valid_res_list.append(valid_result)
            test_res_list.append(test_result)
        #test_res_list.append(test_result)
    print(valid_res_list)
    #print(test_res_list)
    avg_res_micro_f1 = [res['micro-f1'] for res in valid_res_list]
    avg_res_macro_f1 = [res['macro-f1'] for res in valid_res_list]
    print(f'valid avg micro f1: {np.mean(avg_res_micro_f1)}, {np.std(avg_res_micro_f1)}')   
    print(f'valid avg macro f1: {np.mean(avg_res_macro_f1)}, {np.std(avg_res_macro_f1)}')   
    #print(f'test avg micro f1: {np.mean(avg_res_micro_f1_test)}, {np.std(avg_res_micro_f1_test)}')   
    #print(f'test avg macro f1: {np.mean(avg_res_macro_f1_test)}, {np.std(avg_res_macro_f1_test)}')   
    return -(np.mean(avg_res_micro_f1) + np.mean(avg_res_macro_f1))


def run_best(args, study):
    args.hidden_dim = study.best_trial.params['hidden_dim']
    args.num_heads = study.best_trial.params['num_heads']
    args.num_layers = study.best_trial.params['num_layers']
    args.edge_feats = study.best_trial.params['edge_feats']
    args.dropout = study.best_trial.params['dropout']
    print(args)
    valid_res_list = []
    test_res_list = []
    for i in range(args.run_num):
        args.run = i + 1 
        for seed in range(args.run_num_seed):
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            valid_result, test_result = run_model_DBLP(args)
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
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--path', type=str, default='../../data/')
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=8)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--run_num', type=int, default=2)
    ap.add_argument('--train_num', type=int, default=5)
    ap.add_argument('--valid_num', type=int, default=30)
    ap.add_argument('--run_num_seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--cuda_device', type=int, default=0)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--search', default=False, action='store_true')
    ap.add_argument('--simplify', default=False, action='store_true')

    args = ap.parse_args()
    
    if args.search:
        search_space = {'hidden_dim': [16,32,64], 'num_heads': [8], 'edge_feats':[16, 32, 64], 'num_layers':[2,3,4],  'dropout':[0.5,0.7,0.3]}
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(lambda trial: opt_objective(trial, args), n_trials=args.trials)
        joblib.dump(study, f'{args.dataset}_{args.hidden_dim}_{args.num_heads}_{args.num_layers}_{args.simplify}_{args.edge_feats}_study.pkl')
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
                valid_result, test_result = run_model_DBLP(args)
                valid_res_list.append(valid_result)
                test_res_list.append(test_result)
        print(valid_res_list)
        avg_res_micro_f1 = [res['micro-f1'] for res in valid_res_list]
        avg_res_macro_f1 = [res['macro-f1'] for res in valid_res_list]
        print(f'avg micro f1: {np.mean(avg_res_micro_f1)}, {np.std(avg_res_micro_f1)}')   
        print(f'avg macro f1: {np.mean(avg_res_macro_f1)}, {np.std(avg_res_macro_f1)}')   
        avg_res_micro_f1_test = [res['micro-f1'] for res in test_res_list]
        avg_res_macro_f1_test = [res['macro-f1'] for res in test_res_list]
        print(f'avg micro f1: {np.mean(avg_res_micro_f1_test)}, {np.std(avg_res_micro_f1_test)}')   
        print(f'avg macro f1: {np.mean(avg_res_macro_f1_test)}, {np.std(avg_res_macro_f1_test)}')   
