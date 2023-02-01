import sys
sys.path.append('../../')
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data_semi
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from model import MAGNN_nc_mb
from utils.tools import evaluate, evaluate_valid

# Params
out_dim = 4
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
etypes_list = [[0, 1], [0, 2, 4, 1], [0, 3, 5, 1]]


def get_adjlist_pkl(dl, meta, type_id=0, return_dic=True, symmetric=False):
    meta010 = dl.get_meta_path(meta).tocoo()
    adjlist00 = [[] for _ in range(dl.nodes['count'][type_id])]
    for i,j,v in zip(meta010.row, meta010.col, meta010.data):
        adjlist00[i-dl.nodes['shift'][type_id]].extend([j-dl.nodes['shift'][type_id]]*int(v))
    adjlist00 = [' '.join(map(str, [i]+sorted(x))) for i,x in enumerate(adjlist00)]
    meta010 = dl.get_full_meta_path(meta, symmetric=symmetric)
    idx00 = {}
    for k in meta010:
        idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, len(meta)+1])
    if not return_dic:
        idx00 = np.concatenate(list(idx00.values()), axis=0)
    return adjlist00, idx00

def load_DBLP_data(args):
    from utils.data import data_loader
    features_list, adjM, labels, train_val_test_idx,  links, num_nodes, nodes, link_dl = load_data_semi(args.dataset, args.run, args.path, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    dl = data_loader(nodes, link_dl)
    adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], symmetric=True)
    print('meta path 1 done')
    adjlist01, idx01 = get_adjlist_pkl(dl, [(0,1), (1,2), (2,1), (1,0)], symmetric=True)
    print('meta path 2 done')
    adjlist02, idx02 = get_adjlist_pkl(dl, [(0,1), (1,3), (3,1), (1,0)], symmetric=True)
    print('meta path 3 done')
    features = []
    for i in range(4):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
        else:
            if type(th) is np.ndarray:
                features.append(th)
            else:
                features.append(th.toarray())
    features_0, features_1, features_2, features_3 = features
    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    for i in range(4):
        type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
    """
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    """
    return [adjlist00, adjlist01, adjlist02], \
           [idx00, idx01, idx02], \
           [features_0, features_1, features_2, features_3],\
           adjM, \
           type_mask,\
           labels,\
           train_val_test_idx,\
            dl


def run_model_DBLP(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix, args):
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_DBLP_data(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
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
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []
    for _ in range(repeat):
        net = MAGNN_nc_mb(3, 6, etypes_list, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)

                t1 = time.time()
                dur1.append(t1 - t0)

                logits, embeddings = net(
                    (train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch])

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 50 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

            # validation
            net.eval()
            val_logp = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                    logits, embeddings = net(
                        (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    logp = F.log_softmax(logits, 1)
                    val_logp.append(logp)
                val_loss = F.nll_loss(torch.cat(val_logp, 0), labels[val_idx])
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
        test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        test_embeddings = []
        test_logits = []
        with torch.no_grad():
            t1 = time.time()
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(adjlists,
                                                                                             edge_metapath_indices_list,
                                                                                             test_idx_batch,
                                                                                             device, neighbor_samples)
                logits, embeddings = net((test_g_list, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
                test_embeddings.append(embeddings)
                test_logits.append(logits)
            test_embeddings = torch.cat(test_embeddings, 0)
            test_logits = torch.cat(test_logits, 0)
            #np.save('out.npy', test_logits.cpu().numpy())
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(4, dtype=np.int32)
            pred = onehot[pred]
            test_y_true = onehot[labels[test_idx].cpu().numpy()]
            test_result = evaluate(pred, test_y_true)
            t2 = time.time()
            print(test_result)
            print(t2 - t1)
            return test_result
            
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others) We need to try this! Or why did we use glove!;' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='DBLP', help='Postfix for the saved model and result. Default is DBLP.')
    ap.add_argument('--run_num', type=int, default=5, help='run num')
    ap.add_argument('--run_num_seed', type=int, default=3, help='run num')
    ap.add_argument('--train_ratio', type=float, default=0.01)
    ap.add_argument('--valid_ratio', type=float, default=0.01)
    ap.add_argument('--path', type=str, default='../../data/')
    ap.add_argument('--dataset', type=str, default='DBLP')
    args = ap.parse_args()
    test_res_list = []
    for i in range(args.run_num):
        for seed in range(args.run_num_seed):
            args.run = i + 1
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            test_result = run_model_DBLP(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                   args.epoch, args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix, args)
            test_res_list.append(test_result)
    print(test_res_list)
    avg_res_micro_f1_test = [res['micro-f1'] for res in test_res_list]
    avg_res_macro_f1_test = [res['macro-f1'] for res in test_res_list]
    print(f'test avg micro f1: {np.mean(avg_res_micro_f1_test)}, {np.std(avg_res_micro_f1_test)}')   
    print(f'test avg macro f1: {np.mean(avg_res_macro_f1_test)}, {np.std(avg_res_macro_f1_test)}')   
    print(args)

    args.run = 1

