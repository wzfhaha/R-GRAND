import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
from utils_split.make_dataset import get_dataset, get_train_val_test_split
import ipdb
from .data_loader import data_loader

def load_data(prefix='DBLP'):
    dl = data_loader('../HGB_data/'+prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    print("valid idx", val_idx)
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    #ipdb.set_trace()
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl




def load_data_semi(prefix='DBLP', split_seed=0):
    from scripts.data_loader import data_loader
    dl = data_loader('/data/wenzheng/Heter_data_clean/' + prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)

    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[test_idx] = dl.labels_test['data'][test_idx]
    class_num = labels.shape[1]
    random_state = np.random.RandomState(split_seed)
    labels_val_node = np.concatenate((labels[train_idx], labels[test_idx]))

    idx_val_node = np.concatenate((train_idx, test_idx))
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels_val_node, train_size=20* class_num, val_size=30 * class_num)
    idx_train = idx_val_node[idx_train]
    idx_val = idx_val_node[idx_val]
    idx_test = idx_val_node[idx_test]
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = idx_train
    train_val_test_idx['val_idx'] = idx_val
    train_val_test_idx['test_idx'] = idx_test
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    return features, adjM, labels, train_val_test_idx, dl
