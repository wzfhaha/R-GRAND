import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import copy
import torch as th

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from utils.data import load_data_semi
import sys

sys.path.append('../../')
from utils.data import data_loader


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,  # Learning rate
    'num_heads': [8],  # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
}

sampling_configure = {
    'batch_size': 20
}


def setup(args):
    args.update(default_configure)
    #set_random_seed(args['seed'])
    args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm(args, feat_type=0):
    features_list, adjM, labels, train_val_test_idx,  links, num_nodes, nodes, link_dl = load_data_semi(args.dataset, args.run, args.path, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    dl = data_loader(nodes, link_dl)
    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ps', 4: 'pt', 5: 'ap', 6: 'sp', 7: 'tp'}
    #link_type_dic = {0: 'pp', 1: '-pp', 2: 'ap', 3: 'sp', 4: 'tp', 5: 'pa', 6: 'pt', 7: 'sp'}

    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # paper feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    # author labels

    #labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    #labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 3

    train_indices = train_val_test_idx['train_idx']
    train_indices = np.sort(train_indices)
    valid_indices = train_val_test_idx['val_idx']
    valid_indices = np.sort(valid_indices)
    test_indices = train_val_test_idx['test_idx']
    test_indices = np.sort(test_indices)
    train_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    train_mask[train_indices] = True
    valid_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    valid_mask[valid_indices] = True
    test_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    test_mask[test_indices] = True
    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_freebase(feat_type=1):
    dl = data_loader('../../data/Freebase')
    link_type_dic = {0: '00', 1: '01', 2: '03', 3: '05', 4: '06',
                     5: '11',
                     6: '20', 7: '21', 8: '22', 9: '23', 10: '25',
                     11: '31', 12: '33', 13: '35',
                     14: '40', 15: '41', 16: '42', 17: '43', 18: '44', 19: '45', 20: '46', 21: '47',
                     22: '51', 23: '55',
                     24: '61', 25: '62', 26: '63', 27: '65', 28: '66', 29: '67',
                     30: '70', 31: '71', 32: '72', 33: '73', 34: '75', 35: '77',
                     36: '-00', 37: '10', 38: '30', 39: '50', 40: '60',
                     41: '-11',
                     42: '02', 43: '12', 44: '-22', 45: '32', 46: '52',
                     47: '13', 48: '-33', 49: '53',
                     50: '04', 51: '14', 52: '24', 53: '34', 54: '-44', 55: '54', 56: '64', 57: '74',
                     58: '15', 59: '-55',
                     60: '16', 61: '26', 62: '36', 63: '56', 64: '-66', 65: '76',
                     66: '07', 67: '17', 68: '27', 69: '37', 70: '57', 71: '-77',
                     }
    book_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
        # reverse
        if link_type_dic[link_type + 36][0] != '-':
            data_dic[(dst_type, link_type_dic[link_type + 36], src_type)] = dl.links['data'][link_type].T.nonzero()
    hg = dgl.heterograph(data_dic)

    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        indices = np.vstack((np.arange(book_num), np.arange(book_num)))
        indices = th.LongTensor(indices)
        values = th.FloatTensor(np.ones(book_num))
        features = th.sparse.FloatTensor(indices, values, th.Size([book_num, book_num]))
    # author labels

    labels = dl.labels_test['data'][:book_num] + dl.labels_train['data'][:book_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 7

    train_valid_mask = dl.labels_train['mask'][:book_num]
    test_mask = dl.labels_test['mask'][:book_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['00', '00'], ['01', '10'], ['05', '52', '20'], ['04', '40'], ['04', '43', '30'], ['06', '61', '10'],
                  ['07', '70'], ]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_dblp(args, feat_type=0):
    prefix = '../../data/DBLP'
    features_list, adjM, labels, train_val_test_idx,  links, num_nodes, nodes, link_dl = load_data_semi(args.dataset, args.run, args.path, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    dl = data_loader(nodes, link_dl)
    #dl = data_loader(prefix)
    link_type_dic = {0: 'ap', 1: 'pa', 2: 'pt', 3: 'pc', 4: 'tp', 5: 'cp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(author_num))

    # author labels

    #labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    #labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 4

    train_indices = train_val_test_idx['train_idx']
    train_indices = np.sort(train_indices)
    valid_indices = train_val_test_idx['val_idx']
    valid_indices = np.sort(valid_indices)
    test_indices = train_val_test_idx['test_idx']
    test_indices = np.sort(test_indices)
    train_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    train_mask[train_indices] = True
    valid_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    valid_mask[valid_indices] = True
    test_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    test_mask[test_indices] = True

    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths



def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def load_taobao(args, feat_type=0):
    prefix = '../../data/taobao'
    features_list, adjM, labels, train_val_test_idx,  links, num_nodes, nodes, link_dl = load_data_semi(args.dataset, args.run, args.path, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    dl = data_loader(nodes, link_dl)
    #dl = data_loader(prefix)
    link_type_dic = {0: 'u-purchase-b', 1: 'b-purchase-u', 2: 'u-fav-b', 3: 'b-fav-u', 4: 'u-cart-b', 5: 'b-cart-u', 6:'i-prop-b', 7:'b-prop-i', 8:'u-click-i', 9:'i-click-u'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = sp_to_spt(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(author_num))

    # author labels

    #labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    #labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels[:dl.nodes['count'][0]])

    num_classes = 6
    print(labels.shape)
    train_indices = train_val_test_idx['train_idx']
    train_indices = np.sort(train_indices)
    valid_indices = train_val_test_idx['val_idx']
    valid_indices = np.sort(valid_indices)
    test_indices = train_val_test_idx['test_idx']
    test_indices = np.sort(test_indices)
    train_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    train_mask[train_indices] = True
    valid_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    valid_mask[valid_indices] = True
    test_mask = np.zeros(dl.nodes['count'][0], dtype=bool)
    test_mask[test_indices] = True

    meta_paths = [['u-purchase-b', 'b-purchase-u']]#, ['u-click-i','i-prop-b', 'b-prop-i', 'i-click-u']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths

def load_imdb(feat_type=0):
    prefix = '../../data/IMDB'
    dl = data_loader(prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(movie_num))

    # author labels

    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = th.FloatTensor(labels)

    num_classes = 5

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, dl


def load_data(args, dataset, feat_type=0):
    load_fun = None
    if dataset == 'ACM':
        load_fun = load_acm
    elif dataset == 'Freebase':
        feat_type = 1
        load_fun = load_freebase
    elif dataset == 'DBLP':
        load_fun = load_dblp
    elif dataset == 'taobao':
        load_fun = load_taobao
    return load_fun(args, feat_type=feat_type)


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc <= self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
