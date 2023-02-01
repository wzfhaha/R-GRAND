import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
from utils_split.make_dataset import get_dataset, get_train_val_test_split
import ipdb
from .data_loader import data_loader
import os.path as osp
from itertools import product
from scipy import io as sio
import dgl
import torch
from collections import Counter, defaultdict
import os

def re_idx(mat, shift):
    row, col = mat.nonzero()
    idx_unique = np.unique(col)
    reidx_map = dict(zip(idx_unique, np.arange(idx_unique.shape[0])))
    reidx_col = np.asarray([reidx_map[i] for i in col]) + shift
    return row, reidx_col, len(idx_unique)

def load_acm_raw(data_path):
    from dgl.data.utils import download, get_download_dir, _get_dgl_url
    data_path = osp.join(data_path, 'ACM.mat')
    if not osp.exists(data_path):
        url = 'dataset/ACM.mat'
        #data_path = get_download_dir() + '/ACM.mat'
        download(_get_dgl_url(url), path=data_path)
    nodes = {'total':0, 'count':{}, 'attr':{}, 'shift':{}}
    link_dl = {'total':0, 'count':{}, 'meta':{}, 'data':defaultdict(list)}

    data = sio.loadmat(data_path)
    # ipdb.set_trace()
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that
    p_vs_p = data['PvsP']       # paper-paper?
    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]
    p_vs_p = p_vs_p[p_selected,:][:, p_selected]
    p_num = p_selected.shape[0]
    p_vs_p_row, p_vs_p_col = p_vs_p.nonzero()
    p_vs_a_row, p_vs_a_col, a_num = re_idx(p_vs_a, p_num)
    p_vs_l_row, p_vs_l_col, l_num = re_idx(p_vs_l, a_num + p_num)
    p_vs_t_row, p_vs_t_col, t_num = re_idx(p_vs_t, a_num + p_num + l_num)
    all_num = a_num + p_num + l_num + t_num
    all_row_idx = np.concatenate([p_vs_p_row, p_vs_a_row, p_vs_l_row, p_vs_t_row, p_vs_p_col, p_vs_a_col, p_vs_l_col, p_vs_t_col])
    all_col_idx = np.concatenate([p_vs_p_col, p_vs_a_col, p_vs_l_col, p_vs_t_col, p_vs_p_row, p_vs_a_row, p_vs_l_row, p_vs_t_row])
    adjM = sp.coo_matrix((np.ones_like(all_row_idx), (all_row_idx, all_col_idx)), shape=(all_num, all_num))
    nodes['total'] = all_num
    links = {}
    links[0] = list(zip(p_vs_p_row, p_vs_p_col))
    links[1] = list(zip(p_vs_p_col, p_vs_p_row))
    links[2] = list(zip(p_vs_a_row, p_vs_a_col))
    links[3] = list(zip(p_vs_l_row, p_vs_l_col))
    links[4] = list(zip(p_vs_t_row, p_vs_t_col))
    links[5] = list(zip(p_vs_a_col, p_vs_a_row))
    links[6] = list(zip(p_vs_l_col, p_vs_l_row))
    links[7] = list(zip(p_vs_t_col, p_vs_t_row))
    num_nodes = [p_num, a_num, l_num, t_num]
    #num_nodes.append(p_selected.shape[0])
    link_dl['data'][0] = sp.coo_matrix((np.ones(p_vs_p_row.shape[0]), (p_vs_p_row, p_vs_p_col)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['data'][1] = sp.coo_matrix((np.ones(p_vs_p_row.shape[0]), (p_vs_p_col, p_vs_p_row)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['data'][2] = sp.coo_matrix((np.ones(p_vs_a_row.shape[0]), (p_vs_a_row, p_vs_a_col)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['data'][3] = sp.coo_matrix((np.ones(p_vs_l_row.shape[0]), (p_vs_l_row, p_vs_l_col)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['data'][4] = sp.coo_matrix((np.ones(p_vs_t_row.shape[0]), (p_vs_t_row, p_vs_t_col)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['data'][5] = sp.coo_matrix((np.ones(p_vs_a_row.shape[0]), (p_vs_a_col, p_vs_a_row)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['data'][6] = sp.coo_matrix((np.ones(p_vs_l_row.shape[0]), (p_vs_l_col, p_vs_l_row)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['data'][7] = sp.coo_matrix((np.ones(p_vs_t_row.shape[0]), (p_vs_t_col, p_vs_t_row)), shape=(nodes['total'], nodes['total'])).tocsr()
    link_dl['total'] = p_vs_p_row.shape[0] * 2 + p_vs_a_row.shape[0] * 2 + p_vs_l_row.shape[0] * 2 + p_vs_t_row.shape[0] * 2
    link_dl['meta'][0] = (0, 0)
    link_dl['meta'][1] = (0, 0)
    link_dl['meta'][2] = (0, 1)
    link_dl['meta'][3] = (0, 2)
    link_dl['meta'][4] = (0, 3)
    link_dl['meta'][5] = (1, 0)
    link_dl['meta'][6] = (2, 0)
    link_dl['meta'][7] = (3, 0)    
    link_dl['count'][0] =  p_vs_p_row.shape[0]
    link_dl['count'][1] =  p_vs_p_row.shape[0]
    link_dl['count'][2] =  p_vs_a_row.shape[0]
    link_dl['count'][3] =  p_vs_l_row.shape[0]
    link_dl['count'][4] =  p_vs_t_row.shape[0]
    link_dl['count'][5] =  p_vs_a_row.shape[0]
    link_dl['count'][6] =  p_vs_l_row.shape[0]
    link_dl['count'][7] =  p_vs_t_row.shape[0]

    # num_nodes.append()
    p_vs_t_array = p_vs_t.toarray()
    shift = 0
    for i, n in enumerate(num_nodes):
        nodes['count'][i] = n
        nodes['shift'][i] = shift
        shift += n
    features = [p_vs_t_array[:, p_vs_t_array.sum(0)>0]]
    print("feat", features)
    nodes['attr'][0] = features[0]
    for i, n in enumerate(num_nodes[1:]):
        features.append(sp.eye(n))
        nodes['attr'][i+1] = sp.eye(n)
    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    # labels = torch.LongTensor(labels)
    print(labels)
    class_num = max(label_ids) + 1
    labels = np.eye(class_num)[labels]
    print(f'all node count:{all_num}, paper:{p_num}, author:{a_num}, filed:{l_num}, term:{t_num}')
    return features, labels, num_nodes, links, adjM, class_num, nodes, link_dl#, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask

def load_data(prefix='DBLP', path = '../HGB_data/'):
    dl = data_loader(path + prefix)
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
    labels[test_idx] = dl.labels_test['data'][dl.labels_test['mask']]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl

def load_HNE(data_path):
    nodeid2idx = {}
    nodes = {}
    feats = {}
    print(data_path)
    with open(osp.join(data_path, 'node.dat'), 'r', encoding='utf-8') as f:
        # idx = 0
        for line in f:
            th = line.split('\t')
            if len(th) == 4:
                # Then this line of node has attribute
                node_id, node_name, node_type, node_attr = th
                node_id = int(node_id)
                node_type = int(node_type)
                node_attr = list(map(float, node_attr.split(',')))
                if node_type not in nodes:
                    nodes[node_type] = []
                    feats[node_type] = []
                    # nodeidx2type[node_type] = []
                nodes[node_type].append(node_id)
                #nodeid2idx[node_type] = idx
                feats[node_type].append(node_attr)
                # nodes['total'] += 1
            elif len(th) == 3:
                # Then this line of node doesn't have attribute
                node_id, node_name, node_type = th
                node_id = int(node_id)
                node_type = int(node_type)
                if node_type not in nodes:
                    nodes[node_type] = []
                    # nodeidx2type[node_type] = []
                # nodeidx2type[]
                nodes[node_type].append(node_id)
                #nodes['count'][node_type] += 1
                #nodes['total'] += 1
            else:
                raise Exception("Too few information to parse!")
    idx = 0
    num_nodes = []
    features = []
    nodeidx2type = {}
    for i in range(len(nodes)):
        for n_i in nodes[i]:
            nodeid2idx[n_i] = idx
            nodeidx2type[idx] = i
            idx += 1
        if i in feats:
            features.append(np.asarray(feats[i]))
        else:
            features.append(sp.eye(len(nodes[i])))
        num_nodes.append(len(nodes[i]))
    #links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
    links = {}
    all_row, all_col, all_weight = [], [], []
    link_set = set([])
    with open(osp.join(data_path, 'link.dat'), 'r', encoding='utf-8') as f:
        for line in f:
            th = line.split('\t')
            h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
            if r_id not in links:
                links[r_id] = []
            h_id = nodeid2idx[h_id]
            t_id = nodeid2idx[t_id]

            links[r_id].append((h_id, t_id))
            link_set.add((h_id, t_id))
            link_set.add((t_id, h_id))
            all_row.append(h_id)
            all_col.append(t_id)
            all_weight.append(1)
            # all_weight.append(link_weight)
            # links['count'][r_id] += 1
            # links['total'] += 1

    all_node_num = sum(num_nodes)
    adjM = sp.coo_matrix((all_weight, (all_row, all_col)), shape=(all_node_num, all_node_num)).tocsr()
    
    nc = 0
    label_data = [None for i in range(all_node_num)]
    with open(osp.join(data_path, 'label.dat'), 'r', encoding='utf-8') as f:
        for line in f:
            th = line.split('\t')
            node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
            for label in node_label:
                nc = max(nc, label+1)
            label_data[nodeid2idx[node_id]] = node_label
    class_num = nc
    label = np.zeros((all_node_num, class_num), dtype=int)
    for i,x in enumerate(label_data):
        if x is not None:
            for j in x:
                label[i, j] = 1
    #label  
    links_new = {}
    for i, key in enumerate(links.keys()):
        links_new[i] = links[key]
    print(f'link type num:{len(links_new)}')
    print(num_nodes)
    # ipdb.set_trace()
    return adjM, label, links_new, num_nodes


def load_data_taobao(data_path):
    G = torch.load(osp.join(data_path, 'G.pkl'))
    link_type_name = ['user-purchase-brand', 'brand-purchaseby-user', 'user-fav-brand', 'brand-favby-user', 'user-cart-brand', 'brand-cartby-user', 'item-propagandize-brand', 'brand-propagandizeby-user', 'user-click-item', 'item-clickby-user']
    """
    bug:
    brand-propagandizeby-user 应该为 brand-propagandizeby-item
    node type 0: user
    node type 1: item
    node type 2: brand
    """
    nodes = {'total':0, 'count':{}, 'attr':{}, 'shift':{}}
    link_dl = {'total':0, 'count':{}, 'meta':{}, 'data':defaultdict(list)}
    
    links = {}
    num_nodes = []
    num_nodes.append((G.ndata['type'] == 0).sum())
    num_nodes.append((G.ndata['type'] == 1).sum())
    num_nodes.append((G.ndata['type'] == 2).sum())
    nodes['total'] = sum(num_nodes)
    nodes['count'][0] = num_nodes[0]
    nodes['count'][1] = num_nodes[1]
    nodes['count'][2] = num_nodes[2]
    nodes['shift'][0] = 0
    nodes['shift'][1] = num_nodes[1]
    nodes['shift'][2] = num_nodes[1] + num_nodes[2] 
    print(num_nodes)
    for etype, etype_name in enumerate(link_type_name):
        eids = (G.edata['type'] == etype).nonzero().flatten().numpy()
        left_edges = G.edges()[0][eids].numpy()
        right_edges = G.edges()[1][eids].numpy()
        links[etype] = list(zip(left_edges, right_edges))
        link_dl['data'][etype] = sp.coo_matrix((np.ones(left_edges.shape[0]), (left_edges, right_edges)), shape=(nodes['total'], nodes['total'])).tocsr()
        link_dl['total'] += left_edges.shape[0]
        link_dl['count'][etype] = left_edges.shape[0]
    link_dl['meta'][0] = (0, 2)
    link_dl['meta'][1] = (2, 0)
    link_dl['meta'][2] = (0, 2)
    link_dl['meta'][3] = (2, 0)    
    link_dl['meta'][4] = (0, 2)
    link_dl['meta'][5] = (2, 0)      
    link_dl['meta'][6] = (1, 2)
    link_dl['meta'][7] = (2, 1) 
    link_dl['meta'][8] = (0, 1)
    link_dl['meta'][9] = (1, 0) 
    
    row_tensor, col_tensor = G.adj_sparse('coo')
    row_np = row_tensor.numpy()
    col_np = col_tensor.numpy()
    adjM = sp.csr_matrix((np.ones(row_np.shape[0]), (row_np, col_np)), shape=(sum(num_nodes), sum(num_nodes)))
    label = G.ndata['label'].numpy()
    # ipdb.set_trace()
    return adjM, label, links, num_nodes, nodes, link_dl


def load_data_semi(prefix='DBLP', split_seed=0, path='/data/wenzheng/Heter_data_clean/', train_ratio=0.03, valid_ratio=0.005, train_num=None, valid_num=None):
    nodes = {'total':0, 'count':{}, 'attr':{}, 'shift':{}}
    link_dl = {'total':0, 'count':{}, 'meta':{}, 'data':defaultdict(list)}

    # dl = data_loader(path + prefix)
    if prefix == 'DBLP':
        raw_dir = osp.join(path,prefix)
        node_types = ['author', 'paper', 'term', 'conference']
        features = []
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(osp.join(raw_dir, f'features_{i}.npz'))
            features.append(x.toarray())

        x = np.load(osp.join(raw_dir, 'features_2.npy'))
        features.append(x)
        
        node_type_idx = np.load(osp.join(raw_dir, 'node_types.npy'))
        # node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
        num_nodes = []
        for i, type in enumerate(node_types):
            num_nodes.append(int((node_type_idx == i).sum()))
        features.append(sp.eye(num_nodes[-1]))
        
        # data['conference'].num_nodes = int((node_type_idx == 3).sum())

        labels = np.load(osp.join(raw_dir, 'labels.npy'))
        # data['author'].y = torch.from_numpy(y).to(torch.long)
        adjM = sp.load_npz(osp.join(raw_dir, 'adjM.npz'))
        
        class_num = labels.max() + 1
        labels = np.eye(class_num)[labels]
        s = {}
        s['author'] = (0, num_nodes[0])
        s['paper'] = (num_nodes[0], num_nodes[0] + num_nodes[1])
        s['term'] = (num_nodes[0] + num_nodes[1], num_nodes[0] + num_nodes[1] + num_nodes[2])
        s['conference'] = (num_nodes[0] + num_nodes[1] + num_nodes[2], num_nodes[0] + num_nodes[1] + num_nodes[2] + num_nodes[3])
        links = {}
        nodes['total'] = adjM.shape[0]
        etype = 0
        type_map = dict(zip(node_types, range(len(node_types))))
        for src, dst in product(node_types, node_types):
            A_sub = adjM[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = A_sub.row + s[src][0]
                col = A_sub.col + s[dst][0]
                links[etype] = list(zip(row, col))
                link_dl['total'] += A_sub.nnz
                data = np.ones(A_sub.nnz)
                link_dl['data'][etype] = sp.coo_matrix((data, (row, col)), shape=(nodes['total'], nodes['total'])).tocsr()
                link_dl['count'][etype] = A_sub.nnz
                link_dl['meta'][etype] = (type_map[src], type_map[dst])
                etype += 1
                
        print(link_dl['meta'])
        """
        {0: (0, 1), 1: (1, 0), 2: (1, 2), 3: (1, 3), 4: (2, 1), 5: (3, 1)}
        """
        labeled_idx = np.arange(num_nodes[0])
        labels_selected = labels
        shift = 0
        for i in range(4):
            nodes['count'][i] = num_nodes[i]
            nodes['attr'][i] = features[i]
            nodes['shift'][i] = shift
            shift += num_nodes[i]
        
    elif prefix == 'ACM':
        features, labels, num_nodes, links, adjM, class_num, nodes, link_dl = load_acm_raw(osp.join(path, prefix))
        labeled_idx = np.arange(num_nodes[0])
        labels_selected = labels
        
    elif prefix == 'taobao':
        adjM, labels, links, num_nodes, nodes, link_dl = load_data_taobao(osp.join(path, prefix))
        class_num = labels.shape[1]
        labeled_idx = labels.sum(1).nonzero()[0]
        labels_selected = labels[labeled_idx]
        features = []
        for i, num in enumerate(num_nodes):
            features.append(sp.eye(num))
            nodes['attr'][i] = sp.eye(num)
    
    assert (np.absolute(adjM - adjM.T).sum() == 0)
    random_state = np.random.RandomState(split_seed)
    if train_ratio is not None:
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels_selected, train_size=int(float(num_nodes[0] * train_ratio)), val_size=int(float(num_nodes[0]) * valid_ratio))
    else:
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels_selected, train_size=train_num* class_num, val_size=valid_num* class_num)

    idx_train = labeled_idx[idx_train]
    idx_val = labeled_idx[idx_val]
    idx_test = labeled_idx[idx_test]
    print(f'train num {len(idx_train)}, valid num {len(idx_val)}, test num {len(idx_test)}')
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = idx_train
    train_val_test_idx['val_idx'] = idx_val
    train_val_test_idx['test_idx'] = idx_test
    # ipdb.set_trace()
    if prefix != 'IMDB' and prefix != 'Yelp_sample':
        labels = labels.argmax(axis=1)
    return features, adjM, labels, train_val_test_idx, links, num_nodes, nodes, link_dl




class data_loader:
    def __init__(self, nodes, links):
        #self.path = path
        #self.nodes = self.load_nodes()
        #self.links = self.load_links()
        #self.labels_train = self.load_labels('label.dat')
        #self.labels_test = self.load_labels('label.dat.test')
        self.nodes = nodes
        self.links = links
        print(self.links.keys())
    
    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        new_links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        new_labels_train = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        new_labels_test = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg+cnt))
                
                #cnt_label_train = self.labels_train['count'][nt]
                #new_labels_train['count'][nnt] = cnt_label_train
                #new_labels_train['total'] += cnt_label_train
                #cnt_label_test = self.labels_test['count'][nt]
                #new_labels_test['count'][nnt] = cnt_label_test
                #new_labels_test['total'] += cnt_label_test
                
                new_node_type += 1
                new_node_id += cnt
        """
        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]
        """
        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(map(lambda x:old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(self.links['data'][-x - 1].T)
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data'][-meta[0] - 1].T
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now+[col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0]>=0 else self.links['meta'][-meta[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0]>=0 else self.links['meta'][-meta2[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_evaluate(self, test_idx, label, file_name, mode='bi'):
        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == 'multi':
            multi_label=[]
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j]==1]
                multi_label.append(','.join(label_list))
            label=multi_label
        elif mode=='bi':
            label = np.array(label)
        else:
            return
        with open(file_name, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")
                
    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i]+self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        print("info",info)
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]
    
    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i,j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()
    
    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by 
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift+nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes



"""
def load_data_ratio(prefix='DBLP', split_seed=0, path='/data/wenzheng/Heter_data_clean/', train_num=20, valid_num=30):
    
    # dl = data_loader(path + prefix)
    if prefix == 'DBLP':
        raw_dir = osp.join(path,prefix)
        node_types = ['author', 'paper', 'term', 'conference']
        features = []
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(osp.join(raw_dir, f'features_{i}.npz'))
            features.append(x.toarray())

        x = np.load(osp.join(raw_dir, 'features_2.npy'))
        features.append(x)
        
        node_type_idx = np.load(osp.join(raw_dir, 'node_types.npy'))
        # node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
        num_nodes = []
        for i, type in enumerate(node_types):
            num_nodes.append(int((node_type_idx == i).sum()))
        features.append(sp.eye(num_nodes[-1]))
        
        # data['conference'].num_nodes = int((node_type_idx == 3).sum())

        labels = np.load(osp.join(raw_dir, 'labels.npy'))
        # data['author'].y = torch.from_numpy(y).to(torch.long)
        adjM = sp.load_npz(osp.join(raw_dir, 'adjM.npz'))
        
        class_num = labels.max() + 1
        labels = np.eye(class_num)[labels]
        s = {}
        s['author'] = (0, num_nodes[0])
        s['paper'] = (num_nodes[0], num_nodes[0] + num_nodes[1])
        s['term'] = (num_nodes[0] + num_nodes[1], num_nodes[0] + num_nodes[1] + num_nodes[2])
        s['conference'] = (num_nodes[0] + num_nodes[1] + num_nodes[2], num_nodes[0] + num_nodes[1] + num_nodes[2] + num_nodes[3])
        links = {}
        etype = 0
        for src, dst in product(node_types, node_types):
            A_sub = adjM[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = A_sub.row + s[src][0]
                col = A_sub.col + s[dst][0]
                links[etype] = list(zip(row, col))
                etype += 1
        labeled_idx = np.arange(num_nodes[0])
        labels_selected = labels
    elif prefix == 'ACM':
        features, labels, num_nodes, links, adjM, class_num = load_acm_raw(osp.join(path, prefix))
        labeled_idx = np.arange(num_nodes[0])
        labels_selected = labels
        
    elif prefix == 'PubMed':
        adjM, all_labels, links, num_nodes = load_HNE(osp.join(path, prefix))
        class_num = all_labels.shape[1]
        # ipdb.set_trace()
        
    elif prefix == 'Freebase_sample' or prefix == 'Yelp_sample':
        adjM, labels, links, num_nodes = load_HNE(osp.join(path, prefix))
        labeled_idx = labels.sum(1).nonzero()[0]
        labels_selected = labels[labeled_idx]
        class_num = labels.shape[1]
        features = []
        for num in num_nodes:
            features.append(sp.eye(num))
    
    
    assert (np.absolute(adjM - adjM.T).sum() == 0)
    random_state = np.random.RandomState(split_seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels_selected, train_size=train_num* class_num, val_size=valid_num* class_num)

    idx_train = labeled_idx[idx_train]
    idx_val = labeled_idx[idx_val]
    idx_test = labeled_idx[idx_test]
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = idx_train
    train_val_test_idx['val_idx'] = idx_val
    train_val_test_idx['test_idx'] = idx_test
    if prefix != 'IMDB' and prefix != 'Yelp_sample':
        labels = labels.argmax(axis=1)
    return features, adjM, labels, train_val_test_idx, links, num_nodes

"""