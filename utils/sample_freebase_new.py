import os.path as osp
import numpy as np
import scipy.sparse as sp
import ipdb

data_path = '../data/Freebase'
nodeid2idx = {}
nodes = {}
feats = {}
nodes_name_dict = {}
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
                nodes_name_dict[node_type] = []
                feats[node_type] = []
            nodes[node_type].append(node_id)
            nodes_name_dict[node_type].append(node_name)
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
                nodes_name_dict[node_type] = []
            nodes_name_dict[node_type].append(node_name)
            nodes[node_type].append(node_id)
            #nodes['count'][node_type] += 1
            #nodes['total'] += 1
        else:
            raise Exception("Too few information to parse!")
idx = 0
num_nodes = []
features = []
nodeidx2type = {}
nodeidx2name = {}
for i in range(len(nodes)):
    for name, n_i in zip(nodes_name_dict[i],nodes[i]):
        nodeid2idx[n_i] = idx
        nodeidx2type[idx] = i
        nodeidx2name[idx] = name
        idx += 1
    """
    if i in feats:
        features.append(np.asarray(feats[i]))
    else:
        features.append(sp.eye(len(nodes[i])))
    """
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
        #link_set.add((h_id, t_id))
        #link_set.add((t_id, h_id))
        all_row.append(h_id)
        all_col.append(t_id)
        all_weight.append(1)
        # all_weight.append(link_weight)
        # links['count'][r_id] += 1
        # links['total'] += 1
rev_links = {}
for k, link_list in links.items():
    l = []
    for u, v in link_list:
        if nodeidx2type[u] != nodeidx2type[v]:
            l.append((v, u))
            all_row.append(v)
            all_col.append(u)
            all_weight.append(1)
    rev_links[k + len(links)] = l

links = {**links, **rev_links}
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
with open(osp.join(data_path, 'label.dat.test'), 'r', encoding='utf-8') as f:
    # print(osp.join(data_path, 'link.dat.test'))
    for line in f:
        th = line.split('\t')
        # ipdb.set_trace()
        node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
        for label in node_label:
            nc = max(nc, label+1)
        label_data[nodeid2idx[node_id]] = node_label
        #labels['count'][node_type] += 1
        #labels['total'] += 1
class_num = nc
label = np.zeros((all_node_num, class_num), dtype=int)
for i,x in enumerate(label_data):
    if x is not None:
        for j in x:
            label[i, j] = 1
    #label  
# label[:,-1] = 0.0
reserved_class_idx = ((~(label[:,-1] == 1)) & label.sum(1)>0).nonzero()[0]

all_idx = np.arange(sum(num_nodes))
#labeled_nodes = label.sum(1).nonzero()[0]
# valid_idx = np.concatenate([labeled_nodes, np.arange(num_nodes[0], sum(num_nodes))])
#ipdb.set_trace()
"""
valid_idx = np.arange(num_nodes[0], sum(num_nodes))
adjM_ = adjM[valid_idx][:,valid_idx]
# label = label[valid_idx]
sampled_idx = (np.asarray(adjM_.sum(1)).flatten() > 40).nonzero()[0]
sampled_idx = np.concatenate([labeled_nodes, sampled_idx + num_nodes[0]])
adjM_sample = adjM[sampled_idx][:,sampled_idx]
label_sample = label[sampled_idx]
final_idx = (np.asarray(adjM_sample.sum(1)).flatten()>0).nonzero()[0]
adjM_s = adjM_sample[final_idx][:,final_idx]
label_s = label_sample[final_idx]

final_sampled_idx = all_idx[sampled_idx][final_idx]
"""
sampled_idx = (np.asarray(adjM[reserved_class_idx][:,np.arange(num_nodes[0], sum(num_nodes))].sum(0)).flatten() > 1).nonzero()[0]
sampled_idx = np.concatenate([reserved_class_idx, sampled_idx + num_nodes[0]])
adjM_sample = adjM[sampled_idx][:,sampled_idx]
label_sample = label[sampled_idx]
final_idx = (np.asarray(adjM_sample.sum(1)).flatten()>0).nonzero()[0]
adjM_s = adjM_sample[final_idx][:,final_idx]
label_s = label_sample[final_idx]

final_sampled_idx = all_idx[sampled_idx][final_idx]


sampled_map = {}
with open('../data/Freebase_sample_new/node.dat', 'w') as f:
    for new_idx, idx in enumerate(final_sampled_idx):
        sampled_map[idx] = new_idx
        name = nodeidx2name[idx]
        type = nodeidx2type[idx]
        f.write(f'{idx}\t{name}\t{type}\n')

with open('../data/Freebase_sample_new/label.dat', 'w') as f:
    for new_idx, (label, idx) in enumerate(zip(label_s, final_sampled_idx)):
        if np.sum(label) == 0:
            continue
        name = nodeidx2name[idx]
        type = nodeidx2type[idx]
        label_numid = label.argmax()
        f.write(f'{idx}\t{name}\t{type}\t{label_numid}\n')

new_links = {}
for link_type, link_list in links.items():
    new_links[link_type] = []
    for h_id, t_id in link_list:
        link_weight = 1.0
        if h_id in sampled_map and t_id in sampled_map:
            new_links[link_type].append((h_id, t_id))
                #f.write(f'{h_id}\t{t_id}\t{link_type}\t{link_weight}\n')
                
link_num = 0
 
with open('../data/Freebase_sample_new/link.dat', 'w') as f:
    e_t = 0
    for k, v in new_links.items():
        print(k, len(v))
        if len(v) > 5000:
            for h_id, t_id, in v:
                link_weight = 1.0
                f.write(f'{h_id}\t{t_id}\t{e_t}\t{link_weight}\n')
                link_num += 1
            e_t += 1
link_num_2 = sum([len(x) for _,x in new_links.items() if len(x)>5000])
print('link_num', link_num, link_num_2)
# h_id, t_id, r_id, link_weight
#ipdb.set_trace()

#adjM, label, links, num_nodes
