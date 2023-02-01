import dgl
import torch
import numpy as np
import scipy.sparse as sp
import ipdb

G = torch.load('../../JD_data/G_ori.pkl')
# ipdb.set_trace()
np.random.seed(0)

item_num = G.nodes('item').shape[0]

# selected_item = np.random.choice(item_num, 10000)
purchase_adj_tensor = G.adj(etype='purchase-by').coalesce()
purchase_item, purchase_user = purchase_adj_tensor.indices()
purchase_adj_sp = sp.csr_matrix((np.ones(purchase_item.shape[0]), (purchase_item.numpy(), purchase_user.numpy())), shape=(purchase_adj_tensor.shape[0], purchase_adj_tensor.shape[1]))
purchase_item_degree = np.asarray(purchase_adj_sp.sum(0)).flatten()
sort_idx_item = np.argsort(purchase_item_degree)
selected_item = sort_idx_item[-10000:]
# ipdb.set_trace()

selected_user_purchase = purchase_adj_sp[selected_item].nonzero()[1]
del purchase_adj_tensor

click_adj_tensor = G.adj(etype='click-by').coalesce()
click_item, click_user = click_adj_tensor.indices()
click_adj_sp = sp.csr_matrix((np.ones(click_item.shape[0]), (click_item.numpy(), click_user.numpy())), shape=(click_adj_tensor.shape[0], click_adj_tensor.shape[1]))
selected_user_click = click_adj_sp[selected_item].nonzero()[1]
del click_adj_tensor

# ipdb.set_trace()
selected_user = np.asarray(list(set(list(selected_user_purchase)) | set(list(selected_user_click))))

user_num = G.nodes('user').shape[0]
item_num = G.nodes('item').shape[0]
# num_nodes = [user_num, item_num, brand_num]

user_num_selected = len(selected_user)
item_num_selected = len(selected_item)

node_remap_user = {}
for i, u in enumerate(selected_user):
    node_remap_user[u] = i
node_remap_item = {}
for i, v in enumerate(selected_item):
    node_remap_item[v] = i+ user_num_selected
print('1')

links = {}
# p_adj = G.adj(etype='purchase').to_dense().numpy()
brand_edges = purchase_adj_sp[selected_item].nonzero()
brand_row_ = selected_item[brand_edges[0]]
brand_col_ = brand_edges[1]

brand_row = np.asarray([node_remap_item[x] for x in brand_row_])
brand_col = np.asarray([node_remap_user[y] for y in brand_col_])

links['user-purchase-item'] = list(zip(brand_col, brand_row))
links['item-purchaseby-user'] = list(zip(brand_row, brand_col))

edge_set_u_b = set(links['user-purchase-item'])
edge_set_b_u = set(links['item-purchaseby-user'])
# ipdb.set_trace()

# p_adj = G.adj(etype='fav').to_dense().numpy()
# p_adj = G.adj(etype='click').to_dense().numpy()
brand_edges = click_adj_sp[selected_item].nonzero()
brand_row_ = selected_item[brand_edges[0]]
brand_col_ = brand_edges[1]
# ipdb.set_trace()
brand_row = np.asarray([node_remap_item[x] for x in brand_row_])
brand_col = np.asarray([node_remap_user[y] for y in brand_col_])


links['user-click-item'] = list(zip(brand_col, brand_row))
links['item-clickby-user'] = list(zip(brand_row, brand_col))

print('6')

all_links = []
link_etype = []
etype = 0
link_type_name = []
for k, l in links.items():
    all_links += l
    link_etype.append(np.ones(len(l), dtype=int) * etype)
    etype += 1
    link_type_name.append(k)
print(link_type_name)
node_type = []
node_type.append(np.ones(user_num_selected, dtype=int) * 0)
node_type.append(np.ones(item_num_selected, dtype=int) * 1)
link_etype_tensor = torch.Tensor(np.concatenate(link_etype))
node_ntype_tensor = torch.Tensor(np.concatenate(node_type))
label = G.nodes['user'].data['age'][selected_user]
# ipdb.set_trace()
# item_feat = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1), feat3.unsqueeze(1)], dim=-1)
# print(item_feat)
label_vector = torch.eye(label.max()+1)[label]
label_stat = label_vector.sum(0)
print(label_stat)
# ipdb.set_trace()
label = torch.cat([label_vector, torch.zeros((item_num_selected, label.max()+1), dtype=int)], dim=0)
G_sub = dgl.graph(([x[0] for x in all_links], [x[1] for x in all_links]), num_nodes = user_num_selected + item_num_selected)

G_sub.edata['type'] = link_etype_tensor
G_sub.ndata['type'] = node_ntype_tensor
G_sub.ndata['label'] = label
# G_sub.ndata['feat'] = feat_vector
sub_nodes = (((G_sub.in_degrees() > 0) & (G_sub.ndata['type']==0)) | (G_sub.ndata['type']==1) | (G_sub.ndata['type']==2)).nonzero().flatten()

G_sub_ = G_sub.subgraph(sub_nodes)

final_nodes = ((G_sub_.in_degrees() > 0) & ((G_sub_.ndata['label'].argmax(1)!=4) | (G_sub_.ndata['label'].sum(1)==0) )).nonzero().flatten()

G_final = G_sub_.subgraph(final_nodes)
# G_final = G_final.subgraph((G_final.in_degrees()>0).nonzero().flatten())【
# ipdb.set_trace()
label = G_final.ndata['label']
G_final.ndata['label'] = label[:,:-1]
# ipdb.set_trace()
torch.save(G_final, '../data/jd/G.pkl')
# pkl.dump()
# ipdb.set_trace()