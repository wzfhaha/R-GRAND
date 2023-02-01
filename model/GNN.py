import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn.functional as F
from torch.nn import LayerNorm
import numpy as np
from dgl import DropEdge
from conv import SRGC
import copy

class RGRAND(nn.Module):
    def __init__(self,
                 g,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 mcdropedge,
                 negative_slope,
                 feats_type,
                 pre_alpha,
                 edge2type,
                 layer_norm,
                 num_nodes,
                 fix_edge=False,
                 wo_norm=False,
                 drop_blocks=4):
        super(RGRAND, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        self.num_hidden = num_hidden
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden * heads[0], bias=True) for in_dim in in_dims])
        gain = nn.init.calculate_gain('relu')        
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=gain)
        # input projection (no residual)
        self.num_nodes = num_nodes
        # hidden layers
        self.layer_norm = layer_norm
        self.gat_layers.append(SRGC(num_etypes,
                num_hidden * heads[0], num_hidden, heads[0],
                feat_drop, mcdropedge, negative_slope, self.activation, bias=False, pre_alpha=pre_alpha, fix_edge=fix_edge, wo_norm=wo_norm))

        for l in range(1, num_layers):
            self.gat_layers.append(SRGC(num_etypes,
                num_hidden  * heads[l], num_hidden, heads[l],
                feat_drop, mcdropedge, negative_slope, self.activation, bias=False, pre_alpha=pre_alpha, fix_edge=fix_edge, wo_norm=wo_norm)) 
        if self.layer_norm:
            for l in range(0, num_layers):
                self.norm_layers.append(LayerNorm((heads[0], num_hidden)))
                            
        self.predict_layer_attn = SRGC(num_etypes,
            num_hidden * heads[-2], num_classes, heads[-2],
            feat_drop, mcdropedge, negative_slope, None, bias=False, pre_alpha=pre_alpha, fix_edge=fix_edge, wo_norm=wo_norm)
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.feat_drop = feat_drop
        self.num_classes = num_classes
        self.edge2type = edge2type
        self.drop_blocks = drop_blocks


    def forward(self, features_list, e_feat_org):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h0 = h
        print(h.shape)
        g_ = self.g
        e_feat = e_feat_org
        res_attn = None
        res_h = h.view(h.shape[0], -1, self.num_hidden)
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g_, h, e_feat, res_attn, res_h,  drop_blocks=self.drop_blocks)
            if self.layer_norm:
                h = self.norm_layers[l](h)
            res_h = h
            h = h.flatten(1)

        logits, _ = self.predict_layer_attn(g_, h, e_feat, res_attn, drop_blocks=self.num_classes)
        logits = logits.mean(1)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits


class GRAND(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 layer_norm,
                 feat_drop,
                 num_mlp):
        super(GRAND, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        self.num_hidden = num_hidden
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        gain = nn.init.calculate_gain('relu')        
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=gain)
        # input projection (no residual)
        # hidden layers
        self.agg = GraphConv(num_hidden, num_hidden, weight=False, bias=False)
        self.predict_layer = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for i in range(num_mlp - 1)])
        self.predict_layer.append(nn.Linear(num_hidden, num_classes))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.feat_drop = feat_drop
        self.num_classes = num_classes
        self.layer_norm = layer_norm
        self.layer_norm_list = nn.ModuleList([])
        if self.layer_norm:
            for i in range(num_mlp):
                self.layer_norm_list.append(LayerNorm(num_hidden))
        self.dropnode = nn.Dropout(feat_drop)
        self.num_mlp = num_mlp
        for layer in self.predict_layer:
            nn.init.xavier_normal_(layer.weight, gain=gain)
        print("num mlp", num_mlp)

    
    def forward(self, features_list, e_feat_org):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        masks = torch.ones(h.shape[0]).unsqueeze(-1).to(h.device)
        masks = self.dropnode(masks)
        h = masks * h
        h_ = h
        print(h.shape)
        g_ = self.g
        for l in range(self.num_layers):
            h = self.agg(g_, h)
            h_ += h
        h_ = h_ / (self.num_layers + 1.)
        if self.layer_norm:
            h_ = self.layer_norm_list[0](h_)
        for i in range(self.num_mlp-1):
            h_ = self.predict_layer[i](h_)
            h_ = F.elu(h_)
            if self.layer_norm:
                h_ = self.layer_norm_list[i+1](h_)
        logits = self.predict_layer[-1](h_)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits


