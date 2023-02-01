import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn.functional as F
import numpy as np
from dgl import DropEdge
from conv import myGATConv, myGATConv_s#, myGATConv_semi, myGATConv_s_attn, myGATConv_s_attn2, myGATConv_noatt
import ipdb
import copy
"""
class myGATDiff(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 feats_type,
                 edge_alpha,
                 edge2type,
                 layer_norm,
                 fix_edge=False,
                 wo_norm=False):
        super(myGATDiff, self).__init__()
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
        self.alpha = alpha
        # hidden layers
        self.layer_norm = layer_norm
        self.gat_layers.append(myGATConv_s(edge_dim, num_etypes,
                num_hidden * heads[0], num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, self.activation, bias=False, alpha=alpha, edge_alpha=edge_alpha, fix_edge=fix_edge, wo_norm=wo_norm))

        for l in range(1, num_layers):
            self.gat_layers.append(myGATConv_s(edge_dim, num_etypes,
                num_hidden  * heads[l], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, self.activation, bias=False, alpha=alpha, edge_alpha=edge_alpha, fix_edge=fix_edge, wo_norm=wo_norm)) 
        if self.layer_norm:
            for l in range(0, num_layers):
                self.norm_layers.append(LayerNorm((heads[0], num_hidden)))
                            
        self.predict_layer_attn = myGATConv_s(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, None, bias=False, alpha=alpha, edge_alpha=edge_alpha, fix_edge=fix_edge, wo_norm=wo_norm)
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_classes = num_classes
        self.edge2type = edge2type


    def forward(self, features_list, e_feat_org):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h0 = h
        g_ = self.g
        e_feat = e_feat_org
        res_attn = None
        res_h = h.view(h.shape[0], -1, self.num_hidden)
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g_, h, e_feat, res_attn, res_h)
            
            h = (1-self.alpha)* h + self.alpha * F.dropout(h0.view(h.shape[0], -1, h.shape[2]), self.feat_drop, training=self.training)
            if self.layer_norm:
                h = self.norm_layers[l](h)
            res_h = h
            h = h.flatten(1)

        logits, _ = self.predict_layer_attn(g_, h, e_feat, res_attn)
        logits = logits.mean(1)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits
"""

class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 simplify,
                 layer_norm, 
                 mcdropedge=0.0):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden * heads[0], bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        if simplify:
            conv = myGATConv_s
        else:
            conv = myGATConv
        self.layer_norm = layer_norm
        self.gat_layers.append(conv(edge_dim, num_etypes,
            num_hidden * heads[0], num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha, mcdropedge=mcdropedge))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(conv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha, mcdropedge=mcdropedge))
        # output projection
        self.gat_layers.append(conv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha, mcdropedge=mcdropedge))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        
        if self.layer_norm:
            self.norm_layers = nn.ModuleList()
            for idx in range(num_layers):
                self.norm_layers.append(nn.LayerNorm(num_hidden, heads[0]))


    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            if self.layer_norm:
                h = self.norm_layers[l](h)
            h = h.flatten(1)
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits
