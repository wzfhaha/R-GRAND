import torch as th
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv
import torch.nn.functional as F
import ipdb
from dgl.ops import segment_mm, gather_mm
import math
import dgl.function as fn

class BaseGCN(nn.Module):
    def __init__(self, in_dims, h_dim, out_dim,
                 num_hidden_layers=1, dropout=0,
                 use_cuda=False, layer_norm=True, mcdropedge=0.0):
        super(BaseGCN, self).__init__()
        self.in_dims = in_dims
        self.h_dim = h_dim
        self.out_dim = out_dim
        # print(f'num rels {self.num_rels}')
        # print(f'num bases {self.num_bases}')
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.layer_norm = layer_norm
        self.mcdropedge = mcdropedge
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        self.i2h = self.build_input_layer()
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        if self.layer_norm:
            self.norm_layers = nn.ModuleList()
            for idx in range(self.num_hidden_layers):
                self.norm_layers.append(nn.LayerNorm(self.h_dim))
        # h2o
        self.h2o = self.build_output_layer()
        """
        if h2o is not None:
            self.layers.append(h2o)
        """
    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None
    def dropout_func(self, x, p):
        ps = (torch.ones_like(x) * (1-p)).to(x.device)
        masks = torch.bernoulli(ps).to(x.device)
        masks = masks/ ps
        return x * masks

    def forward(self, g, features_list):
        h = []
        for i2h, feature in zip(self.i2h, features_list):
            h.append(i2h(feature))
        h = th.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, self.dropout, training=self.training)
            edge_weight = None
            if self.mcdropedge > 0.0:
                edge_weight = torch.ones((g.num_edges(), h.shape[1])).to(h.device)
                edge_weight = self.dropout_func(edge_weight, self.mcdropedge)
            h = layer(g, h, edge_weight=edge_weight)
            if self.layer_norm:
                h = self.norm_layers[i](h)
        h = self.h2o(g, h)
        #ipdb.set_trace()
        h = h/(th.max(th.norm(h, dim=1, keepdim=True), th.FloatTensor([1e-12]).to(h.device)))
        return h


class SamplingGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, device,
                 num_hidden_layers=1, use_embed=False, dropout=0, layer_norm=True):
        super(SamplingGCN, self).__init__()
        # self.in_dims = in_dims
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.device = device
        # print(f'num rels {self.num_rels}')
        # print(f'num bases {self.num_bases}')
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        # self.use_cuda = use_cuda
        self.layer_norm = layer_norm
        self.use_embed = use_embed
        # create rgcn layers
        self.build_model()

    def build_model(self):
               
        self.layers = nn.ModuleList()
        # i2h
        self.i2h = self.build_input_layer()
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        if self.layer_norm:
            self.norm_layers = nn.ModuleList()
            for idx in range(self.num_hidden_layers):
                self.norm_layers.append(nn.LayerNorm(self.h_dim).to(self.device))
        # h2o
        self.h2o = self.build_output_layer()
        """
        if h2o is not None:
            self.layers.append(h2o)
        """

    def build_input_layer(self):
        if self.use_embed:
            i2h = nn.Embedding(self.in_dim, self.h_dim)
        else:
            i2h = nn.Linear(self.in_dim, self.h_dim, bias=True).to(self.device)
            #return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims]) 
        return i2h
    
    def build_hidden_layer(self, idx):
        return GraphConv(self.h_dim, self.h_dim, activation=F.relu).to(self.device)

    def build_output_layer(self):
        #return nn.Linear(self.h_dim, self.out_dim)
        return GraphConv(self.h_dim, self.out_dim).to(self.device)

    def forward(self, blocks, feats, device):
        # h = []
        """
        for i2h, feature in zip(self.i2h):
            h.append(i2h(feature))
        """
        #h = th.cat(h, 0)
        #if self.embed_nodes > 0:
        h = self.i2h(feats).to(device)
        # blocks = blocks.to(device)
        #elif self.in_dim > 0:
        #    h = 
        for i, (layer, block) in enumerate(zip(self.layers, blocks[:-1])):
            block = block.to(device)
            h = F.dropout(h, self.dropout, training=self.training)
            h = layer(block, h)
            if self.layer_norm:
                h = self.norm_layers[i](h)
        h = self.h2o(blocks[-1].to(device), h)
        #ipdb.set_trace()
        h = h/(th.max(th.norm(h, dim=1, keepdim=True), th.FloatTensor([1e-12]).to(h.device)))
        torch.cuda.empty_cache()        
        return h
