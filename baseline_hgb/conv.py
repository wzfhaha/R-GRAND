"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import GraphConv
import numpy as np
# pylint: enable=W0235


class SRGC(nn.Module):
    def __init__(self,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 pre_alpha = 0.2,
                 fix_edge=False,
                 wo_norm=False,
                 no_drop=False,
                 aug='mcdropedge'):
        super(SRGC, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fix_edge = fix_edge
        if not self.fix_edge:
            self.edge_emb = nn.Embedding(num_etypes, num_heads)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.aug = aug
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.wo_norm = wo_norm
        self.no_drop = no_drop
        self.pre_alpha = pre_alpha


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if not self.fix_edge:
           self.edge_emb.weight.data.normal_()
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, res_feat=None, drop_blocks=4):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            #graph.srcdata.update({'ft': feat_src})
            #graph.dstdata.update({'ft_dst': feat_dst})
            #if not self.no_drop:
                # print(feat_src.shape, self._num_heads, drop_blocks)
            #    feat_src = feat_src.view(-1, self._num_heads, drop_blocks, self._out_feats//drop_blocks)
            if res_feat != None:
                feat_src = (1 - self.pre_alpha) * feat_src + self.pre_alpha + res_feat
            if self.aug == 'dropout':
                feat_src = self.attn_drop(feat_src)
            elif self.aug == 'dropnode':
                masks = th.ones(feat_src.shape[0]).unsqueeze(-1).unsqueeze(-1).to(feat_src.device)
                masks = self.attn_drop(masks)
                feat_src = masks * feat_src
            elif self.aug == 'mcdropedge':
                feat_src = feat_src.view(-1, self._num_heads, drop_blocks, self._out_feats//drop_blocks)
            graph.srcdata.update({'ft':  feat_src})
            if self.fix_edge:
                e_feat = th.ones((e_feat.shape[0], self._num_heads)).to(feat_src.device)
            else:
                e_feat = self.edge_emb(e_feat)
            
            ee = e_feat.view(e_feat.shape[0], self._num_heads, -1)
            graph.edata.update({'ee': ee})
            e = graph.edata.pop('ee')
            attn = edge_softmax(graph, e)
            if self.aug == 'dropedge':
                masks = th.ones(attn.shape[0]).unsqueeze(-1).unsqueeze(-1).to(attn.device)
                masks = self.attn_drop(masks)
                attn = attn * masks
            elif self.aug == 'mcdropedge':
                attn_expand = attn.repeat(1,1,drop_blocks).unsqueeze(-1)
                attn = self.attn_drop(attn_expand)
            graph.edata['a'] = attn
            # message passing
            # ipdb.set_trace()
            #rst = graph.dstdata['ft']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            if self.wo_norm:
                rst = graph.dstdata['ft']
            else:
                rst = graph.dstdata['ft']#/(attn_sum + 1e-12)
            # bias
            rst = rst.view(-1, self._num_heads, self._out_feats)
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, attn#.detach()

"""
class Agg(nn.Module):
    def __init__(self,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 pre_alpha = 0.2,
                 fix_edge=False,
                 wo_norm=False,
                 no_drop=False,
                 aug='mcdropedge'):
        super(Agg, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fix_edge = fix_edge
        if not self.fix_edge:
            self.edge_emb = nn.Embedding(num_etypes, num_heads)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.aug = aug
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.wo_norm = wo_norm
        self.no_drop = no_drop
        self.pre_alpha = pre_alpha


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if not self.fix_edge:
           self.edge_emb.weight.data.normal_()
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, res_feat=None, drop_blocks=4):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            if res_feat != None:
                feat_src = (1 - self.pre_alpha) * feat_src + self.pre_alpha + res_feat
            graph.srcdata.update({'ft':  feat_src})
            if self.fix_edge:
                e_feat = th.ones((e_feat.shape[0], self._num_heads)).to(feat_src.device)
            else:
                e_feat = self.edge_emb(e_feat)
            
            ee = e_feat.view(e_feat.shape[0], self._num_heads, -1)
            graph.edata.update({'ee': ee})
            e = graph.edata.pop('ee')
            attn = edge_softmax(graph, e)
            graph.edata['a'] = attn
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            # bias
            rst = rst.view(-1, self._num_heads, self._out_feats)
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, attn#.detach()



class SRGC(nn.Module):
    def __init__(self,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 pre_alpha = 0.2,
                 fix_edge=False,
                 wo_norm=False,
                 drop_connect=False,
                 no_drop=False):
        super(SRGC, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fix_edge = fix_edge
        if not self.fix_edge:
            self.edge_emb = nn.Embedding(num_etypes, num_heads)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.node_emb = nn.Embedding(num_nodes, num_heads)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.wo_norm = wo_norm
        self.no_drop = no_drop
        self.drop_connect = drop_connect
        self.pre_alpha = pre_alpha
        

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if not self.fix_edge:
            self.edge_emb.weight.data.normal_()
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, res_feat=None, drop_blocks=4):
        #print(self.edge_emb.weight)
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            if res_feat != None:
                feat_src = (1 - self.pre_alpha) * feat_src + self.pre_alpha + res_feat
            #graph.srcdata.update({'ft': feat_src})
            #graph.dstdata.update({'ft_dst': feat_dst})
            if not self.no_drop:
                # print(feat_src.shape, self._num_heads, drop_blocks)
                feat_src = feat_src.view(-1, self._num_heads, drop_blocks, self._out_feats//drop_blocks)
            graph.srcdata.update({'ft': feat_src })
            if self.fix_edge:
                e_feat = th.ones((e_feat.shape[0], self._num_heads)).to(feat_src.device)
            else:
                e_feat = self.edge_emb(e_feat)
                # print("edge weight", self.edge_emb.weight)
            # left_heads = self.node_emb(left_nodes).view(e_feat.shape[0], self._num_heads,-1)
            # right_heads = self.node_emb(right_nodes).view(e_feat.shape[0], self._num_heads,-1)
            ee = e_feat.view(e_feat.shape[0], self._num_heads, -1)
            graph.edata.update({'ee': ee})
            e = graph.edata.pop('ee')
            attn = edge_softmax(graph, e)
            if self.drop_connect:
                attn_expand = attn.repeat(1,1,self._out_feats)
            else:
                attn_expand = attn            
            if self.no_drop:
                graph.edata['a'] = attn
            else:
                attn_expand = attn.repeat(1,1,drop_blocks).unsqueeze(-1)
                graph.edata['a'] = self.attn_drop(attn_expand)
            # message passing
            # ipdb.set_trace()
            #rst = graph.dstdata['ft']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            if self.wo_norm:
                rst = graph.dstdata['ft']
            else:
                rst = graph.dstdata['ft']#/(attn_sum + 1e-12)
            # bias
            rst = rst.view(-1, self._num_heads, self._out_feats)
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, attn#.detach()
"""