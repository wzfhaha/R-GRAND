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
import numpy as np
# pylint: enable=W0235

class myGATConv_edge_only_diff(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 order=5):
        super(myGATConv_edge_only_diff, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, 1)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads * out_feats)))
        self.alpha = alpha
        self.order = order
 
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        #nn.init.xavier_normal_(self.attn_l, gain=gain)
        #nn.init.xavier_normal_(self.attn_r, gain=gain)
        #nn.init.xavier_normal_(self.attn_e, gain=gain)
        #nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
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
                feat_src = h_src.view(-1, self._num_heads * self._out_feats)
                #feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = h_src.view(-1, self._num_heads * self._out_feats)
                #feat_src = feat_dst = self.fc(h_src).view(
                #    -1, self._num_heads, self._out_feats)
                #if graph.is_block:
                #    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            graph.edata['a'] = edge_softmax(graph, e_feat)

            y = self.alpha * feat_src
            x = self.alpha * feat_src
            attn_sum = self.alpha * th.ones((feat_src.shape[0], self._num_heads * self._out_feats)).to(feat_src.device)
            attn_sum_0 = attn_sum
            for i in range(self.order):
                graph.srcdata.update({'ft': x})
                graph.srcdata.update({'attn_sum': attn_sum})
                if i == self.order - 1:
                    graph.edata['a'] = self.attn_drop(edge_softmax(graph, e_feat))
                else:
                    graph.edata['a'] = edge_softmax(graph, e_feat)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
                graph.update_all(fn.u_mul_e('attn_sum', 'a', 'm2'),
                             fn.sum('m2', 'attn_sum'))
                attn_sum = (1 - self.alpha) * graph.dstdata['attn_sum'] + self.alpha * attn_sum_0
                x = (1 - self.alpha) * graph.dstdata['ft'] + self.alpha * feat_src
                y = x
            
            rst = y/attn_sum #graph.dstdata['ft']
            # residual
            
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # bias
            return rst, graph.edata.pop('a').detach()




class myGATConv_edge_multi(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv_edge_only, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        # self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        nn.init.xavier_normal_(self.fc_e.weight, gain=1.414)
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        #nn.init.xavier_normal_(self.attn_l, gain=gain)
        #nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, detach_edge=False):
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
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src})
            print(ee.shape)
            e = self.leaky_relu(ee)
            if detach_edge:
                e = e.detach()
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            attn_sum = th.ones((feat_src.shape[0], self._num_heads, 1)).to(feat_src.device)
            graph.srcdata.update({'attn_sum': attn_sum})
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            graph.update_all(fn.u_mul_e('attn_sum', 'a', 'm2'),
                         fn.sum('m2', 'attn_sum'))
            attn_sum = graph.dstdata['attn_sum']

            rst = graph.dstdata['ft']/(attn_sum + 1e-12)
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()



class myGATConv_edge_only(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv_edge_only, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        # self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        nn.init.xavier_normal_(self.fc_e.weight, gain=1.414)
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        #nn.init.xavier_normal_(self.attn_l, gain=gain)
        #nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, detach_edge=False):
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
            #e_feat = e_feat.view(-1, self._num_heads, self._edge_feats)
            #e_feat = F.relu(self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats))
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            #el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            #er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src})
            #graph.dstdata.update({'er': er})
            #graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            print(ee.shape)
            e = self.leaky_relu(ee)
            if detach_edge:
                e = e.detach()
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            #if res_attn is not None:
            #    graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            attn_sum = th.ones((feat_src.shape[0], self._num_heads, 1)).to(feat_src.device)
            graph.srcdata.update({'attn_sum': attn_sum})
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            graph.update_all(fn.u_mul_e('attn_sum', 'a', 'm2'),
                         fn.sum('m2', 'attn_sum'))
            attn_sum = graph.dstdata['attn_sum']

            rst = graph.dstdata['ft']/(attn_sum + 1e-12)
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()



class myGATConv_rel(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv_rel, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        nn.init.xavier_normal_(self.fc_e.weight, gain=1.414)
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.edge_emb = nn.Embedding(num_etypes, out_feats * num_heads)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, detach_edge=False):
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
                feat_src = self.fc_src(h_src)#.view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst)#.view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src)#.view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            #ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src})
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e_feat))
            #if res_attn is not None:
            #    graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))

            rst = graph.dstdata['ft']#/(th.max(th.norm(graph.dstdata['ft'], dim=1, keepdim=True), 1e-12))
            print(rst.shape)
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()


class myGATConv_s(nn.Module):
    def __init__(self,
                 edge_feats,
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
                 alpha=0.,
                 edge_alpha=0.05):
        super(myGATConv_s, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
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
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha
        self.beta = edge_alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, res_feat=None):
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
            if res_feat != None:
                feat_src = (1 - self.beta) * feat_src + self.beta + res_feat

            graph.srcdata.update({'ft': feat_src })
            e_feat = self.edge_emb(e_feat)
            
            ee = e_feat.view(e_feat.shape[0], self._num_heads, -1)
            graph.edata.update({'ee': ee})
            e = graph.edata.pop('ee')
            if res_attn != None:
                attn = (1 - self.beta) * edge_softmax(graph, e) + self.beta * res_attn
            else:
                attn = edge_softmax(graph, e)
            graph.edata['a'] = self.attn_drop(attn)
            # message passing
            attn_sum = th.ones((feat_src.shape[0], self._num_heads, 1)).to(feat_src.device)
            graph.srcdata.update({'attn_sum': attn_sum})
            graph.update_all(fn.u_mul_e('attn_sum', 'a', 'm2'),
                         fn.sum('m2', 'attn_sum'))
            attn_sum = graph.dstdata['attn_sum']

            #rst = graph.dstdata['ft']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']/(attn_sum + 1e-12)
            # bias
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, attn.detach()




class myGATConv_semi(nn.Module):
    def __init__(self,
                 edge_feats,
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
                 alpha=0.):
        super(myGATConv_semi, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        # self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_e = nn.Parameter(th.FloatTensor(size=(edge_feats, num_heads)))
        self.fc_e = nn.Linear(edge_feats, num_heads)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads * out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        #nn.init.xavier_normal_(self.attn_l, gain=gain)
        #nn.init.xavier_normal_(self.attn_r, gain=gain)
        #nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat):
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
            """
            if attn_e == None:
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1) #/ (th.norm(e_feat, dim=-1, keepdim=True) * th.norm(attn_e, dim=-1, keepdim=True) + 1e-12)
            else:
                ee = (e_feat * attn_e).sum(dim=-1).unsqueeze(-1) #/ (th.norm(e_feat, dim=-1, keepdim=True) * th.norm(attn_e, dim=-1, keepdim=True) + 1e-12)
            if attn_l == None:
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_src, dim=-1, keepdim=True) * th.norm(self.attn_l, dim=-1, keepdim=True) + 1e-12) #/ np.sqrt(self._out_feats)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_dst, dim=-1, keepdim=True) * th.norm(self.attn_r, dim=-1, keepdim=True) + 1e-12)#/ np.sqrt(self._out_feats)
            else:
                el = (feat_src * attn_l).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_src, dim=-1, keepdim=True) * th.norm(attn_l, dim=-1, keepdim=True) + 1e-12) #/ np.sqrt(self._out_feats)
                er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_dst, dim=-1, keepdim=True) * th.norm(attn_r, dim=-1, keepdim=True) + 1e-12)#/ np.sqrt(self._out_feats)


            graph.edata.update({'ee': ee})
            """
            graph.srcdata.update({'ft': feat_src})
            graph.dstdata.update({'ft_dst': feat_dst})
            #e_feat = self.edge_emb(e_feat)
            
            ee = self.fc_e(e_feat).view(e_feat.shape[0], self._num_heads, -1)
            graph.edata.update({'ee': ee})
            #e = self.leaky_relu((graph.edata.pop('e')+graph.edata.pop('ee')))
            # e = th.tanh((graph.edata.pop('e')+graph.edata.pop('ee')))
            e = graph.edata.pop('ee')
            
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            attn_sum = th.ones((feat_src.shape[0], self._num_heads, 1)).to(feat_src.device)
            graph.srcdata.update({'attn_sum': attn_sum})
            graph.update_all(fn.u_mul_e('attn_sum', 'a', 'm2'),
                         fn.sum('m2', 'attn_sum'))
            attn_sum = graph.dstdata['attn_sum']

            #rst = graph.dstdata['ft']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']/(attn_sum + 1e-12)
            # bias
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()





class myGATConv_s_attn2(nn.Module):
    def __init__(self,
                 edge_feats,
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
                 alpha=0.):
        super(myGATConv_s_attn2, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, num_heads * edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        # self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_e = nn.Parameter(th.FloatTensor(size=(edge_feats, num_heads)))
        self.fc_e = nn.Linear(edge_feats, num_heads * out_feats)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads * out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_l, gain=gain)
        # nn.init.xavier_normal_(self.attn_r, gain=gain)
        #nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, h0, attn_l, attn_r, attn_e, res_attn=None, detach_edge=False):
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
            """
            if attn_e == None:
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1) #/ (th.norm(e_feat, dim=-1, keepdim=True) * th.norm(attn_e, dim=-1, keepdim=True) + 1e-12)
            else:
                ee = (e_feat * attn_e).sum(dim=-1).unsqueeze(-1) #/ (th.norm(e_feat, dim=-1, keepdim=True) * th.norm(attn_e, dim=-1, keepdim=True) + 1e-12)
            if attn_l == None:
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_src, dim=-1, keepdim=True) * th.norm(self.attn_l, dim=-1, keepdim=True) + 1e-12) #/ np.sqrt(self._out_feats)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_dst, dim=-1, keepdim=True) * th.norm(self.attn_r, dim=-1, keepdim=True) + 1e-12)#/ np.sqrt(self._out_feats)
            else:
                el = (feat_src * attn_l).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_src, dim=-1, keepdim=True) * th.norm(attn_l, dim=-1, keepdim=True) + 1e-12) #/ np.sqrt(self._out_feats)
                er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_dst, dim=-1, keepdim=True) * th.norm(attn_r, dim=-1, keepdim=True) + 1e-12)#/ np.sqrt(self._out_feats)


            graph.edata.update({'ee': ee})
            """
            # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) 
            el = feat_src
            er = th.zeros(feat_dst.shape).to(feat_src.device)
            graph.srcdata.update({'ft': feat_src})
            graph.dstdata.update({'ft_dst': feat_dst})
            graph.srcdata.update({'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            ee = self.fc_e(e_feat).view(e_feat.shape[0], self._num_heads, -1)

            graph.edata.update({'ee': ee})
            
            #e = self.leaky_relu((graph.edata.pop('e')+graph.edata.pop('ee')))
            # e = th.tanh((graph.edata.pop('e')+graph.edata.pop('ee')))
            e = (graph.edata.pop('ee') * graph.edata.pop('e')).sum(dim=-1).unsqueeze(-1)
            
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            attn_sum = th.ones((feat_src.shape[0], self._num_heads, 1)).to(feat_src.device)
            graph.srcdata.update({'attn_sum': attn_sum})
            graph.update_all(fn.u_mul_e('attn_sum', 'a', 'm2'),
                         fn.sum('m2', 'attn_sum'))
            attn_sum = graph.dstdata['attn_sum']

            #rst = graph.dstdata['ft']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']/(attn_sum + 1e-12)
            # bias
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()



class myGATConv_s_attn(nn.Module):
    def __init__(self,
                 edge_feats,
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
                 alpha=0.):
        super(myGATConv_s_attn, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, num_heads * edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_e = nn.Parameter(th.FloatTensor(size=(edge_feats, num_heads)))
        self.fc_e = nn.Linear(edge_feats, num_heads)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads * out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        #nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, h0, attn_l, attn_r, attn_e, res_attn=None, detach_edge=False):
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
            """
            if attn_e == None:
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1) #/ (th.norm(e_feat, dim=-1, keepdim=True) * th.norm(attn_e, dim=-1, keepdim=True) + 1e-12)
            else:
                ee = (e_feat * attn_e).sum(dim=-1).unsqueeze(-1) #/ (th.norm(e_feat, dim=-1, keepdim=True) * th.norm(attn_e, dim=-1, keepdim=True) + 1e-12)
            if attn_l == None:
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_src, dim=-1, keepdim=True) * th.norm(self.attn_l, dim=-1, keepdim=True) + 1e-12) #/ np.sqrt(self._out_feats)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_dst, dim=-1, keepdim=True) * th.norm(self.attn_r, dim=-1, keepdim=True) + 1e-12)#/ np.sqrt(self._out_feats)
            else:
                el = (feat_src * attn_l).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_src, dim=-1, keepdim=True) * th.norm(attn_l, dim=-1, keepdim=True) + 1e-12) #/ np.sqrt(self._out_feats)
                er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(-1) #/ (th.norm(feat_dst, dim=-1, keepdim=True) * th.norm(attn_r, dim=-1, keepdim=True) + 1e-12)#/ np.sqrt(self._out_feats)


            graph.edata.update({'ee': ee})
            """
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) 
            graph.srcdata.update({'ft': feat_src})
            graph.dstdata.update({'ft_dst': feat_dst})
            graph.srcdata.update({'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            ee = self.fc_e(e_feat).view(e_feat.shape[0], self._num_heads, -1)
            graph.edata.update({'ee': ee})
            
            #e = self.leaky_relu((graph.edata.pop('e')+graph.edata.pop('ee')))
            # e = th.tanh((graph.edata.pop('e')+graph.edata.pop('ee')))
            e = (graph.edata.pop('ee') + graph.edata.pop('e'))
            
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            attn_sum = th.ones((feat_src.shape[0], self._num_heads, 1)).to(feat_src.device)
            graph.srcdata.update({'attn_sum': attn_sum})
            graph.update_all(fn.u_mul_e('attn_sum', 'a', 'm2'),
                         fn.sum('m2', 'attn_sum'))
            attn_sum = graph.dstdata['attn_sum']

            #rst = graph.dstdata['ft']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']/(attn_sum + 1e-12)
            # bias
            if self.bias:
                rst = rst + self.bias_param
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()





class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 mcdropedge=0.0,
                 simplify=False):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        self.simplify = simplify
        self.mcdropedge = mcdropedge
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads * out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
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
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            attn = edge_softmax(graph, e)#self.attn_drop(edge_softmax(graph, e))
            attn = attn.repeat(1,1,self._out_feats)
            graph.edata['a'] = F.dropout(attn, p=self.mcdropedge, training=self.training)
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()



class myGATConv_noatt(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv_noatt, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        #self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        #self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads * out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        #nn.init.xavier_normal_(self.attn_l, gain=gain)
        #nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
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
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            #el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            #er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src})
            #graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            #graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('ee'))
            # e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = edge_softmax(graph, e)#self.attn_drop(edge_softmax(graph, e))
            
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()




class myGATConv_rel(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv_rel, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        nn.init.xavier_normal_(self.fc_e.weight, gain=1.414)
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.edge_emb = nn.Embedding(num_etypes, out_feats * num_heads)

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None, detach_edge=False):
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
                feat_src = self.fc_src(h_src)#.view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst)#.view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src)#.view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            graph.srcdata.update({'ft': feat_src})
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e_feat))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))

            rst = graph.dstdata['ft']#/(th.max(th.norm(graph.dstdata['ft'], dim=1, keepdim=True), 1e-12))
            print(rst.shape)
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()





