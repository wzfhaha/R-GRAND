import torch as th
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv
import torch.nn.functional as F
import ipdb
from dgl.ops import segment_mm, gather_mm
import math
import dgl.function as fn

class TypedLinear(nn.Module):
    r"""Linear transformation according to types.

    For each sample of the input batch :math:`x \in X`, apply linear transformation
    :math:`xW_t`, where :math:`t` is the type of :math:`x`.

    The module supports two regularization methods (basis-decomposition and
    block-diagonal-decomposition) proposed by "`Modeling Relational Data
    with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"

    The basis regularization decomposes :math:`W_t` by:

    .. math::

       W_t^{(l)} = \sum_{b=1}^B a_{tb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{tb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_t` into :math:`B`
    block-diagonal matrices. We refer to :math:`B` as the number of bases:

    .. math::

       W_t^{(l)} = \oplus_{b=1}^B Q_{tb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{tb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)\times(d^{l}/B)}`.

    Parameters
    ----------
    in_size : int
        Input feature size.
    out_size : int
        Output feature size.
    num_types : int
        Total number of types.
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":

         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.

        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Typically smaller
        than ``num_types``.
        Default: ``None``.

    Examples
    --------

    No regularization.

    >>> from dgl.nn import TypedLinear
    >>> import torch
    >>>
    >>> x = torch.randn(100, 32)
    >>> x_type = torch.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    torch.Size([100, 64])

    With basis regularization

    >>> x = torch.randn(100, 32)
    >>> x_type = torch.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5, regularizer='basis', num_bases=4)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    torch.Size([100, 64])
    """
    def __init__(self, in_size, out_size, num_types,
                 regularizer=None, num_bases=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_types = num_types
        if regularizer is None:
            self.W = nn.Parameter(torch.Tensor(num_types, in_size, out_size))
        elif regularizer == 'basis':
            if num_bases is None:
                raise ValueError('Missing "num_bases" for basis regularization.')
            self.W = nn.Parameter(torch.Tensor(num_bases, in_size, out_size))
            self.coeff = nn.Parameter(torch.Tensor(num_types, num_bases))
            self.num_bases = num_bases
        elif regularizer == 'bdd':
            if num_bases is None:
                raise ValueError('Missing "num_bases" for bdd regularization.')
            if in_size % num_bases != 0 or out_size % num_bases != 0:
                raise ValueError(
                    'Input and output sizes must be divisible by num_bases.'
                )
            self.submat_in = in_size // num_bases
            self.submat_out = out_size // num_bases
            self.W = nn.Parameter(torch.Tensor(
                num_types, num_bases * self.submat_in * self.submat_out))
            self.num_bases = num_bases
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}')
        self.regularizer = regularizer
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters"""
        with torch.no_grad():
            # Follow torch.nn.Linear 's initialization to use kaiming_uniform_ on in_size
            if self.regularizer is None:
                nn.init.uniform_(self.W, -1/math.sqrt(self.in_size), 1/math.sqrt(self.in_size))
            elif self.regularizer == 'basis':
                nn.init.uniform_(self.W, -1/math.sqrt(self.in_size), 1/math.sqrt(self.in_size))
                nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('relu'))
            elif self.regularizer == 'bdd':
                nn.init.uniform_(self.W, -1/math.sqrt(self.submat_in), 1/math.sqrt(self.submat_in))
            else:
                raise ValueError(
                    f'Supported regularizer options: "basis", "bdd", but got {regularizer}')

    def get_weight(self):
        """Get type-wise weight"""
        if self.regularizer is None:
            return self.W
        elif self.regularizer == 'basis':
            W = self.W.view(self.num_bases, self.in_size * self.out_size)
            return (self.coeff @ W).view(self.num_types, self.in_size, self.out_size)
        elif self.regularizer == 'bdd':
            return self.W
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}')

    def forward(self, x, x_type, sorted_by_type=False):
        """Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            A 2D input tensor. Shape: (N, D1)
        x_type : torch.Tensor
            A 1D integer tensor storing the type of the elements in ``x`` with one-to-one
            correspondenc. Shape: (N,)
        sorted_by_type : bool, optional
            Whether the inputs have been sorted by the types. Forward on pre-sorted inputs may
            be faster.

        Returns
        -------
        y : torch.Tensor
            The transformed output tensor. Shape: (N, D2)
        """
        w = self.get_weight()
        if self.regularizer == 'bdd':
            w = w.index_select(0, x_type).view(-1, self.submat_in, self.submat_out)
            x = x.view(-1, 1, self.submat_in)
            return torch.bmm(x, w).view(-1, self.out_size)
        elif sorted_by_type:
            pos_l = torch.searchsorted(x_type, torch.arange(self.num_types, device=x.device))
            pos_r = torch.cat([pos_l[1:], torch.tensor([len(x_type)], device=x.device)])
            seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize
            return segment_mm(x, w, seglen_a=seglen)
        else:
            return gather_mm(x, w, idx_b=x_type)

    def __repr__(self):
        if self.regularizer is None:
            return (f'TypedLinear(in_size={self.in_size}, out_size={self.out_size}, '
                    f'num_types={self.num_types})')
        else:
            return (f'TypedLinear(in_size={self.in_size}, out_size={self.out_size}, '
                    f'num_types={self.num_types}, regularizer={self.regularizer}, '
                    f'num_bases={self.num_bases})')

class TypedLinear_s(nn.Module):
    r"""
    Modified TypedLinear: add softmax normalization on coefficients
    Linear transformation according to types.

    For each sample of the input batch :math:`x \in X`, apply linear transformation
    :math:`xW_t`, where :math:`t` is the type of :math:`x`.

    The module supports two regularization methods (basis-decomposition and
    block-diagonal-decomposition) proposed by "`Modeling Relational Data
    with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"

    The basis regularization decomposes :math:`W_t` by:

    .. math::

       W_t^{(l)} = \sum_{b=1}^B a_{tb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{tb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_t` into :math:`B`
    block-diagonal matrices. We refer to :math:`B` as the number of bases:

    .. math::

       W_t^{(l)} = \oplus_{b=1}^B Q_{tb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{tb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)\times(d^{l}/B)}`.

    Parameters
    ----------
    in_size : int
        Input feature size.
    out_size : int
        Output feature size.
    num_types : int
        Total number of types.
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":

         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.

        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Typically smaller
        than ``num_types``.
        Default: ``None``.

    Examples
    --------

    No regularization.

    >>> from dgl.nn import TypedLinear
    >>> import torch
    >>>
    >>> x = torch.randn(100, 32)
    >>> x_type = torch.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    torch.Size([100, 64])

    With basis regularization

    >>> x = torch.randn(100, 32)
    >>> x_type = torch.randint(0, 5, (100,))
    >>> m = TypedLinear(32, 64, 5, regularizer='basis', num_bases=4)
    >>> y = m(x, x_type)
    >>> print(y.shape)
    torch.Size([100, 64])
    """
    def __init__(self, in_size, out_size, num_types,
                 regularizer=None, num_bases=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_types = num_types
        if regularizer is None:
            self.W = nn.Parameter(torch.Tensor(num_types, in_size, out_size))
        elif regularizer == 'basis':
            if num_bases is None:
                raise ValueError('Missing "num_bases" for basis regularization.')
            self.W = nn.Parameter(torch.Tensor(num_bases, in_size, out_size))
            self.coeff = nn.Parameter(torch.Tensor(num_types, num_bases))
            self.num_bases = num_bases
        elif regularizer == 'bdd':
            if num_bases is None:
                raise ValueError('Missing "num_bases" for bdd regularization.')
            if in_size % num_bases != 0 or out_size % num_bases != 0:
                raise ValueError(
                    'Input and output sizes must be divisible by num_bases.'
                )
            self.submat_in = in_size // num_bases
            self.submat_out = out_size // num_bases
            self.W = nn.Parameter(torch.Tensor(
                num_types, num_bases * self.submat_in * self.submat_out))
            self.num_bases = num_bases
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}')
        self.regularizer = regularizer
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters"""
        with torch.no_grad():
            # Follow torch.nn.Linear 's initialization to use kaiming_uniform_ on in_size
            if self.regularizer is None:
                nn.init.uniform_(self.W, -1/math.sqrt(self.in_size), 1/math.sqrt(self.in_size))
            elif self.regularizer == 'basis':
                nn.init.uniform_(self.W, -1/math.sqrt(self.in_size), 1/math.sqrt(self.in_size))
                nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('relu'))
            elif self.regularizer == 'bdd':
                nn.init.uniform_(self.W, -1/math.sqrt(self.submat_in), 1/math.sqrt(self.submat_in))
            else:
                raise ValueError(
                    f'Supported regularizer options: "basis", "bdd", but got {regularizer}')

    def get_weight(self):
        """Get type-wise weight"""
        if self.regularizer is None:
            return self.W
        elif self.regularizer == 'basis':
            W = self.W.view(self.num_bases, self.in_size * self.out_size)
            return (F.softmax(self.coeff, dim=0) @ W).view(self.num_types, self.in_size, self.out_size)
        elif self.regularizer == 'bdd':
            return self.W
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", "bdd", but got {regularizer}')

    def forward(self, x, x_type, sorted_by_type=False):
        """Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            A 2D input tensor. Shape: (N, D1)
        x_type : torch.Tensor
            A 1D integer tensor storing the type of the elements in ``x`` with one-to-one
            correspondenc. Shape: (N,)
        sorted_by_type : bool, optional
            Whether the inputs have been sorted by the types. Forward on pre-sorted inputs may
            be faster.

        Returns
        -------
        y : torch.Tensor
            The transformed output tensor. Shape: (N, D2)
        """
        w = self.get_weight()
        if self.regularizer == 'bdd':
            w = w.index_select(0, x_type).view(-1, self.submat_in, self.submat_out)
            x = x.view(-1, 1, self.submat_in)
            return torch.bmm(x, w).view(-1, self.out_size)
        elif sorted_by_type:
            pos_l = torch.searchsorted(x_type, torch.arange(self.num_types, device=x.device))
            pos_r = torch.cat([pos_l[1:], torch.tensor([len(x_type)], device=x.device)])
            seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize
            return segment_mm(x, w, seglen_a=seglen)
        else:
            return gather_mm(x, w, idx_b=x_type)

    def __repr__(self):
        if self.regularizer is None:
            return (f'TypedLinear(in_size={self.in_size}, out_size={self.out_size}, '
                    f'num_types={self.num_types})')
        else:
            return (f'TypedLinear(in_size={self.in_size}, out_size={self.out_size}, '
                    f'num_types={self.num_types}, regularizer={self.regularizer}, '
                    f'num_bases={self.num_bases})')


class RelGraphConv_s(nn.Module):
    r"""Relational graph convolution layer from `Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__

    It can be described in as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}e_{j,i}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`e_{j,i}` is the normalizer. :math:`\sigma` is an activation
    function. :math:`W_0` is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":

         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.

        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> etype = th.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer=None,
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False):
        super().__init__()
        self.linear_r = TypedLinear_s(in_feat, out_feat, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        """Message function."""
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        if 'norm' in edges.data:
            m = m * edges.data['norm']
        return {'m' : m}

    def forward(self, g, feat, etypes, norm=None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            g.edata['etype'] = etypes
            # message passing
            g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata['h']
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[:g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h

class BaseRGCN(nn.Module):
    def __init__(self, in_dims, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False, simplify=False, original = False, layer_norm=True):
        super(BaseRGCN, self).__init__()
        self.in_dims = in_dims
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_rels if num_bases < 0 else num_bases
        print(f'num rels {self.num_rels}')
        print(f'num bases {self.num_bases}')
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.simplify = simplify        
        self.layer_norm = layer_norm
        self.original = original
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
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, features_list, r, norm):
        h = []
        for i2h, feature in zip(self.i2h, features_list):
            h.append(i2h(feature))
        h = th.cat(h, 0)
        for layer in self.layers:
            # h = F.dropout(h, self.dropout, training=self.training)
            h = layer(g, h, r, norm)

        #ipdb.set_trace()
        h = h / (th.max(th.norm(h, dim=1, keepdim=True), th.FloatTensor([1e-12]).to(h.device)))
        return h

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    embed_name : str, optional
        Embed name
    """

    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 sparse_emb=False,
                 embed_name='embed'):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = dev_id
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.num_nodes = num_nodes
        self.sparse_emb = sparse_emb

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.num_of_ntype = num_of_ntype
        self.idmap = th.empty(num_nodes).long()

        for ntype in range(num_of_ntype):
            if input_size[ntype] is not None:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(
                    th.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

        self.node_embeds = th.nn.Embedding(
            node_tids.shape[0], self.embed_size, sparse=self.sparse_emb)
        nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.node_embeds.weight.device)
        embeds = th.empty(node_ids.shape[0],
                          self.embed_size, device=self.dev_id)
        for ntype in range(self.num_of_ntype):
            if features[ntype] is not None:
                loc = node_tids == ntype
                embeds[loc] = features[ntype][type_ids[loc]].to(
                    self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id)
            else:
                loc = node_tids == ntype
                embeds[loc] = self.node_embeds(tsd_ids[loc]).to(self.dev_id)

        return embeds




class RelGraphConv(nn.Module):
    r"""Relational graph convolution layer from `Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__

    It can be described in as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}e_{j,i}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`e_{j,i}` is the normalizer. :math:`\sigma` is an activation
    function. :math:`W_0` is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str, optional
        Which weight regularizer to use "basis" or "bdd":

         - "basis" is short for basis-decomposition.
         - "bdd" is short for block-diagonal-decomposition.

        Default applies no regularization.
    num_bases : int, optional
        Number of bases. Needed when ``regularizer`` is specified. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> etype = th.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer=None,
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False,
                 mcdropedge=0.0):
        super().__init__()
        self.linear_r = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.mcdropedge = mcdropedge

    def message(self, edges):
        """Message function."""
        m = self.linear_r(edges.src['h'], edges.data['etype'], self.presorted)
        if 'norm' in edges.data:
            m = m * edges.data['norm']
        # print(m.shape)
        m = F.dropout(m, p=self.mcdropedge, training=self.training)
        return {'m' : m}

    def forward(self, g, feat, etypes, norm=None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            g.edata['etype'] = etypes
            # message passing
            #print(g.edata['m'])
            # print(etypes.shape)
            g.update_all(self.message, fn.sum('m', 'h'))
            # apply bias and activation
            h = g.dstdata['h']
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[:g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
