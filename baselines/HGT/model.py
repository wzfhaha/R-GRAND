import torch as th
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv
import torch.nn.functional as F
import ipdb
from dgl.ops import segment_mm, gather_mm
import math
import dgl.function as fn
from dgl.nn.pytorch import TypedLinear
from dgl.ops import edge_softmax
class HGTConv_s(nn.Module):
    r"""Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    <https://arxiv.org/abs/2003.01332>`__

    Given a graph :math:`G(V, E)` and input node features :math:`H^{(l-1)}`,
    it computes the new node features as follows:

    Compute a multi-head attention score for each edge :math:`(s, e, t)` in the graph:

    .. math::

      Attention(s, e, t) = \text{Softmax}\left(||_{i\in[1,h]}ATT-head^i(s, e, t)\right) \\
      ATT-head^i(s, e, t) = \left(K^i(s)W^{ATT}_{\phi(e)}Q^i(t)^{\top}\right)\cdot
        \frac{\mu_{(\tau(s),\phi(e),\tau(t)}}{\sqrt{d}} \\
      K^i(s) = \text{K-Linear}^i_{\tau(s)}(H^{(l-1)}[s]) \\
      Q^i(t) = \text{Q-Linear}^i_{\tau(t)}(H^{(l-1)}[t]) \\

    Compute the message to send on each edge :math:`(s, e, t)`:

    .. math::

      Message(s, e, t) = ||_{i\in[1, h]} MSG-head^i(s, e, t) \\
      MSG-head^i(s, e, t) = \text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\phi(e)} \\

    Send messages to target nodes :math:`t` and aggregate:

    .. math::

      \tilde{H}^{(l)}[t] = \sum_{\forall s\in \mathcal{N}(t)}\left( Attention(s,e,t)
      \cdot Message(s,e,t)\right)

    Compute new node features:

    .. math::

      H^{(l)}[t]=\text{A-Linear}_{\tau(t)}(\sigma(\tilde(H)^{(l)}[t])) + H^{(l-1)}[t]

    Parameters
    ----------
    in_size : int
        Input node feature size.
    head_size : int
        Output head size. The output node feature size is ``head_size * num_heads``.
    num_heads : int
        Number of heads. The output node feature size is ``head_size * num_heads``.
    num_ntypes : int
        Number of node types.
    num_etypes : int
        Number of edge types.
    dropout : optional, float
        Dropout rate.
    use_norm : optiona, bool
        If true, apply a layer norm on the output node feature.

    Examples
    --------
    """
    def __init__(self,
                 in_size,
                 head_size,
                 num_heads,
                 num_ntypes,
                 num_etypes,
                 dropout=0.2,
                 activation=F.relu,
                 use_norm=False,
                 residue=False):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm

        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = self.linear_k #TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = self.linear_k #TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(head_size * num_heads, head_size * num_heads, num_ntypes)

        self.relation_pri = nn.ParameterList([nn.Parameter(torch.ones(num_etypes))
                                              for i in range(num_heads)])
        self.relation_att = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes)
                                           for i in range(num_heads)])
        #self.relation_msg = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes)
        #                                   for i in range(num_heads)])
        self.relation_msg = self.relation_att
        self.residue = residue
        if self.residue:
            self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)
        self.activation = activation
        if use_norm:
            self.norm = nn.LayerNorm(head_size * num_heads)
        if in_size != head_size * num_heads:
            self.residual_w = nn.Parameter(torch.Tensor(in_size, head_size * num_heads))
            nn.init.xavier_uniform_(self.residual_w)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        x : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        self.presorted = presorted
        with g.local_scope():
            k = self.linear_k(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
            q = self.linear_q(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
            v = self.linear_v(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
            g.srcdata['k'] = k
            g.dstdata['q'] = q
            g.srcdata['v'] = v
            g.edata['etype'] = etype
            g.apply_edges(self.message)
            g.edata['m'] = g.edata['m'] * edge_softmax(g, g.edata['a']).unsqueeze(-1)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
            h = g.dstdata['h'].view(-1, self.num_heads * self.head_size)

            h = self.linear_a(h, ntype, presorted)
            if self.activation != None:
                h = self.activation(h)
            # target-specific aggregation
            h = self.drop(h)
            if self.residue:
                alpha = torch.sigmoid(self.skip[ntype]).unsqueeze(-1)

                if x.shape != h.shape:
                    h = h * alpha + (x @ self.residual_w) * (1 - alpha)
                else:
                    h = h * alpha + x * (1 - alpha)
            
            if self.use_norm:
                h = self.norm(h)
            return h


    def message(self, edges):
        """Message function."""
        a, m = [], []
        etype = edges.data['etype']
        k = torch.unbind(edges.src['k'], dim=1)
        q = torch.unbind(edges.dst['q'], dim=1)
        v = torch.unbind(edges.src['v'], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)  # (E, O)
            a.append((kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d)   # (E,)
            # a.append(self.relation_pri[i][etype])
            m.append(self.relation_msg[i](v[i], etype, self.presorted))  # (E, O)
        return {'a' : torch.stack(a, dim=1), 'm' : torch.stack(m, dim=1)}



class HGTConv(nn.Module):
    r"""Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    <https://arxiv.org/abs/2003.01332>`__

    Given a graph :math:`G(V, E)` and input node features :math:`H^{(l-1)}`,
    it computes the new node features as follows:

    Compute a multi-head attention score for each edge :math:`(s, e, t)` in the graph:

    .. math::

      Attention(s, e, t) = \text{Softmax}\left(||_{i\in[1,h]}ATT-head^i(s, e, t)\right) \\
      ATT-head^i(s, e, t) = \left(K^i(s)W^{ATT}_{\phi(e)}Q^i(t)^{\top}\right)\cdot
        \frac{\mu_{(\tau(s),\phi(e),\tau(t)}}{\sqrt{d}} \\
      K^i(s) = \text{K-Linear}^i_{\tau(s)}(H^{(l-1)}[s]) \\
      Q^i(t) = \text{Q-Linear}^i_{\tau(t)}(H^{(l-1)}[t]) \\

    Compute the message to send on each edge :math:`(s, e, t)`:

    .. math::

      Message(s, e, t) = ||_{i\in[1, h]} MSG-head^i(s, e, t) \\
      MSG-head^i(s, e, t) = \text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\phi(e)} \\

    Send messages to target nodes :math:`t` and aggregate:

    .. math::

      \tilde{H}^{(l)}[t] = \sum_{\forall s\in \mathcal{N}(t)}\left( Attention(s,e,t)
      \cdot Message(s,e,t)\right)

    Compute new node features:

    .. math::

      H^{(l)}[t]=\text{A-Linear}_{\tau(t)}(\sigma(\tilde(H)^{(l)}[t])) + H^{(l-1)}[t]

    Parameters
    ----------
    in_size : int
        Input node feature size.
    head_size : int
        Output head size. The output node feature size is ``head_size * num_heads``.
    num_heads : int
        Number of heads. The output node feature size is ``head_size * num_heads``.
    num_ntypes : int
        Number of node types.
    num_etypes : int
        Number of edge types.
    dropout : optional, float
        Dropout rate.
    use_norm : optiona, bool
        If true, apply a layer norm on the output node feature.

    Examples
    --------
    """
    def __init__(self,
                 in_size,
                 head_size,
                 num_heads,
                 num_ntypes,
                 num_etypes,
                 dropout=0.2,
                 activation=F.relu,
                 use_norm=False,
                 residue=False):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm
        self.activation = activation
        self.residue = residue
        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(head_size * num_heads, head_size * num_heads, num_ntypes)

        self.relation_pri = nn.ParameterList([nn.Parameter(torch.ones(num_etypes))
                                              for i in range(num_heads)])
        self.relation_att = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes)
                                           for i in range(num_heads)])
        self.relation_msg = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes)
                                           for i in range(num_heads)])
        if self.residue:
            self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norm = nn.LayerNorm(head_size * num_heads)
        if in_size != head_size * num_heads:
            self.residual_w = nn.Parameter(torch.Tensor(in_size, head_size * num_heads))
            nn.init.xavier_uniform_(self.residual_w)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        x : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        self.presorted = presorted
        with g.local_scope():
            k = self.linear_k(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
            q = self.linear_q(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
            v = self.linear_v(x, ntype, presorted).view(-1, self.num_heads, self.head_size)
            g.srcdata['k'] = k
            g.dstdata['q'] = q
            g.srcdata['v'] = v
            g.edata['etype'] = etype
            g.apply_edges(self.message)
            g.edata['m'] = g.edata['m'] * edge_softmax(g, g.edata['a']).unsqueeze(-1)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
            h = g.dstdata['h'].view(-1, self.num_heads * self.head_size)
            h = self.linear_a(h, ntype, presorted)
            if self.activation != None:
                h = self.activation(h)
            # target-specific aggregation
            h = self.drop(h)
            if self.residue:
                alpha = torch.sigmoid(self.skip[ntype]).unsqueeze(-1)
                if x.shape != h.shape:
                    h = h * alpha + (x @ self.residual_w) * (1 - alpha)
                else:
                    h = h * alpha + x * (1 - alpha)
            if self.use_norm:
                h = self.norm(h)
            return h

    def message(self, edges):
        """Message function."""
        a, m = [], []
        etype = edges.data['etype']
        k = torch.unbind(edges.src['k'], dim=1)
        q = torch.unbind(edges.dst['q'], dim=1)
        v = torch.unbind(edges.src['v'], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)  # (E, O)
            a.append((kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d)   # (E,)
            m.append(self.relation_msg[i](v[i], etype, self.presorted))  # (E, O)
        return {'a' : torch.stack(a, dim=1), 'm' : torch.stack(m, dim=1)}


class BaseHGT(nn.Module):
    def __init__(self, in_dims, h_dim, out_dim, num_heads, node_type, edge_type,
                 num_hidden_layers=1, dropout=0,
                 simplify=False, original = False, layer_norm=True, residue=False):
        super(BaseHGT, self).__init__()
        self.in_dims = in_dims
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.edge_type = edge_type
        self.node_type = node_type
        # self.num_bases = num_rels if num_bases < 0 else num_bases
        print(f'edge type num {self.edge_type}')
        print(f'node type num {self.node_type}')
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.simplify = simplify        
        self.layer_norm = layer_norm
        self.original = original
        self.residue = residue
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
        self.out_layer = h2o
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

    def forward(self, g, features_list, ntype, etype):
        h = []
        for i2h, feature in zip(self.i2h, features_list):
            h.append(i2h(feature))
        h = th.cat(h, 0)
        # ipdb.set_trace()
        for layer in self.layers:
            # h = F.dropout(h, self.dropout, training=self.training)
            h = layer(g, h, ntype, etype, presorted=True)
        h = self.out_layer(h)
        h = torch.reshape(h, (ntype.shape[0], -1, self.out_dim))
        h = h.mean(1)
        #ipdb.set_trace()
        h = h/(th.max(th.norm(h, dim=1, keepdim=True), th.FloatTensor([1e-12]).to(h.device)))
        return h
