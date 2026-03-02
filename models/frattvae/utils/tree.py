import torch
import numpy as np
import logging

_logger = logging.getLogger(__name__)

"""
Reference: https://github.com/microsoft/icecaps/blob/master/icecaps/util/trees.py
           https://github.com/inyukwo1/tree-lstm/blob/master/tree_lstm/tree_lstm.py

DGL replaced with a pure PyTorch/Python implementation so that the module works
on Python 3.13+ where no DGL wheel is available.
"""


# ---------------------------------------------------------------------------
# Minimal graph object that mimics the DGL API used by FragmentTree
# ---------------------------------------------------------------------------

class _SimpleGraph:
    """
    Lightweight directed graph that mirrors the DGL DGLGraph interface used by
    FragmentTree.  Edges are stored as (src, dst) lists; per-node and per-edge
    features are stored in plain dicts of Tensors.
    """

    def __init__(self):
        # Node features: key → Tensor(N, ...)
        self._ndata: dict = {}
        # Edge endpoints
        self._src: list = []
        self._dst: list = []
        # Edge features: key → list of Tensors, each shape (1, ...)
        self._edata_list: dict = {}

    # ── node count ────────────────────────────────────────────────────────────

    def num_nodes(self) -> int:
        if not self._ndata:
            return 0
        return next(iter(self._ndata.values())).shape[0]

    def number_of_nodes(self) -> int:
        return self.num_nodes()

    def nodes(self):
        return torch.arange(self.num_nodes())

    # ── add / remove nodes ───────────────────────────────────────────────────

    def add_nodes(self, n: int, data: dict = None):
        """Add *n* nodes.  *data* gives per-new-node features (shape (n, ...)).
        Existing feature keys that are absent from *data* are extended with zeros."""
        existing = self.num_nodes()
        # Extend existing keys not present in data
        for key in list(self._ndata.keys()):
            if data and key in data:
                continue
            cur = self._ndata[key]
            pad = cur.new_zeros((n,) + cur.shape[1:])
            self._ndata[key] = torch.cat([cur, pad], dim=0)
        # Add new keys from data
        for key, val in (data or {}).items():
            val = val.detach()
            if key not in self._ndata:
                if existing > 0:
                    pad = val.new_zeros((existing,) + val.shape[1:])
                    self._ndata[key] = torch.cat([pad, val], dim=0)
                else:
                    self._ndata[key] = val.clone()
            else:
                self._ndata[key] = torch.cat([self._ndata[key], val], dim=0)

    def remove_nodes(self, node_id: int):
        for key in list(self._ndata.keys()):
            t = self._ndata[key]
            self._ndata[key] = torch.cat([t[:node_id], t[node_id + 1:]], dim=0)
        new_src, new_dst, keep = [], [], []
        for i, (s, d) in enumerate(zip(self._src, self._dst)):
            if s == node_id or d == node_id:
                continue
            new_src.append(s - int(s > node_id))
            new_dst.append(d - int(d > node_id))
            keep.append(i)
        self._src, self._dst = new_src, new_dst
        for key in list(self._edata_list.keys()):
            self._edata_list[key] = [self._edata_list[key][i] for i in keep]

    # ── add / remove edges ───────────────────────────────────────────────────

    def add_edges(self, src, dst, data: dict = None):
        src = int(src) if isinstance(src, torch.Tensor) else src
        dst = int(dst) if isinstance(dst, torch.Tensor) else dst
        self._src.append(src)
        self._dst.append(dst)
        for key, val in (data or {}).items():
            if key not in self._edata_list:
                self._edata_list[key] = []
            self._edata_list[key].append(val.detach())

    def remove_edges(self, edge_id: int):
        self._src.pop(edge_id)
        self._dst.pop(edge_id)
        for key in list(self._edata_list.keys()):
            self._edata_list[key].pop(edge_id)

    def all_edges(self):
        srcs = torch.tensor(self._src, dtype=torch.long)
        dsts = torch.tensor(self._dst, dtype=torch.long)
        return srcs, dsts

    # ── adjacency queries ────────────────────────────────────────────────────

    def predecessors(self, nid) -> torch.Tensor:
        """Nodes u such that edge u→nid exists (i.e. children when edges go child→parent)."""
        nid = int(nid) if isinstance(nid, torch.Tensor) else nid
        return torch.tensor([s for s, d in zip(self._src, self._dst) if d == nid],
                            dtype=torch.long)

    def successors(self, nid) -> torch.Tensor:
        """Nodes v such that edge nid→v exists (i.e. parent when edges go child→parent)."""
        nid = int(nid) if isinstance(nid, torch.Tensor) else nid
        return torch.tensor([d for s, d in zip(self._src, self._dst) if s == nid],
                            dtype=torch.long)

    # ── data properties ──────────────────────────────────────────────────────

    @property
    def ndata(self) -> dict:
        return self._ndata

    @ndata.setter
    def ndata(self, value: dict):
        self._ndata = value

    @property
    def edata(self) -> dict:
        """Returns a dict of stacked edge-feature tensors (E, ...)."""
        result = {}
        for key, vlist in self._edata_list.items():
            if vlist:
                result[key] = torch.cat(vlist, dim=0)
            else:
                result[key] = torch.empty(0)
        return result

    # ── device / copy ────────────────────────────────────────────────────────

    def to(self, device):
        self._ndata = {k: v.to(device) for k, v in self._ndata.items()}
        for k, vlist in self._edata_list.items():
            self._edata_list[k] = [v.to(device) for v in vlist]
        return self

    def clone(self):
        g = _SimpleGraph()
        g._ndata = {k: v.clone() for k, v in self._ndata.items()}
        g._src = list(self._src)
        g._dst = list(self._dst)
        g._edata_list = {k: [v.clone() for v in vlist] for k, vlist in self._edata_list.items()}
        return g


def _reverse_graph(g: _SimpleGraph, copy_ndata=True, copy_edata=True) -> _SimpleGraph:
    """Equivalent to dgl.reverse."""
    rg = _SimpleGraph()
    rg._src = list(g._dst)
    rg._dst = list(g._src)
    if copy_ndata:
        rg._ndata = {k: v.clone() for k, v in g._ndata.items()}
    if copy_edata:
        rg._edata_list = {k: [v.clone() for v in vlist] for k, vlist in g._edata_list.items()}
    return rg


class FragmentTree:
    def __init__(self, dgl_graph=None):
        self.dgl_graph = dgl_graph if dgl_graph else _SimpleGraph()
        self.max_depth = 0
        self.max_degree = 0

    def add_node(self, parent_id=None, feature: torch.Tensor = torch.Tensor(), fid: int = -1, bondtype: int = 1, data: dict = None):
        if data is None:
            data = {
                'x': feature.unsqueeze(0),
                'fid': torch.tensor([fid]).unsqueeze(0)
            }
        self.dgl_graph.add_nodes(1, data=data)
        added_node_id = self.dgl_graph.number_of_nodes() - 1

        if parent_id is not None:
            self.dgl_graph.ndata['level'][added_node_id] = self.dgl_graph.ndata['level'][parent_id] + 1
            self.dgl_graph.add_edges(added_node_id, parent_id, data={'w': torch.tensor([bondtype]).unsqueeze(0)})
            self.max_degree = max(self.max_degree, len(self.dgl_graph.predecessors(parent_id)))
        elif added_node_id > 0:
            self.dgl_graph.ndata['level'][added_node_id] = torch.tensor([0]).int()
        else:
            self.dgl_graph.ndata['level'] = torch.tensor([0]).int()

        self.max_depth = self.dgl_graph.ndata['level'].max().item()
        return added_node_id

    def add_link(self, child_id, parent_id, bondtype: int = 1):
        self.dgl_graph.add_edges(child_id, parent_id, data={'w': torch.tensor([bondtype]).unsqueeze(0)})

    def remove_node(self, node_id: int):
        self.dgl_graph.remove_nodes(node_id)

    def remove_edge(self, edge_id: int):
        self.dgl_graph.remove_edges(edge_id)

    def adjacency_matrix(self):
        n_node = self.dgl_graph.num_nodes()
        if n_node < 2:
            adj = torch.tensor([[0]])
        else:
            indices = torch.stack(self.dgl_graph.all_edges())
            values = self.dgl_graph.edata['w'].squeeze()
            adj = torch.sparse_coo_tensor(indices, values, size=(n_node, n_node)).to_dense()
        return adj

    def to(self, device: str = 'cpu'):
        self.dgl_graph = self.dgl_graph.to(device)
        return self

    def reverse(self):
        self.dgl_graph = _reverse_graph(self.dgl_graph, copy_ndata=True, copy_edata=True)
        return self

    def set_all_positional_encoding(self, d_pos: int = None, n: int = None):
        """If n is not None, encode as if all nodes have n children."""
        d_pos = d_pos if d_pos else self.max_depth * self.max_degree
        self.dgl_graph.ndata['pos'] = torch.zeros(self.dgl_graph.num_nodes(), d_pos)
        for nid in self.dgl_graph.nodes()[1:]:
            parent = self.dgl_graph.successors(nid)
            if len(parent) > 0:
                parent = parent[0]
            else:
                continue
            children = self.dgl_graph.predecessors(parent).tolist()

            n_used = n if n else len(children)
            assert n_used >= len(children)
            positional_encoding = [0.0 for _ in range(n_used)]
            positional_encoding[children.index(nid.item() if isinstance(nid, torch.Tensor) else nid)] = 1.0
            positional_encoding += self.dgl_graph.ndata['pos'][parent].tolist()

            self.dgl_graph.ndata['pos'][nid] = torch.tensor(positional_encoding)[:d_pos]

    def set_positional_encoding(self, nid: int, num_sibling: int = None, d_pos: int = None):
        d_pos = d_pos if d_pos else self.max_depth * self.max_degree

        parents = self.dgl_graph.successors(nid)
        if len(parents) == 0:
            self.dgl_graph.ndata['pos'] = torch.zeros(self.dgl_graph.num_nodes(), d_pos)
            positional_encoding = [0.0] * d_pos
        else:
            parent = parents[0]
            sibling = self.dgl_graph.predecessors(parent).tolist()
            num_sibling = num_sibling if num_sibling is not None else len(sibling)
            assert num_sibling >= len(sibling)

            positional_encoding = [0.0 for _ in range(num_sibling)]
            positional_encoding[sibling.index(nid)] = 1.0
            positional_encoding += self.dgl_graph.ndata['pos'][parent].tolist()

        self.dgl_graph.ndata['pos'][nid] = torch.tensor(positional_encoding)[:d_pos]

    def width(self, level: int):
        return (self.dgl_graph.ndata['level'] == level).sum()


class BatchedFragmentTree:
    """
    Holds a list of FragmentTree objects whose set_all_positional_encoding has
    been called.  Provides get_ndata / get_tree_list / to / reverse compatible
    with the original DGL-based implementation.
    """

    def __init__(self, tree_list, max_depth: int = None, max_degree: int = None):
        depth_list, degree_list = zip(*[(tree.max_depth, tree.max_degree) for tree in tree_list])
        if (max_depth is None) or (max_degree is None):
            self.max_depth = max(depth_list)
            self.max_degree = max(degree_list)
        else:
            if (max_depth < max(depth_list)) or (max_degree < max(degree_list)):
                _logger.warning(f'[FRATTVAE] max depth:{max_depth} < {max(depth_list)} or max degree:{max_degree} < {max(degree_list)}')
            self.max_depth = max_depth
            self.max_degree = max_degree

        d_pos = self.max_depth * self.max_degree
        self._graphs: list = []
        for tree in tree_list:
            tree.set_all_positional_encoding(d_pos=d_pos)
            self._graphs.append(tree.dgl_graph.clone())

    def get_ndata(self, key: str, node_ids: list = None, pad_value: int = 0):
        max_nodes_num = max(g.num_nodes() for g in self._graphs)
        ndatas = []
        for i, graph in enumerate(self._graphs):
            if node_ids:
                node_id = node_ids[i] if i < len(node_ids) else node_ids[0]
                states = graph.ndata[key][node_id]
            else:
                states = graph.ndata[key]
                node_num, state_num = states.size()
                if node_num < max_nodes_num:
                    padding = states.new_full((max_nodes_num - node_num, state_num), pad_value)
                    states = torch.cat((states, padding), dim=0)
            ndatas.append(states)
        return torch.stack(ndatas)

    def get_tree_list(self):
        return [FragmentTree(dgl_graph=g.clone()) for g in self._graphs]

    def to(self, device: str = 'cpu'):
        self._graphs = [g.to(device) for g in self._graphs]
        return self

    def reverse(self):
        self._graphs = [_reverse_graph(g, copy_ndata=True, copy_edata=True) for g in self._graphs]
        return self


def make_tree(frag_indices: list, ecfps: torch.Tensor, bond_types: list, bondMapNums: list, d_pos: int = None) -> FragmentTree:
    """
    frag_indices: list of fragment indices
    ecfps: ecfps of fragments, shape= (len(frag_indices), n_bits)
    bond_types: list of bondtype (1: single, 2: double, 3: triple)
    bondMapNums: list of connection order lists.
    d_pos: dimension of positional encoding
    """
    if type(ecfps) == list:
        ecfps = torch.tensor(ecfps).float()

    tree = FragmentTree()
    tree.add_node(parent_id=None, feature=ecfps[0], fid=frag_indices[0], bondtype=0)

    stack = [0]
    node_ids = [0] * len(frag_indices)
    while max(map(len, bondMapNums)) > 0:
        if stack:
            parent = stack[-1]
            pid = node_ids[parent]
            if bondMapNums[parent]:
                b = bondMapNums[parent].pop(0)
            else:
                stack.pop(-1)
                continue
        else:
            idx = [i for i in range(len(frag_indices)) if len(bondMapNums[i]) > 0][0]
            stack.append(idx)
            add_node_id = tree.add_node(parent_id=None, feature=ecfps[idx], fid=frag_indices[idx], bondtype=0)
            node_ids[idx] = add_node_id
            continue

        child_list = [b in mapnums for mapnums in bondMapNums]
        if np.any(child_list):
            c = child_list.index(True)
            add_node_id = tree.add_node(parent_id=pid, feature=ecfps[c], fid=frag_indices[c], bondtype=bond_types[b - 1])
            node_ids[c] = add_node_id
            stack.append(c)

    if d_pos:
        tree.set_all_positional_encoding(d_pos)

    return tree


def get_tree_features(frag_indices: list, ecfps: torch.Tensor, bond_types: list, bondMapNums: list,
                      max_depth: int = None, max_degree: int = None, free_n: bool = False):
    tree = make_tree(frag_indices, ecfps, bond_types, bondMapNums)

    max_depth = max_depth if max_depth else tree.max_depth
    max_degree = max_degree if max_degree else tree.max_degree

    if (max_depth < tree.max_depth) or (max_degree < tree.max_degree):
        _logger.debug(f'[FRATTVAE] tree depth {tree.max_depth} > max_depth {max_depth} '
                      f'or degree {tree.max_degree} > max_degree {max_degree}; truncating.')

    n = None if free_n else max_degree
    tree.set_all_positional_encoding(d_pos=max_depth * max_degree, n=n)
    fids = tree.dgl_graph.ndata['fid'].squeeze(-1)
    positions = tree.dgl_graph.ndata['pos']
    features = tree.dgl_graph.ndata['x']

    return fids, features, positions


def get_pad_features(tree_list, key: str, max_nodes_num: int):
    ndatas = []
    for tree in tree_list:
        states = tree.dgl_graph.ndata[key]
        node_num, state_num = states.size()
        if len(states) < max_nodes_num:
            padding = states.new_full((max_nodes_num - node_num, state_num), 0)
            states = torch.cat((states, padding), dim=0)
        ndatas.append(states)
    return torch.stack(ndatas)
