#!/bin/python3.6
from math import ceil, log2

import networkx as nx
import itertools
import random
from typing import Union, Tuple, Dict, List, Iterator, FrozenSet, TextIO, Set, \
    Any, Iterable
import sys, os
from operator import itemgetter
from functools import reduce
from collections import OrderedDict

import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


# general purpose functions

def first(obj):
    """return first element from object
    (also consumes it if obj is an iterator)"""
    return next(iter(obj))


def pick(obj):
    """randomly pick an element from obj"""
    return random.sample(obj, 1)[0]


def pairs(obj):
    """return all unordered pairs made from
    distinct elements of obj"""
    return itertools.combinations(obj, 2)#输出obj的所有可能的对 combinations方法可以轻松的实现排列组合。


def opairs(obj):
    """return all ordered pairs made from
    distinct elements of obj"""
    return itertools.permutations(obj, 2)


def ord_triples(obj):
    """return all ordered triples made from
    distinct elements of obj"""
    return itertools.permutations(obj, 3)


def elem_apply(fns, elems):
    """apply list of functions to list of elements,
    element-wise"""
    return map(lambda a, b: a(b), fns, elems)


def posdict_to_ordering(positions: dict):
    ordering = [-1]*len(positions)
    for elem, pos in positions.items():
        ordering[pos] = elem
    return ordering


def replicate(d: OrderedDict):
    """
    convert a dict with (element, count) into a list
    with each element replicated count many times
    """
    l = []
    for element, count in d.items():
        l.extend([element]*count)
    return l


def shuffled(l):
    s = [e for e in l]
    random.shuffle(s)
    return s

def minscore(res):
    list={}
    for id,bag_node in res.td.bags.items():
        #print(id)
        node_id=[]
        for node in bag_node:
            node_id.append(node)
        list[id]=node_id
    score={}
    for id ,node in list.items():
        score[id]=res.compute_score(node)
    min_score=min(score.values())
    #print(score)
    for id ,score in score.items():
        if score==min_score:
            return id
    #return list ,score,id
# i/o utility functions

def file_score_max(bn,set):
    offsets = dict()
    score=0
    for node, psets in stream_bn(bn.input_file, normalize=False):
        #print(node,psets)
        scores = psets.values()
        offsets[node] = max(scores)
    for node in set:
        score +=offsets[node]
    return score

def minscore_pre(res):
    list={}
    for id,bag_node in res.td.bags.items():
        #print(id)
        node_id=[]
        for node in bag_node:
            node_id.append(node)
        list[id]=node_id
    score={}

    for id ,node in list.items():
        score[id]=res.compute_score(node)-file_score_max(res,node)#-list(offsets.values())[node]
    min_score=max(score.values())
    #print(score)
    for id ,score in score.items():
        if score==min_score:
            return id

class FileReader(object):  # todo[safety]: add support for `with` usage
    def __init__(self, filename: str, ignore="#"):
        self.file = open(filename)
        self.ignore = ignore

    def readline(self):#得到jkl文件中不包含#的第一行
        line = self.file.readline()#while false 结束循环
        while line.startswith(self.ignore):#Python startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
            line = self.file.readline()
        return line

    def readints(self):
        return map(int, self.readline().split())

    def readint(self):
        return int(self.readline().strip())

    def close(self):
        self.file.close()


# bn datatypes
Psets = Dict[FrozenSet[int], float]
BNStream = Iterator[Tuple[int, Psets]]
BNData = Dict[int, Psets]

# constrained bn datatype
Constraints = Dict[str, List[Tuple]]
IntPairs = Iterable[Tuple[int, int]]


def stream_jkl(filename: str, normalize=True):
    reader = FileReader(filename, ignore="#")
    n = reader.readint()#20 test.jkl文件
    for i in range(n):
        psets: Psets = dict()
        minscore = 1e9
        node, numsets = reader.readints()# 变量 变量可能的父集数（0,6）
        for j in range(numsets):#第i个变量的可能父集的数量 6
            score, parents = reader.readline().split(sep=" ", maxsplit=1)#分割次数 score -1610.8712       parents  1 11
            score = float(score)
            parents = frozenset(map(int, parents.split()[1:]))# parents ：11
            psets[parents] = score
            minscore = min(score, minscore)
        if normalize:
            psets = {pset: score-minscore for pset, score in psets.items()}
        yield node, psets
        #print(node,psets)
    reader.close()
    #print( node,psets)
#输出 节点 和{父节点，分数减去最小的分数}

def read_jkl(filename: str, normalize=True):
    return dict(stream_jkl(filename, normalize))


def num_nodes_jkl(filename: str):
    reader = FileReader(filename, ignore="#")
    n = reader.readint()
    reader.close()
    return n


def write_jkl(data, filename):
    n = len(data)
    with open(filename, 'w') as outfile:
        outfile.write(f"{n}\n")
        for node, psets in sorted(data.items(), key=itemgetter(0)):
            outfile.write(f"{node} {len(psets)}\n")
            for parents, score in psets.items():
                outfile.write(f"{score:.4f} {len(parents)}")
                for parent in sorted(parents):
                    outfile.write(f" {parent}")
                outfile.write("\n")


def stream_bn(filename: str, normalize=True) -> BNStream:
    path, ext = os.path.splitext(filename)#将文件的格式分开 os.path.splitext("test/test.jkl")

    if ext == ".jkl":
        return stream_jkl(filename, normalize)
    else:
        print(f"unknown file format '{ext}'")


def read_bn(filename: str, normalize=True) -> BNData:
    return dict(stream_bn(filename, normalize))


def remove_zero_weight_parents(bndata: BNData, debug=False) -> BNData:
    """removes non-trivial psets whose score will get rounded down to zero"""
    newbndata: BNData = dict()
    for v, psets in bndata.items():
        newbndata[v] = {pset: score for pset, score in psets.items()
                              if int(score) > 0 or len(pset) == 0}
    if debug:
        oldpsets = sum(map(len, bndata.values()))
        newpsets = sum(map(len, newbndata.values()))
        print(f"removed {oldpsets-newpsets} zero weighted parents")
    return newbndata


def num_nodes_bn(filename: str) -> int:
    path, ext = os.path.splitext(filename)
    if ext == ".jkl":
        return num_nodes_jkl(filename)
    else:
        print(f"unknown file format '{ext}'")


def filter_stream_bn(filename: str, filterset, normalize=True) -> BNStream:
    for node, psets in stream_bn(filename, normalize):
        if node in filterset:
            yield node, psets

#jkl文件 选择的包中的变量
def filter_read_bn(filename: str, filterset, normalize=True) -> BNData:
    return dict(filter_stream_bn(filename, filterset, normalize))
#输出只包含选择的包中的变量的 节点，{父节点 分数（这个父节点的分数-属于节点的所有父节点的分数中最小的）}

def get_bn_stats(filename: str) -> Tuple[float, float, Dict[int, float]]:
    """
    returns sum of all scores and node-wise offsets used for normalizing
    返回用于标准化的所有分数和节点偏移的总和
    :param filename:
    :return:  (sum_score, Dict[node, min_score])
    """
    sum_score = 0
    best_score = 0
    offsets = dict()
    for node, psets in stream_bn(filename, normalize=False):
        scores = psets.values()
        sum_score += sum(scores)
        best_score += max(scores)
        offsets[node] = min(scores)
    return sum_score, best_score, offsets


def get_domain_sizes(filename: str) -> Dict[int, int]:
    with open(filename, 'r') as datfile: #只读模式打开文件，读文件内容的指针会放在文件的开头。
        _ = datfile.readline() # 读取第一行 # skip header line readline() 方法用于从文件读取整行，包括 “\n” 字符。运行一次读取一行 再次运行读取第二行
        domain_sizes = [int(ds) for ds in datfile.readline().split()] # 读取第二行 #split() 通过指定分隔符对字符串进行切片 默认为所有的空字符
        num_vars = len(domain_sizes)#zip() 函数是 Python 内置函数之一，它可以将多个序列（列表、元组、字典、集合、字符串以及 range() 区间构成的列表）
        # “压缩”成一个 zip 对象。
        # 所谓“压缩”，其实就是将这些序列中对应位置的元素重新组合，生成一个个新的元组。
        return dict(zip(range(num_vars), domain_sizes))


def get_vardata(filename: str) -> OrderedDict:
    with open(filename, 'r') as datfile:
        names = datfile.readline().strip().split()
        domain_sizes = [int(ds) for ds in datfile.readline().strip().split()]
        return OrderedDict(zip(names, domain_sizes))


# complexity width related function

def weight_from_domain_size(domain_size):
    # return ceil(log2(domain_size))
    # return ceil(2*log2(domain_size))
    return log2(domain_size)


def weights_from_domain_sizes(domain_sizes):
    return {node: weight_from_domain_size(size) for node, size in domain_sizes.items()}


def compute_complexity(bag: Union[Set, FrozenSet], domain_sizes: Dict[int, int],
                       approx=False) -> int:
    values = weights_from_domain_sizes(domain_sizes) if approx else domain_sizes
    if approx:
        reducer = lambda x, y: x+y
    else:
        reducer = lambda x, y: x*y
    return reduce(reducer, (values[var] for var in bag))


def compute_complexities(td: 'TreeDecomposition', domain_sizes: Dict[int, int],
                         approx=False) -> Dict[int, int]:
    values = weights_from_domain_sizes(domain_sizes) if approx else domain_sizes
    if approx:
        reducer = lambda x,y: x+y
    else:
        reducer = lambda x,y: x*y
    # reducer = int.__add__ if approx else int.__mul__
    complexities: Dict[int, int] = dict()
    for bag_idx, bag in td.bags.items():
        complexity = reduce(reducer, (values[var] for var in bag))#计算列表的乘积1*2*3*4*5
        complexities[bag_idx] = complexity
    return complexities


def compute_complexity_width(td: 'TreeDecomposition', domain_sizes: Dict[int, int],
                             approx=False, include=None) -> int:
    if include is not None:
        return max(val for bag_idx, val in compute_complexities(td, domain_sizes, approx).items()
                   if bag_idx in include)
    return max(compute_complexities(td, domain_sizes, approx).values())


def log_bag_metrics(td: 'TreeDecomposition', domain_sizes: Dict[int, int], append=False):
    if not domain_sizes: return  # don't run if domain_sizes not provided
    mode = 'a' if append else 'w'#mode=w
    with open("bag_metrics.txt", mode) as outfile:#打开文件 没有的话自己建一个文件 #r、w、a、x，分别代表读、写、追加、创建新文件。
        outfile.write(",".join(f"{len(bag)}" for bag in td.bags.values()))
        #连接任意数量的字符串（包括要连接的元素字符串、元组、列表、字典），用新的目标分隔符连接，返回新的字符串。
        outfile.write("\n")#换行 换行符
        outfile.write(",".join(map(str, compute_complexities(td, domain_sizes).values())))
        outfile.write("\n")


# constrained bnsl related functions

def read_constraints(conpath: str, typecast_node=None) -> Constraints:
    """
    read constraints from .con file (custom file format)

    :param conpath: path to .con file
    :param typecast_node: type to convert node names to (typically int)
    :return: dict with constraint types as keys and list of tuples of variables as values
    """
    constraints: Constraints = dict(posarc=[], negarc=[], undarc=[],
                                    posanc=[], neganc=[], undanc=[])
    with open(conpath) as confile:
        for line in confile:
            typ, u, v = line.split()
            if typecast_node is not None:
                constraints[typ].append((typecast_node(u), typecast_node(v)))
            else:
                constraints[typ].append((u, v))
    return constraints


def count_satisfied_constraints(bn, constraints):
    parents = {node: set(bn.dag.predecessors(node)) for node in bn.dag}
    ancestors = {node: set(nx.ancestors(bn.dag, node)) for node in bn.dag}

    base_checkers = {"arc": (lambda u, v: u in parents[v]),
                     "anc": (lambda u, v: u in ancestors[v])}

    counts = dict.fromkeys(constraints, 0)
    for typ, cons in constraints.items():
        if not cons: continue
        subtyp, basetyp = typ[:3], typ[3:]
        if subtyp == "neg":
            checker = lambda u, v: not base_checkers[basetyp](u, v)
        elif subtyp == "und":
            checker = lambda u, v: (base_checkers[basetyp](u, v) or base_checkers[basetyp](v, u))
        else:
            checker = base_checkers[basetyp]
        cur_satisfied = sum(map(checker, *zip(*cons)))
        # print(f"{typ} satisfied: {cur_satisfied}/{len(cons)}")
        counts[typ] = cur_satisfied
    return counts


def count_constraints(constraints: Constraints):
    return sum(map(len, constraints.values()))


def total_satisfied_constraints(bn, constraints):
    splits = count_satisfied_constraints(bn, constraints)
    return sum(splits.values())


def filter_satisfied_constraints(bn, constraints, debug=False) -> Constraints:
    satisfied = {typ: [] for typ in constraints}
    parents = {node: set(bn.dag.predecessors(node)) for node in bn.dag}
    ancestors = {node: set(nx.ancestors(bn.dag, node)) for node in bn.dag}

    base_checkers = {"arc": (lambda u, v: u in parents[v]),
                     "anc": (lambda u, v: u in ancestors[v])}

    counts = dict.fromkeys(constraints, 0)
    for typ, cons in constraints.items():
        satisfied[typ] = []
        if not cons: continue
        subtyp, basetyp = typ[:3], typ[3:]
        if subtyp == "neg":
            checker = lambda u, v: not base_checkers[basetyp](u, v)
        elif subtyp == "und":
            checker = lambda u, v: (base_checkers[basetyp](u, v) or base_checkers[basetyp](v, u))
        else:
            checker = base_checkers[basetyp]
        for u, v in cons:
            if checker(u, v): satisfied[typ].append((u, v))
    if debug:
        old_total = count_constraints(constraints)
        new_total = count_constraints(satisfied)
        print(f"filtered constraints {old_total} -> {new_total}")
    return satisfied


def spot_failing_constraint(bn, constraints):
    parents = {node: set(bn.dag.predecessors(node)) for node in bn.dag}
    ancestors = {node: set(nx.ancestors(bn.dag, node)) for node in bn.dag}

    base_checkers = {"arc": (lambda u, v: u in parents[v]),
                     "anc": (lambda u, v: u in ancestors[v])}

    for typ, cons in constraints.items():
        if not cons: continue
        subtyp, basetyp = typ[:3], typ[3:]
        if subtyp == "neg":
            checker = lambda u, v: not base_checkers[basetyp](u, v)
        elif subtyp == "und":
            checker = lambda u, v: (base_checkers[basetyp](u, v) or base_checkers[basetyp](v, u))
        else:
            checker = base_checkers[basetyp]
        for u, v in cons:
            if not checker(u, v): 
                return typ, u, v
    return None


class NoSolutionError(BaseException): pass
class UnsatisfiableInstanceError(BaseException): pass


def read_model(output: Union[str, TextIO]) -> set:
    status = ""
    model = None
    #print('isinstance',isinstance(output, str))
    if isinstance(output, str):# isinstance()函数是python中的一个内置函数，作用：判断一个函数是否是一个已知类型，类似type()。
        output = output.split("\n")
    for line in output:
        if line.startswith("v"):
            return set(map(int, line.split()[1:]))
        if "UNSATISFIABLE" in line:  # only if model not found
            raise UnsatisfiableInstanceError
    # if model found, this line should not be reached
    if isinstance(output, TextIO): output.seek(0)
    with open("err-output.log", 'w') as err_out:
        for line in output:
            err_out.write(line)
    raise NoSolutionError("model not found (no line starting with 'v'\n\t"
                          "output written to err-output.log")


def read_model_from_file(filename: str) -> set:
    with open(filename) as out:
        return read_model(out)


def is_optimal(output: Union[str, TextIO]) -> bool:
    if isinstance(output, str):
        output = output.split("\n")
    for line in output:
        if line.strip() == "s OPTIMUM FOUND":
            return True
    return False


# treewidth related functions

def filled_in(graph, order) -> Tuple[nx.Graph, int]:
    fgraph = graph.copy()
    cur_nodes = set(graph.nodes)
    max_degree = -1
    for u in order:
        trunc_graph = fgraph.subgraph(cur_nodes)
        try:
            max_degree = max(max_degree, trunc_graph.degree(u))
        except nx.NetworkXError as err:  #debug
            print("### networkx error while processing node", u)
            raise err
        neighbors = trunc_graph.neighbors(u)
        fgraph.add_edges_from(itertools.combinations(neighbors, 2))
        cur_nodes.remove(u)
    return fgraph, max_degree


def find_first_by_order(elements: set, order):
    for element in order:
        if element in elements: return element


def check_subgraph(graph: nx.Graph, subgraph: nx.Graph):
    for edge in subgraph.edges:
        if not graph.has_edge(*edge):
            return False
    return True


def topsort(graph: nx.DiGraph, seed=None):
    #todo: complete
    rng = random.Random(seed)
    graph = graph.copy()
    sources = [v for v in graph if graph.in_degree(v) == 0]
    while graph:
        index = rng.randint(0, len(sources)-1)
        chosen = sources[index]
        yield chosen
        del sources[index]
        for nbr in graph.successors(chosen):
            if graph.in_degree(nbr) == 1:
                sources.append(nbr)
        graph.remove_node(chosen)


class TreeDecomposition(object):
    def __init__(self, graph: nx.Graph, order, width=0):#道德图 消去序 树宽
        self.bags: Dict[int, frozenset] = dict()
        self.decomp = nx.Graph()
        self.graph = graph
        self.elim_order = order
        self.width = width
        self._bag_ctr = 0
        if order is not None:
            self.decomp_from_ordering(graph, order, width)

    @staticmethod
    def from_td(tdstr: str):
        # parse tree decomposition
        tdlines = tdstr.split("\n")
        header = tdlines.pop(0)
        ltype, _, nbags, maxbagsize, nverts = header.split()
        assert ltype == "s", "invalid header ({ltype}) in tree decomposition"

        self = TreeDecomposition(None, None)
        self.width = int(maxbagsize) - 1
        self._bag_ctr = int(nbags) + 1

        for line in tdlines:
            if not line: continue
            ltype, rest = line.split(maxsplit=1)
            if ltype == "c":  # comment line, ignore
                continue
            elif ltype == "b":
                bag_idx, rest = rest.split(maxsplit=1)
                bag_idx = int(bag_idx)-1
                bag = frozenset(map(int, rest.split()))
                self.bags[bag_idx] = bag
                self.decomp.add_node(bag_idx)
            else:  # edge of tree decomp
                u, v = int(ltype), int(rest)
                self.decomp.add_edge(u-1, v-1)

        self.elim_order = self.recompute_elim_order()
        return self
        
    def add_bag(self, nodes: Union[set, frozenset], parent: int = -1):
        nodes = frozenset(nodes)
        bag_idx = self._bag_ctr
        self._bag_ctr += 1
        self.bags[bag_idx] = nodes
        self.decomp.add_node(bag_idx)
        if parent >= 0:
            self.decomp.add_edge(parent, bag_idx)
        return bag_idx

    def decomp_from_ordering(self, graph, order, width):
        graph, max_degree = filled_in(graph, order)
        if width > 0:
            assert max_degree <= width, \
                f"Treewidth({width}) exceeded by ordering({max_degree}): {order}"
            self.width = width
        else:
            self.width = max_degree
        revorder = order[::-1]
        # try:
        cur_nodes = {revorder[0]}
        # except IndexError:
        #     print("index error", order, revorder)
        #     return
        root_bag = self.add_bag(cur_nodes)
        blame = {node: root_bag for node in cur_nodes}
        for u in revorder[1:]:
            cur_nodes.add(u)
            neighbors = set(graph.subgraph(cur_nodes).neighbors(u))
            if neighbors:
                first_neighbor = find_first_by_order(neighbors, order)
                parent = blame[first_neighbor]
            else:
                parent = root_bag
            bag_idx = self.add_bag(neighbors | {u}, parent)
            blame[u] = bag_idx
        if __debug__: self.verify()

    def verify(self, graph: nx.Graph=None):
        if graph is None: graph = self.graph
        # check if tree
        assert nx.is_tree(self.decomp), "decomp is not a tree"
        # check width
        max_bag_size = max(map(len, self.bags.values()))
        assert max_bag_size <= self.width + 1, \
            f"decomp width too high ({max_bag_size} > {self.width + 1})"
        # check vertex connected subtree
        for node in graph.nodes:
            bags_containing = [bag_id for bag_id in self.bags
                               if node in self.bags[bag_id]]
            assert nx.is_connected(self.decomp.subgraph(bags_containing)), \
                f"subtree for vertex {node} is not connected"
        # check if every edge covered
        for edge in graph.edges:
            for bag in self.bags.values():#issuperset() 方法用于判断指定集合的所有元素是否都包含在原始的集合中，如果是则返回 True，否则返回 False。
                if bag.issuperset(edge):
                    break
            else:
                raise AssertionError(f"edge {edge} not covered by decomp")
            continue
        new_elim_order = self.recompute_elim_order()
        fgraph, max_degree = filled_in(graph, new_elim_order)
        assert max_degree <= self.width, f"newly computed elim order invalid" \
                                         f"{max_degree} > {self.width}"

    def get_boundary_intersections(self, selected) -> Dict[int, Dict[int, frozenset]]:#输入选择的包的id
        intersections = {bag_id: dict() for bag_id in selected}#定义一个字典
        for bag_id, nbr_id in nx.edge_boundary(self.decomp, selected):#输出选择的包的邻居 输出包的id 及其邻居节点的id
#edge_boundary找的不在selected中与selected相邻的节点
            #print(bag_id,nbr_id)
            assert bag_id in selected, "edge boundary pattern assumption failed"#边缘边界模式假设失败
            # Python assert(断言)用于判断一个表达式,在表达式条件为 false 的时候触发异常
            intersections[bag_id][nbr_id] = self.bags[bag_id] & self.bags[nbr_id] #交集
        return intersections #输出选择的包和其邻居节点包中的变量的交集

    def draw(self, subset=None):
        if subset is None:
            decomp = self.decomp
        else:
            decomp = self.decomp.subgraph(subset)
        pos = graphviz_layout(decomp, prog='dot')
        labels = {bag_idx: f"{bag_idx}{list(bag)}"
                  for (bag_idx, bag) in self.bags.items() if bag_idx in pos}
        nx.draw(decomp, pos)
        nx.draw_networkx_labels(decomp, pos, labels=labels)
        plt.show()

    def replace(self, selected, forced_cliques, new_td: 'TreeDecomposition'):
        remap = dict()
        # add new bags
        for old_id, bag in new_td.bags.items():
            new_id = self.add_bag(bag)
            remap[old_id] = new_id
        # add new edges
        for b1, b2 in new_td.decomp.edges:
            self.decomp.add_edge(remap[b1], remap[b2])

        # connect new bags to those outside selected
        # todo[opt]: smart patching (avoid brute force, use elim ordering td)
        for nbr_id, intersection in forced_cliques.items():
            # find bag in new_td which contains intersection
            req_bag_id = new_td.bag_containing(intersection)
            assert req_bag_id != -1,\
                f"required bag containing {set(intersection)} not found"
            self.decomp.add_edge(remap[req_bag_id], nbr_id)

        # noinspection PyUnreachableCode
        if __debug__:
            covered_nodes = set()
            for bag in new_td.bags.values():
                covered_nodes.update(bag)
            existing_nodes = set()
            for sel_idx in selected:
                existing_nodes.update(self.bags[sel_idx])
            assert covered_nodes == existing_nodes, \
                f"replacement td mismatch, " \
                f"existing: {existing_nodes}\tcovered: {covered_nodes}"

        # delete old bags which have been replaced
        for sel_idx in selected:
            del self.bags[sel_idx]
            self.decomp.remove_node(sel_idx)

    def bag_containing(self, members: Union[set, frozenset],
                       exclude: Set[int] = None) -> int:
        """
        returns the id of a bag containing given members
        if no such bag exists, returns -1
        """
        exclude = set() if exclude is None else exclude
        for bag_id, bag in self.bags.items():
            if bag_id in exclude: continue
            if bag.issuperset(members):
                return bag_id
        return -1

    def recompute_elim_order(self) -> list:
        """
        recomputes elimination ordering based on possibly modified
        decomposition bags

        :return: new elim_order as list
        """
        rootbag = first(self.bags)  # arbitrarily choose a root bag
        elim_order = list(self.bags[rootbag])  # initialize eo with rootbag
        for u, v in nx.dfs_edges(self.decomp, source=rootbag):
            forgotten = self.bags[v] - self.bags[u]
            elim_order.extend(forgotten)
        elim_order.reverse()
        return elim_order

    def compute_width(self) -> int:
        """compute treewidth"""
        return max(map(len, self.bags.values())) - 1


class CWDecomposition(TreeDecomposition):

    def __init__(self, graph: nx.Graph, order, width, domain_sizes):
        # initialize common stuff, exclude order to skip td construction
        super().__init__(graph, None)
        self.elim_order = order
        self.width = width
        self.domain_sizes = domain_sizes
        self.rootbag_size = -1
        self.decomp_from_ordering(graph, order, width, domain_sizes)

    def decomp_from_ordering(self, graph, order, width, domain_sizes):
        graph, max_degree = filled_in(graph, order)
        self.width = width
        revorder = order[::-1]
        rootbag_complexity, rootbag_size = 1, 0
        for node in revorder:
            rootbag_complexity *= domain_sizes[node]
            if rootbag_complexity > width: break
            rootbag_size += 1
        self.rootbag_size = rootbag_size
        # try:
        cur_nodes = {revorder[0]}
        # except IndexError:
        #     print("index error", order, revorder)
        #     return
        cur_nodes = set(revorder[:rootbag_size])
        root_bag = self.add_bag(cur_nodes)
        blame = {node: root_bag for node in cur_nodes}
        for u in revorder[rootbag_size:]:
            cur_nodes.add(u)
            neighbors = set(graph.subgraph(cur_nodes).neighbors(u))
            if neighbors:
                first_neighbor = find_first_by_order(neighbors, order)
                parent = blame[first_neighbor]
            else:
                parent = root_bag
            bag_idx = self.add_bag(neighbors | {u}, parent)
            blame[u] = bag_idx
        #if __debug__: self.verify()

    def verify(self):
        raise NotImplementedError
    
if __name__ == '__main__':
    from blip import parse_res
    #bn = parse_res("../input/alarm-5000.jkl", 4, "../past-work/blip-publish/tmp.res")
    #bn = parse_res("../input/sachs-5000.jkl", 4, "../past-work/blip-publish/tmp.res")
    bn = parse_res("../input/hepar2-5000.jkl", 7, "../past-work/blip-publish/temp.res")
    constraints = read_constraints("../input/constraint-files/all/hepar2-50-1.con", int)
    #constraints = read_constraints("../input/constraint-files/sachs-20-4.con", int)
    print("satisfied splits:", count_satisfied_constraints(bn, constraints))
    print("satisfied total:", total_satisfied_constraints(bn, constraints))
    total_constraints = count_constraints(constraints)
    print("total constraints:", total_constraints)
    print({t: len(v) for t,v in constraints.items()})

    # g = nx.bull_graph()
    # g.add_edges_from([(4, 5), (2, 5)])
    # ordering = list(range(len(g)))
    # fgraph = filled_in(g, ordering)
    # td = TreeDecomposition(g, ordering, 3)
    # td.draw()
