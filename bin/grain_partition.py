"""Say you have a mesh with labels on elements and connectivity like so:

    N1       N2        N3      N4
    +---------+--------+-------+
    | B (1)   | B (2)  | A (3) |
 N5 +---------+--------+-------+ N8
    | B (4)   | B (5)  | A (6) |
 N9 +---------+--------+-------+ N12
    | A (7)   | A (8)  | C (9) |
N13 +---------+--------+-------+ N16

ABC are the labels, 1-9 are the element numbers.

We want something like this out:
{
    A: (3,6),
    A: (7,8),
    B: (1,2,4,5),
    C: (9,)
}

So groups of connected elements, attached with edges in common.

Assumes mesh is a surface (no T-juctions or anything like that).
"""

import collections
import functools
import itertools
import math
import typing

import multidict

_DEBUG_OUT = False


class EdgeKey(typing.NamedTuple):
    """Have the nodes in sorted order so we don't have to worry about going back or forward..."""

    n_low: int
    n_high: int


class LabeledElement(typing.NamedTuple):
    label: str
    num: int
    connection: typing.Tuple[int, ...]

    def generate_edges(self) -> typing.Iterable[EdgeKey]:
        conn_next = self.connection[1:] + self.connection[0:1]
        for e1, e2 in zip(self.connection, conn_next):
            n_low, n_high = sorted([e1, e2])
            yield EdgeKey(n_low=n_low, n_high=n_high)


class Cluster(typing.NamedTuple):
    """Note that this is immuateble but you can still add/remove from the mutable attributes."""

    label: str
    elements: set
    cluster_edges: typing.Dict[EdgeKey, LabeledElement]

    def __str__(self):
        e_nums = sorted(e.num for e in self.elements)
        free_edges = sorted({(e.n_low, e.n_high) for e in self.cluster_edges.keys()})
        return f"{self.label}: {e_nums} / {free_edges}"

    def _toggle_edge(self, edge: EdgeKey, elem: LabeledElement):
        """All edges should come in pairs which cancel out. So if there's
        one edge, that's a free edge. If there are two, that's no longer free."""

        if edge in self.cluster_edges:
            del self.cluster_edges[edge]

        else:
            self.cluster_edges[edge] = elem

    def toggle_from_element(self, elem: LabeledElement):
        for edge in elem.generate_edges():
            self._toggle_edge(edge, elem)

    def toggle_from_cluster(self, cluster: "Cluster"):
        for edge, elem in cluster.cluster_edges.items():
            self._toggle_edge(edge, elem)


class ClusterSet:
    free_edges: typing.Dict[EdgeKey, Cluster]
    clusters: multidict.MultiDict  # Label -> Cluster. There can be duplicated labels.

    def __init__(self):
        self.free_edges = dict()
        self.clusters = multidict.MultiDict()

    def add_all_elems(self, elems: typing.Iterable[LabeledElement]):
        for elem in elems:
            self.add_element(elem)

    def add_element(self, elem: LabeledElement):

        if _DEBUG_OUT:
            print()
            print(self.in_canonial_form())
            print(f"Adding {elem}")

        # Find the cluster(s) this is a member of.
        existing_clusters = list()
        newly_exposed_edges = list()
        for new_edge in elem.generate_edges():
            # Use "pop" to remove it from the free edges set - they're not free anymore!
            maybe_existing_cluster = self.free_edges.pop(new_edge, None)

            # Each edge interface either increases or decreases the exposed edges by one.
            if maybe_existing_cluster:
                # Can have multiple edges pointing to the same existing cluster.
                if maybe_existing_cluster not in existing_clusters:
                    existing_clusters.append(maybe_existing_cluster)

            else:
                newly_exposed_edges.append(new_edge)

        same_label_cluster = [c for c in existing_clusters if c.label == elem.label]
        if not same_label_cluster:
            active_cluster = self._make_new_cluster_for(elem)

        else:
            active_cluster = self._add_to_existing_cluster_and_merge(
                elem, same_label_cluster
            )

        # Add any new edges from this element which didn't connect somewhere else.
        for new_exposed_edge in newly_exposed_edges:
            self.free_edges[new_exposed_edge] = active_cluster

        # Local cluster free edges
        active_cluster.toggle_from_element(elem)

    def _make_new_cluster_for(self, elem: LabeledElement):
        new_cluster = Cluster(
            label=elem.label,
            elements={elem},
            cluster_edges=dict(),
        )

        self.clusters.add(new_cluster.label, new_cluster)

        return new_cluster

    def _add_to_existing_cluster_and_merge(
        self, elem: LabeledElement, same_label_cluster: typing.List[Cluster]
    ):
        # Any clusters which have this element AND match the label are to be merged.
        existing_cluster_to_keep = same_label_cluster[0]
        existing_cluster_to_keep.elements.add(elem)

        # There may be some clusters which are now considered to be the same.
        to_remove = same_label_cluster[1:]
        if to_remove:
            self._merge_cluster_into(existing_cluster_to_keep, to_remove)

        return existing_cluster_to_keep

    def _merge_cluster_into(self, to_keep: Cluster, to_remove: typing.List[Cluster]):

        keys = {c.label for c in to_remove}
        if len(keys) != 1:
            raise ValueError(f"Expected on key, got {keys}")

        all_matching_cluster = self.clusters.popall(keys.pop())

        # Sanity checks!
        if to_keep not in all_matching_cluster:
            raise ValueError(f"Shouldn't we have to_keep {to_keep} in here?")

        for to_go in to_remove:
            if to_go not in all_matching_cluster:
                raise ValueError(f"Shouldn't we have to_go {to_go} in here?")

        # Migate the elements across
        for to_go in to_remove:
            to_keep.elements.update(to_go.elements)

        # Just retain the "to keep" one
        self.clusters[to_keep.label] = to_keep

        # migrate free edges to the "keep" one.
        to_apply_edges = set()
        for edge, cluster in self.free_edges.items():
            if cluster in to_remove:
                to_apply_edges.add(edge)

        for edge in to_apply_edges:
            self.free_edges[edge] = to_keep

        # Local "cluster only" version of the free edges stuff
        for to_go in to_remove:
            to_keep.toggle_from_cluster(to_go)

    def in_canonial_form(self) -> str:
        def edge_list(el):
            edges = sorted(el)
            bits = [f"{n_low}-{n_high}" for n_low, n_high in edges]
            return ", ".join(bits)

        # Free edges
        free_edges = collections.defaultdict(list)
        for free_edge, cluster in self.free_edges.items():

            free_edges[str(cluster)].append((free_edge.n_low, free_edge.n_high))

        lines = []
        keys = sorted(set(self.clusters.keys()))
        for key in keys:
            # lines.append(f"  {key}:")

            # Sort the clusters by lowest element first
            elem_edge_lists = []
            for cluster in self.clusters.getall(key):
                elems = sorted(e.num for e in cluster.elements)
                global_free_edges = free_edges.get(str(cluster), [])
                cluster_free_edges = set(
                    (fe.n_low, fe.n_high) for fe in cluster.cluster_edges.keys()
                )
                elem_edge_lists.append((elems, global_free_edges, cluster_free_edges))

            elem_edge_lists.sort()
            for elem_list, global_free_edges, cluster_free_edges in elem_edge_lists:
                # Global free edges are exposed on the cluster
                cluster_only = {
                    fe for fe in cluster_free_edges if fe not in global_free_edges
                }

                lines.append(
                    f"  {key}  {elem_list}\tLocal: {edge_list(cluster_only)}\tGlobal: {edge_list(global_free_edges)}"
                )

        for clust in sorted(free_edges.keys()):
            edge_bits = [(n1, n2) for (n1, n2) in free_edges[clust]]
            edge_bits.sort()

            # lines.append(f" {clust} <- {edge_bits}")

        return "\n".join(lines)

    ### Testing stuff only

    """
        N1       N2        N3      N4
        +---------+--------+-------+
        | B (1)   | B (2)  | A (3) |
     N5 +---------+--------+-------+ N8
        | B (4)   | B (5)  | A (6) |
     N9 +---------+--------+-------+ N12
        | A (7)   | A (8)  | C (9) |
    N13 +---------+--------+-------+ N16
    """


_test_elems = [
    LabeledElement("B", 1, (1, 2, 6, 5)),
    LabeledElement("B", 2, (2, 3, 7, 6)),
    LabeledElement("A", 3, (3, 4, 8, 7)),
    LabeledElement("B", 4, (5, 6, 10, 9)),
    LabeledElement("B", 5, (6, 7, 11, 10)),
    LabeledElement("A", 6, (7, 8, 12, 11)),
    LabeledElement("A", 7, (9, 10, 14, 13)),
    LabeledElement("A", 8, (10, 11, 15, 14)),
    LabeledElement("C", 9, (11, 12, 16, 15)),
]


@functools.lru_cache
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)


def nth_permutation(idx, length, alphabet=None, prefix=()):
    if alphabet is None:
        alphabet = [i for i in range(length)]
    if length == 0:
        return prefix
    else:
        branch_count = factorial(length - 1)
        for d in alphabet:
            if d not in prefix:
                if branch_count <= idx:
                    idx -= branch_count
                else:
                    return nth_permutation(idx, length - 1, alphabet, prefix + (d,))


def _get_nth_permutation(idx):
    total = math.perm(len(_test_elems), len(_test_elems))
    if idx >= total:
        raise ValueError(idx)

    perm_idxs = nth_permutation(idx, len(_test_elems))

    elems = [_test_elems[i] for i in perm_idxs]
    return elems


def test_clustering():

    cluster_set = ClusterSet()

    for elem in _test_elems:
        cluster_set.add_element(elem)

    print()
    print()
    print(cluster_set.in_canonial_form())


def test_all_orderings():
    """This should be independent of the order of the inputs!"""

    cluster_set = ClusterSet()
    cluster_set.add_all_elems(_test_elems)
    ref = cluster_set.in_canonial_form()

    total = math.perm(len(_test_elems), len(_test_elems))
    for idx, one_ordering in enumerate(itertools.permutations(_test_elems)):
        if idx % 10000 == 0:
            print(f"{idx} / {total}\t{idx/total:%}")

        try:
            cs = ClusterSet()
            cs.add_all_elems(one_ordering)
            this = cs.in_canonial_form()
            if ref != this:
                print(ref)
                print(this)
                print("idx", idx)
                raise ValueError(this)

        except ValueError as e:
            print(e)
            print(idx)
            raise e


def test_specific():
    n = 6480
    elems = _get_nth_permutation(6480)
    cs = ClusterSet()
    cs.add_all_elems(elems)


if __name__ == "__main__":

    test_clustering()

    test_specific()

    test_all_orderings()
