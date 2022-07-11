from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .value import Value
    from .comp_node import CompNode

from collections.abc import Iterable
from typing import Set, Dict, Union

from mizore.utils.typed_log import TypedLog




class CompGraph:
    def __init__(self, output_elems):
        self._output_vals = []
        # Construct _output_vals according to the input types
        self._add_output_vals(output_elems)
        self.values: Set[Value] = set()
        self.nodes = []
        self.nodes_set = set()
        self.out_graph_nodes = set()
        for val in self._output_vals:
            # Add all connected values/nodes in the graph
            self.add_child_elems(val)
        self.node_in: Dict = {node: set() for node in self.nodes}
        self.node_out: Dict = {node: set() for node in self.nodes}
        self.build_node_graph()
        self._layers = []
        self.build_node_layers()
        self.log = TypedLog()

    def add_output_vals(self, output_vals, reconstruct=True):
        self._add_output_vals(output_vals)
        if reconstruct:
            self.reconstruct()

    def _add_output_vals(self, output_vals):
        if isinstance(output_vals, Iterable):
            for elem in output_vals:
                self._output_vals.append(elem)
        else:
            self._output_vals.append(output_vals)

    def reconstruct(self):
        # Reconstruct the graph
        self.values = set()
        self.nodes = []
        self.nodes_set = set()
        self.out_graph_nodes = set()
        for elem in self._output_vals:
            # Add all connected values/nodes in the graph
            self.add_child_elems(elem)
        self.node_in: Dict = {node: set() for node in self.nodes}
        self.node_out: Dict = {node: set() for node in self.nodes}
        self.build_node_graph()
        self._layers = []
        self.build_node_layers()

    def del_all_cache(self):
        val: Value
        for val in self.values:
            val.del_cache()

    def copy_graph(self):
        """
        To use this function, you must make sure all the nodes only contain references to other Value
        in its inputs and outputs list
        :return:
        """
        # TODO test this
        new_elem_dict = {}
        new_output_elems = []
        for val in self._output_vals:
            new_val = val.copy_with_map_dict(new_elem_dict)
            new_output_elems.append(new_val)
        return CompGraph(new_output_elems)

    def build_node_layers(self):
        temp_node_in = {}
        for node, in_nodes in self.node_in.items():
            temp_node_in[node] = in_nodes.copy()
        no_in_nodes = CompGraph.get_no_in_nodes(temp_node_in)
        while len(no_in_nodes) != 0:
            self._layers.append(no_in_nodes)
            CompGraph.delete_nodes(no_in_nodes, temp_node_in)
            no_in_nodes = CompGraph.get_no_in_nodes(temp_node_in)

    @classmethod
    def delete_nodes(cls, nodes: Set, graph_dict: Dict[CompNode, Set]):
        """
        :param nodes: The nodes to delete from graph_dict
        :param graph_dict: In nodes dict or Out nodes dict
        """
        for node_to_del in nodes:
            del graph_dict[node_to_del]
            for nodes in graph_dict.values():
                nodes.discard(node_to_del)

    @classmethod
    def get_no_in_nodes(cls, node_in: Dict[CompNode, Set]) -> Set:
        """
        Get the nodes that have no in-node
        :param node_in:
        :return:
        """
        no_in_nodes = set()
        for node, in_nodes in node_in.items():
            if len(in_nodes) == 0:
                no_in_nodes.add(node)
        return no_in_nodes

    def build_node_graph(self):
        for node in self.nodes:
            direct_child = set()
            for val in node.inputs.values():
                CompGraph.get_direct_node_child(val, direct_child)
            self.node_in[node] = direct_child
            for child in direct_child:
                self.node_out[child].add(node)

    @classmethod
    def get_direct_node_child(cls, value: Value, child_set: Set):
        if value.home_node is not None and value.home_node.in_graph:
            child_set.add(value.home_node)
        for val in value.args:
            CompGraph.get_direct_node_child(val, child_set)

    def add_child_elems(self, root: Value) -> None:
        """
        Add all elements (CompNode and Value) in self.elems
        Add all CompNode in self.nodes
        :param root: the root element where the search is from
        """
        self.values.add(root)
        if root.home_node is not None:
            if root.home_node not in self.values:
                if root.home_node.in_graph:
                    self.add_child_elems_for_node(root.home_node)
                else:
                    self.out_graph_nodes.add(root.home_node)
        for val in root.args:
            if val not in self.values:
                self.add_child_elems(val)
        if root.is_indep_random and root.var_constructed is not None:
            self.add_child_elems(root.var_constructed)

    def add_child_elems_for_node(self, root: CompNode):
        assert root.in_graph  # The initial elem must be in_graph
        if root not in self.nodes:
            self.nodes_set.add(root)
            self.nodes.append(root)
        for val in root.inputs.values():
            if val not in self.values:
                self.values.add(val)
                self.add_child_elems(val)

    def iter_nodes_by_type(self, typename) -> Iterable[CompNode]:
        def iterable():
            for layer in self._layers:
                for node in layer:
                    if isinstance(node, typename):
                        yield node

        return GraphIterator(iterable(), self)

    def iter_nodes_by_tag(self, tag: Union[Iterable, any]):
        def iterable():
            _tags: Set
            if isinstance(tag, Iterable):
                _tags = set(tag)
            else:
                _tags = {tag}
            for layer in self._layers:
                node: CompNode
                for node in layer:
                    if not node.tags.isdisjoint(_tags):
                        yield node

        return GraphIterator(iterable(), self)

    def iter_nodes_by_prefix(self, prefix):
        def iterable():
            for layer in self._layers:
                node: CompNode
                for node in layer:
                    if node.name is not None and node.name.startswith(prefix):
                        yield node

        return GraphIterator(iterable(), self)

    def __str__(self):
        res = ["===CompGraph===\n"]
        res.append(f"Number of nodes: {len(self.nodes)}\n")
        for i in range(len(self._layers)):
            res.append(f"Layer {i}: ({len(self._layers[i])} nodes)\n")
            for node in self._layers[i]:
                res.append(str(node) + "\n")
            res.append("\n")
        res.append("===============\n")
        return "".join(res)

    def last_layer(self):
        return GraphIterator(self._layers[-1], self)

    def layers(self):
        for layer in self._layers:
            yield GraphIterator(layer, self)

    def all(self):
        def iterable():
            for layer in self._layers:
                for node in layer:
                    yield node

        return GraphIterator(iterable(), self)

    def __iter__(self):
        for node in self.all():
            yield node

    def reconstruct_graph(self):
        self.reconstruct()

    def by_type(self, _type):
        for layer in self._layers:
            for node in layer:
                if isinstance(node, _type):
                    yield node

    @property
    def comp_graph(self):
        return self


class GraphIterator:
    def __init__(self, iterable, comp_graph: CompGraph):
        self._comp_graph: CompGraph = comp_graph
        self.iterable = iterable

    @property
    def comp_graph(self):
        return self._comp_graph

    def reconstruct_graph(self):
        self._comp_graph.reconstruct()

    def by_type(self, _type):
        for node in self.iterable:
            if isinstance(node, _type):
                yield node

    def __iter__(self):
        for node in self.iterable:
            yield node


def graph_dict_str(graph_dict):
    res = []
    for node, in_or_out_nodes in graph_dict.items():
        res.append(node.name + ":\n")
        if len(in_or_out_nodes) == 0:
            res.append("    Independent\n")
            continue
        for in_node in in_or_out_nodes:
            res.append("    " + in_node.name + "\n")

    return "".join(res)
