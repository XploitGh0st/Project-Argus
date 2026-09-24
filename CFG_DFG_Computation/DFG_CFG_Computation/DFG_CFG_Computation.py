from __future__ import absolute_import
from __future__ import print_function

from collections import defaultdict
from glob import glob
from json import dumps
import json
import os
import sys
from optparse import OptionParser
from typing import Tuple

import networkx as nx
import pydot
import pyverilog
import pyverilog.utils.util as util
import pyverilog.vparser.ast as vast
import torch
import torch.nn.functional as F
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.controlflow.controlflow_analyzer import VerilogControlflowAnalyzer as PyControlflowAnalyzer
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer as PyDataflowAnalyzer
from pyverilog.dataflow.graphgen import VerilogGraphGenerator as PyGraphGenerator
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer as PyDataflowOptimizer
from pyverilog.vparser.parser import parse
from sklearn.model_selection import train_test_split

from hw2vec.graph2vec.trainers import *  # noqa: F401

# Module path setup
sys.path.append(os.path.dirname(sys.path[0]))


class JsonGraphParser:
    """Parses hardware graph data from JSON files for use in GNN training and evaluation.

    Manages collections of hardware design graphs (as NetworkX DiGraphs), supports
    train/test splitting, and provides graph-pair utilities for the GNN4IP pipeline.
    """

    def __init__(self, cfg):
        """Initialises the parser with the given configuration object.

        Args:
            cfg: Configuration object containing dataset path, split ratio, random seed,
                 debug flag, and pickle path attributes.
        """
        self.cfg = cfg
        self.root_path = self.cfg.raw_dataset_path.resolve()

        # self.graphs is a dict that stores all the hardware graphs that can be parsed from the raw datatset folder.
        # 'all': means it is a collection of all graph datasets.
        # 'train' and 'test': store the collection of graph instances according to a prespliting result.
        self.graphs = {}

        self.node_labels = set()

        # for GNN4IP
        # self.graph_pairs/_train/_test stores the graph-pairing info. e.g. (0, 1, -1) means graph 0 and graph 1 is dissimilar.
        self.graph_pairs = []
        self.graph_pairs_train = []
        self.graph_pairs_test = []

        # self.trunk stores the hardware category information about each hardware design. e.g. in [0, 1, 1, 2, 3], hw #1 belongs to category 1.
        self.trunk = []

        # list to store names of graphs in each set
        self.training_graph_count = 0
        self.testing_graph_count = 0

        self.do_normalize = True

    def append_training_graph_data(self, data):
        """Appends a graph data object to the training split collection.

        Args:
            data: A graph data object (e.g. PyG Data) to add to the training set.
        """
        if 'train' not in self.graphs:
            self.graphs['train'] = []
        self.graphs['train'].append(data)

    def append_testing_graph_data(self, data):
        """Appends a graph data object to the testing split collection.

        Args:
            data: A graph data object (e.g. PyG Data) to add to the test set.
        """
        if 'test' not in self.graphs:
            self.graphs['test'] = []
        self.graphs['test'].append(data)

    def append_graph_data(self, data):
        """Appends a graph data object to the combined (all) collection.

        Args:
            data: A graph data object to add to the full dataset.
        """
        if 'all' not in self.graphs:
            self.graphs['all'] = []
        self.graphs['all'].append(data)

    def append_graph_pair(self, pair):
        """Appends a graph pair tuple to the graph-pair list used by GNN4IP.

        Args:
            pair (tuple): A 3-tuple of the form (graph_id_a, graph_id_b, label)
                          where label is 1 (similar) or -1 (dissimilar).
        """
        self.graph_pairs.append(pair)

    def get_graphs(self):
        """Returns the train and test graph collections.

        If pre-split 'train'/'test' keys exist in self.graphs they are returned directly.
        Otherwise the 'all' collection is split according to cfg.ratio and cfg.seed.

        Returns:
            tuple: (train_graphs, test_graphs) — two lists of graph data objects.
        """
        if 'train' in self.graphs and 'test' in self.graphs:
            return self.graphs['train'], self.graphs['test']
        elif 'all' in self.graphs:
            return self.split_dataset(ratio=self.cfg.ratio, seed=self.cfg.seed, dataset=self.graphs['all'])

    def get_pairs(self):
        """Splits the graph-pair list into training and testing subsets.

        In debug mode all pairs are used; otherwise the first 100 pairs are used.

        Returns:
            tuple: (graph_pairs_train, graph_pairs_test) — two lists of pair tuples.
        """
        graph_pairs = self.graph_pairs if self.cfg.debug else self.graph_pairs[:100]
        self.graph_pairs_train, self.graph_pairs_test = self.split_dataset(
            ratio=self.cfg.ratio, seed=self.cfg.seed, dataset=graph_pairs
        )

        return self.graph_pairs_train, self.graph_pairs_test

    def do_pickle_dataset(self):
        """Serialises the current parser state to a pickle file at cfg.pkl_path.

        In debug mode the collections are truncated (100 train/test items or
        200 'all' items) before pickling to reduce file size during development.
        """
        with open(self.cfg.pkl_path, 'wb') as f:
            if self.cfg.debug and len(self.graph_pairs) == 0:
                if 'train' in self.graphs and 'test' in self.graphs:
                    self.graphs['train'] = self.graphs['train'][:100]
                    self.graphs['test'] = self.graphs['test'][:100]
                elif 'all' in self.graphs:
                    self.graphs['all'] = self.graphs['all'][:200]
            pkl.dump(self, f)

    def normalize(self):
        """Placeholder for graph normalisation logic (not yet implemented)."""
        pass

    def read_node_labels(self, key):
        """Reads and registers all unique node-type labels found in the dataset.

        Scans every topModule.json file under the given dataset key, extracts the
        node-type component from each signal name, and populates self.node_labels,
        self.label2idx, and self.idx2label.

        Args:
            key (str): Dataset split key (e.g. 'TjFree', 'TjIn') used to locate
                       the correct sub-folder within self.root_path.
        """
        # read thru all the node labels in a dataset.
        for json_path in glob("%s/**/**/topModule.json" % str(self.root_path / key), recursive=True):
            with open(str(json_path), 'r') as json_file:
                edge_list_dict = json.loads(json_file.read())
                for src in edge_list_dict:
                    node_name = src
                    if '_graphrename' in src:
                        node_name = src[:src.index('_graphrename')]
                    if '.' in node_name:
                        type_of_node = node_name.split('.')[-1]
                    elif '_' in node_name:
                        type_of_node = node_name.split('_')[-1]
                    else:
                        type_of_node = node_name.lower()
                    self.node_labels.add(type_of_node)

        self.label2idx = {v: k for k, v in enumerate(list(self.node_labels))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def split_dataset(self, ratio, seed, dataset):
        """Splits a dataset into stratified train and test subsets.

        Stratification is based on the binary label of each item (1 or -1),
        preserving class balance across the split.

        Args:
            ratio (float): Fraction of the dataset to use for training (e.g. 0.8).
            seed (int): Random seed for reproducible shuffling.
            dataset (list): List of graph data objects or pair tuples to split.

        Returns:
            tuple: (train_set, test_set) — two lists produced by sklearn's
                   train_test_split with stratification.
        """
        train_size = int(len(dataset) * ratio)

        sim_diff_label = []

        for data in dataset:
            if type(data) == tuple:
                if data[2] == 1:
                    sim_diff_label.append(1)
                else:
                    sim_diff_label.append(-1)
            else:
                if data.label == 1:
                    sim_diff_label.append(1)
                else:
                    sim_diff_label.append(-1)

        return train_test_split(dataset, train_size=train_size, shuffle=True, stratify=sim_diff_label, random_state=seed)

    def get_graph(self, json_string):
        """Constructs a NetworkX DiGraph from a JSON edge-list dictionary.

        Parses node names to extract their type label and adds them to the graph.
        Edge labels are read but not currently attached to the graph edges.

        Args:
            json_string (dict): Dictionary with an 'edge_index' key mapping each
                                source node name to a list of (edge_label, dst) pairs.

        Note:
            Normalisation step is not yet implemented (marked TODO).
        """
        # create nx graph.
        edge_list_dict = json_string['edge_index']
        hardware_graph = nx.DiGraph()
        for src in edge_list_dict:
            node_name = src
            if '_graphrename' in src:
                node_name = src[:src.index('_graphrename')]
            if '.' in node_name:
                type_of_node = node_name.split('.')[-1]
            elif '_' in node_name:
                type_of_node = node_name.split('_')[-1]
            else:
                type_of_node = node_name.lower()
            hardware_graph.add_node(src, x=self.label2idx[type_of_node], label=type_of_node)
            assert(type(edge_list_dict[src]) == list)
            for neighbor in edge_list_dict[src]:
                edge_label = neighbor[0]
                dst = neighbor[1]
                hardware_graph.add_edge(src, dst)
        #TODO: perform normalization

    def get_graph_from_json(self, json_path):
        """Loads a topModule.json file and returns a NetworkX DiGraph.

        Args:
            json_path (str | Path): Absolute or relative path to the JSON file
                                    containing the edge-list dictionary.

        Returns:
            nx.DiGraph: Directed graph with node attributes 'x' (integer label index)
                        and 'label' (string type name).
        """
        with open(str(json_path), 'r') as json_file:
            hardware_graph = nx.DiGraph()
            edge_list_dict = json.loads(json_file.read())
            for src in edge_list_dict:
                node_name = src
                if '_graphrename' in src:
                    node_name = src[:src.index('_graphrename')]
                if '.' in node_name:
                    type_of_node = node_name.split('.')[-1]
                elif '_' in node_name:
                    type_of_node = node_name.split('_')[-1]
                else:
                    type_of_node = node_name.lower()
                hardware_graph.add_node(src, x=self.label2idx[type_of_node], label=type_of_node)
                assert(type(edge_list_dict[src]) == list)
                for neighbor in edge_list_dict[src]:
                    edge_label = neighbor[0]
                    dst = neighbor[1]
                    hardware_graph.add_edge(src, dst)

        return hardware_graph


class VerilogParser:
    """The only class that interfaces with pyverilog.

    Wraps PyVerilog's dataflow analyser, optimizer, graph generator, and
    control-flow analyser behind a unified API. Depending on the flags passed
    to __init__, it can operate in DFG mode, CFG mode, or AST mode.

    References:
        https://github.com/Microsoft/vscode-tips-and-tricks#intellisense
    """

    # Holds a graph generator instance
    def __init__(self, verilog_file, output_directory, top_module, generate_cfg=False, generate_ast=False):
        """Initialises the VerilogParser and runs the appropriate PyVerilog analysis pass.

        Args:
            verilog_file (str): Path to the input Verilog (.v) source file.
            output_directory (str): Directory path where generated output files are written.
            top_module (str): Name of the top-level Verilog module to analyse.
            generate_cfg (bool): If True, run control-flow analysis instead of DFG. Defaults to False.
            generate_ast (bool): If True, run AST extraction instead of DFG. Defaults to False.

        Raises:
            IOError: If verilog_file does not exist on disk.
        """
        print("Verilog file: ", verilog_file)
        print("Output directory: ", output_directory)
        if not os.path.exists(verilog_file):
            raise IOError("File Not Found:  ")
        self.output_directory = output_directory

        # Options
        self.dfg_graph_generator = None
        self.cfg_graph_generator = None
        self.ast = None
        self.ARRAY_GEN = None
        self.CONST_DICTIONARY_GEN = None
        self.DICTIONARY_GEN = None

        self._create_graphgen_obj(verilog_file, top_module, generate_cfg, generate_ast)

    # Helper function for __init__: creates a graph object used to generate JSON
    def _create_graphgen_obj(self, verilog_file, top_module, generate_cfg, generate_ast):
        """Runs PyVerilog's analysis pipeline and stores the appropriate generator object.

        Executes the dataflow analyser and optimizer unconditionally, then creates
        one of: a CFG analyser, an AST parse result, or a DFG graph generator,
        depending on the generate_cfg / generate_ast flags.

        Args:
            verilog_file (str): Path to the Verilog source file.
            top_module (str): Name of the top-level module.
            generate_cfg (bool): Create a control-flow analyser if True.
            generate_ast (bool): Parse the AST if True (takes priority over DFG when
                                 generate_cfg is False).
        """
        dataflow_analyzer = PyDataflowAnalyzer(verilog_file, top_module)
        dataflow_analyzer.generate()
        binddict = dataflow_analyzer.getBinddict()
        terms = dataflow_analyzer.getTerms()

        dataflow_optimizer = PyDataflowOptimizer(terms, binddict)
        dataflow_optimizer.resolveConstant()
        resolved_terms = dataflow_optimizer.getResolvedTerms()
        resolved_binddict = dataflow_optimizer.getResolvedBinddict()
        constlist = dataflow_optimizer.getConstlist()

        if generate_cfg:
            fsm_vars = tuple(['fsm', 'state', 'count', 'cnt', 'step', 'mode'])
            self.cfg_graph_generator = PyControlflowAnalyzer("top", terms, binddict,
                                        resolved_terms, resolved_binddict, constlist, fsm_vars)
        elif generate_ast:
            #when generating AST, determines which substructure (dictionary/array) to generate
            #before converting the json-like structure into actual json
            self.DICTIONARY_GEN = ["Source", "Description", "Ioport", "Decl", "Lvalue"]
            self.ARRAY_GEN = ["ModuleDef", "Paramlist", "Portlist", "Input", "Width", "Reg", "Wire", "Rvalue", "ParseSelect",
            "Uplus", "Uminus", "Ulnot", "Unot", "Uand", "Unand", "Uor", "Unor", "Uxnor", "Power", "Times", "Divide", "Mod", "Plus",
            "Minus", "Sll", "Srl", "Sla", "Sra", "LessThan", "GreaterThan", "LessEq", "GreaterEq", "Eq", "Eql", "NotEq", "Eql", "NotEql",
            "And", "Xor", "Xnor", "Or", "Land", "Lor", "Cond", "Assign", "Always", "AlwaysFF", "AlwaysComb", "AlwaysLatch",
            "SensList", "Sens", "Substitution", "BlockingSubstitution", "NonblockingSubstitution", "IfStatement", "Block",
            "Initial", "Plus", "Output", "Partselect"]
            self.CONST_DICTIONARY_GEN = ["IntConst", "FloatConst", "StringConst", "Identifier"]

            self.ast, _ = parse([verilog_file])

        else: #generate dfg
            self.dfg_graph_generator = PyGraphGenerator(top_module, terms, binddict, resolved_terms,
                                resolved_binddict, constlist,
                                f'{self.output_directory}seperate_modules.pdf')

    # Generates nested dictionary for conversion to JSON (AST helper)
    def _generate_ast_dict(self, ast_node):
        """Recursively converts a PyVerilog AST node into a nested Python dictionary.

        The structure of the output dictionary mirrors the AST node hierarchy and
        is suitable for serialisation to JSON via json.dumps().

        Args:
            ast_node: A PyVerilog AST node object.

        Returns:
            dict: Nested dictionary representation of the AST subtree rooted at ast_node.

        Raises:
            Exception: If ast_node's class name is not present in any of ARRAY_GEN,
                       DICTIONARY_GEN, or CONST_DICTIONARY_GEN.
        """
        class_name = ast_node.__class__.__name__
        structure = {}
        #based on the token class_name, determine the value type of class_name
        if class_name in self.ARRAY_GEN:
            structure[class_name] = [getattr(ast_node, n) for n in ast_node.attr_names] if ast_node.attr_names else []
            for c in ast_node.children():
                structure[class_name].append(self._generate_ast_dict(c))
        elif class_name in self.DICTIONARY_GEN:
            structure[class_name] = self._generate_ast_dict(ast_node.children()[0])
        elif class_name in self.CONST_DICTIONARY_GEN:
            structure = {}
            structure[class_name] = getattr(ast_node, ast_node.attr_names[0])

            return structure
        else:
            raise Exception(f"Error. Token name {class_name} is invalid or has not yet been supported")

        return structure

    # Generates abstract syntax tree for conversion (AST helper)
    def export_ast(self, nested_dictionary):
        """Serialises a nested AST dictionary to a JSON file named ast.json.

        Writes the file to self.output_directory/ast.json with 2-space indentation.

        Args:
            nested_dictionary (dict): The nested dictionary produced by _generate_ast_dict().
        """
        print(f'Saving abstract syntax tree as json')
        #print(f'{self.output_directory}/ast.json')
        with open(f'{self.output_directory}/ast.json', 'w') as f:
            f.write(dumps(nested_dictionary, indent=2))
        print('List of root nodes saved in ast.json.\n')

    # Generates dot file (CFG helper)
    def generate_dot_file(self, graph_format='png', no_label=False):
        """Extracts finite state machines from the CFG analyser and writes dot/image files.

        Iterates over all FSMs detected by PyVerilog's control-flow analyser,
        prints a summary to stdout, and saves each FSM as a graph image.

        Args:
            graph_format (str): Image format for the FSM output file (e.g. 'png', 'pdf').
                                Defaults to 'png'.
            no_label (bool): If True, omit transition labels from the graph. Defaults to False.

        Raises:
            AssertionError: If called when the parser was not initialised in CFG mode
                            (i.e. self.cfg_graph_generator is None).
        """
        assert self.cfg_graph_generator is not None, "Error: Generate dot file only if you are generating CFG's "

        fsms = self.cfg_graph_generator.getFiniteStateMachines()

        print("VIEWING FSM's")
        print("LENGTH OF FSM: ", len(fsms))
        for signame, fsm in fsms.items():
            print('# SIGNAL NAME: %s' % signame)
            print('# DELAY CNT: %d' % fsm.delaycnt)
            fsm.view()
            fsm.tograph(filename=util.toFlatname(signame) + '.' + graph_format, nolabel=no_label)

    # Generates CFG from dot file
    def export_cfg_graph(self, output='graph'):
        """Reads a dot file and exports CFG data as JSON in the requested format.

        Parses ./file.dot (produced by generate_dot_file) and writes one of:
        root nodes, all nodes, all edges, or the full adjacency dictionary to a JSON
        file in self.output_directory.

        Args:
            output (str): Selects the export format. One of:
                          'roots'  — only root nodes (no incoming edges),
                          'nodes'  — all node names,
                          'edges'  — all edges as [src, dst, label] triples,
                          'graph'  — full adjacency dict (default).
        """
        def default_val():
            """Returns the default value for a new defaultdict entry (empty list)."""
            return []

        graph = pydot.graph_from_dot_file("./file.dot")[0]

        nodes = [node.get_name() for node in graph.get_node_list()]
        root_nodes = [node.get_name() for node in graph.get_node_list() if node.obj_dict['parent_graph'] is None]
        edges = [[edge.get_source(), edge.get_destination(), edge.obj_dict['attributes']['label']] for edge in graph.get_edge_list()]
        # key: node, val: list of outgoing edges
        top_module = defaultdict(default_val)

        for edge in edges:
            if edge[2] == 'None':
                top_module[edge[0]].append("")
            else:
                top_module[edge[0]].append(edge[2])

        if (output == 'roots'):
            print(f'Saving all {len(root_nodes)} nodes as json')
            with open(f'{self.output_directory}root_nodes.json', 'w') as f:
                f.write(dumps(root_nodes, indent=4))
            print('List of root nodes saved in root_nodes.json.\n')

        elif (output == 'nodes'):
            print(f'Saving all {len(nodes)} nodes as json')
            with open(f'{self.output_directory}all_nodes.json', 'w') as f:
                f.write(dumps(nodes, indent=4))
            print('List of nodes saved in all_nodes.json.\n')

        elif (output == 'edges'):
            print(f'Saving all {len(edges)} edges as json')
            with open(f'{self.output_directory}all_edges.json', 'w') as f:
                f.write(dumps(edges, indent=4))
            print('List of edges saved in all_edges.json.\n')

        elif (output == 'graph'):
            print(f'Saving cfg graph dictionary as a json')
            with open(f'{self.output_directory}topModule.json', 'w') as f:
                f.write(dumps(top_module, indent=4))
            print('Saving cfg graph dictionary as a json.\n')
            print('The graph is saved as topModule.json.\n')

    # This function returns True, if the child is a child of checkParent
    def _isChild(self, graph, checkParent, child):
        """Checks whether a node is a descendant of a given ancestor in the graph.

        Args:
            graph: An AGraph (PyGraphviz) or compatible directed graph object.
            checkParent: The candidate ancestor node to check against.
            child: The node whose ancestry is being tested.

        Returns:
            bool: True if checkParent appears in the ancestor chain of child,
                  False otherwise.
        """
        # This function recursively returns a list of all the parents of a node up to the root
        def get_all_parents(node):
            """Recursively collects all ancestor nodes up to the graph root(s).

            Args:
                node: The node whose ancestors are to be collected.

            Returns:
                list: All ancestor nodes of the given node, in no particular order.
            """
            # if node has no parents
            if graph.in_degree(node) == 0:
                return []
            # if node has no grandparents
            elif sum([graph.in_degree(parent) for parent in graph.predecessors(node)]) == 0:
                return graph.predecessors(node)
            # recursive call, node has unknown generations of parents
            else:
                retlist = list()
                for parent in graph.predecessors(node):
                    x = get_all_parents(parent)
                    x.append(parent)
                    retlist += x

                return retlist

        all_parents = get_all_parents(child)

        return checkParent in all_parents

    # Generates separate subgraph per module
    def graph_separate_modules(self, draw_graph=False):
        """Generates an individual DFG subgraph for every signal in the bind dictionary.

        Iterates over all signals (sorted case-insensitively) and calls the PyVerilog
        graph generator for each one. Optionally renders the combined result to PDF.

        Args:
            draw_graph (bool): If True, saves the combined subgraphs as a PDF file.
                               Defaults to False.
        """
        # binddict with string keys
        signals = [str(bind) for bind in self.dfg_graph_generator.binddict]

        print(f'{len(signals)} signals to generate seprate subgraphs...')
        for num, signal in enumerate(sorted(signals, key=str.casefold), start=1):
            self.dfg_graph_generator.generate(signal, walk=False)
            print(f'\rProgress : {num} / {len(signals)}', end='', flush=True)
        print('\nThe subgraphs are generated.\n')

        if draw_graph:
            print(f'Saving subgraphs with {len(self.dfg_graph_generator.graph.nodes())} nodes as a pdf...')
            self.dfg_graph_generator.draw()
            print('The subgraphs are saved.\n')

    #merge the graphs
    def merge_graphs(self, draw_graph=False):
        """Merges duplicate signal nodes that arise from dot-notation vs underscore-notation naming.

        For every node whose label, when underscores are replaced with dots, matches
        a root node, the duplicate is removed and its incoming edges are redirected
        to the canonical (dot-notation) root node, provided doing so would not
        introduce a cycle.

        Args:
            draw_graph (bool): If True, saves the merged graph as a PDF file.
                               Defaults to False.
        """
        label_to_node = dict()
        for node in self.dfg_graph_generator.graph.nodes():
            if self.dfg_graph_generator.graph.in_degree(node) == 0:
                label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
                label_to_node[label] = node

        deleted = 0
        print('Merging subgraphs... ')
        for num, node in enumerate(self.dfg_graph_generator.graph.nodes(), start=1):
            label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
            if '_' in label and label.replace('_', '.') in label_to_node:
                parents = self.dfg_graph_generator.graph.predecessors(node)
                self.dfg_graph_generator.graph.delete_node(node)
                deleted += 1
                for parent in parents:
                    if not self._isChild(self.dfg_graph_generator.graph, label_to_node[label.replace('_', '.')], parent):
                        self.dfg_graph_generator.graph.add_edge(parent, label_to_node[label.replace('_', '.')])
            print(f'\rProgress : {num - deleted} / {len(self.dfg_graph_generator.graph.nodes())}', end='', flush=True)
        print('\nThe signals subgraphs are merged.\n')

        if draw_graph:
            print(f'Saving merged graph with {len(self.dfg_graph_generator.graph.nodes())} nodes as a pdf...')
            self.dfg_graph_generator.draw(f'{self.output_directory}merged_graph.pdf')
            print('The graphs are saved.\n')

    def get_root_nodes(self):
        """Returns all root nodes (nodes with no incoming edges) from the DFG.

        Returns:
            list | None: List of root nodes if a DFG generator is available,
                         otherwise None.
        """
        if self.dfg_graph_generator:
            return [node for node in self.dfg_graph_generator.graph.nodes() if self.dfg_graph_generator.graph.in_degree(node) == 0]

    def get_nodes(self):
        """Returns the complete node view of the DFG.

        Returns:
            NodeView | None: All nodes in the DFG graph, or None if no DFG is available.
        """
        if self.dfg_graph_generator:
            return self.dfg_graph_generator.graph.nodes()

    def get_edges(self):
        """Returns all edges of the DFG as (source, destination, label) triples.

        Returns:
            list[tuple] | None: A list of 3-tuples (src, dst, edge_label), or None
                                if no DFG is available.
        """
        if self.dfg_graph_generator:
            all_edges = list()
            for edge in self.dfg_graph_generator.graph.edges():
                all_edges.append((edge[0], edge[1], edge.attr['label']))

            return all_edges

    def get_edge_list(self):
        """Returns the DFG adjacency dictionary mapping each node to its outgoing edges.

        Returns:
            dict | None: A dictionary of the form
                         { node_str: [(edge_label, dst_str), ...], ... },
                         or None if no DFG is available.
        """
        if self.dfg_graph_generator:
            jsondict = {}
            for node in self.dfg_graph_generator.graph.nodes():
                jsondict[str(node)] = list()
                for child in self.dfg_graph_generator.graph.successors(node):
                    edge_label = self.dfg_graph_generator.graph.get_edge(node, child).attr['label']
                    jsondict[str(node)].append((edge_label, str(child)))

            return jsondict

    # export the dfg graphs
    def export_dfg_graph(self, output='graph'):
        """Exports DFG data to a JSON file in the requested format.

        Writes one of: root nodes, all nodes, all edges, or the full adjacency
        dictionary to self.output_directory.

        Args:
            output (str): Selects the export format. One of:
                          'roots'  — only root nodes (no incoming edges),
                          'nodes'  — all node names,
                          'edges'  — all edges as (src, dst, label) triples,
                          'graph'  — full adjacency dict saved as topModule.json (default).
        """
        if (output == 'roots'):
            root_nodes = [node for node in self.dfg_graph_generator.graph.nodes() if self.dfg_graph_generator.graph.in_degree(node) == 0]
            print(f'Saving {len(root_nodes)} root nodes as a json...')
            with open(f'{self.output_directory}root_nodes.json', 'w') as f:
                f.write(dumps(root_nodes, indent=4))
            print('List of root nodes saved in root_nodes.json.\n')

        elif (output == 'nodes'):
            all_nodes = (self.dfg_graph_generator.graph.nodes())
            print(f'Saving all {len(all_nodes)} nodes as a json...')
            with open(f'{self.output_directory}all_nodes.json', 'w') as f:
                f.write(dumps(all_nodes, indent=4))
            print('List of nodes saved in all_nodes.json.\n')

        elif (output == 'edges'):
            all_edges = list()
            for edge in self.dfg_graph_generator.graph.edges():
                all_edges.append((edge[0], edge[1], edge.attr['label']))
            print(f'Saving all {len(all_edges)} edges as a json...')
            with open(f'{self.output_directory}all_edges.json', 'w') as f:
                f.write(dumps(all_edges, indent=4))
            print('List of edges is saved in all_edges.json.\n')

        elif (output == 'graph'):
            jsondict = {}
            for node in self.dfg_graph_generator.graph.nodes():
                jsondict[str(node)] = list()
                for child in self.dfg_graph_generator.graph.successors(node):
                    edge_label = self.dfg_graph_generator.graph.get_edge(node, child).attr['label']
                    jsondict[str(node)].append((edge_label, str(child)))
            print(f'Saving graph dictionary as a json...')
            f = open(f'{self.output_directory}topModule.json', 'w')
            jsonstr = dumps(jsondict, indent=4)
            f.write(jsonstr)
            f.close()
            print('The graph is saved as topModule.json.\n')

    # TODO: to be refactored
    def graph_input_dependencies(self, draw_graph=False):
        """Filters the DFG to retain only nodes reachable from primary input ports.

        Identifies all nodes that are not connected (directly or transitively) to
        any module input port and removes them from the graph.

        Args:
            draw_graph (bool): If True, saves the pruned graph as a PDF after removal.
                               Defaults to False.
        """
        label_to_names = dict()
        for node in self.dfg_graph_generator.graph.nodes():
            label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
            if label not in label_to_names:
                label_to_names[label] = list()
            label_to_names[label].append(str(node))

        inputs = [self.dfg_graph_generator.graph.get_node(*label_to_names[str(term).replace('.', '_')]) for x, term in zip(self.dfg_graph_generator.terms, self.dfg_graph_generator.terms.values()) if len(x.get_module_list()) == 1 and 'Input' in term.termtype]

        print('Locating nodes not connected to inputs...')
        to_delete = list()
        for num, node in enumerate(self.dfg_graph_generator.graph.nodes(), start=1):
            label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
            if label not in inputs:
                is_disconnected = True
                for input_ in inputs:
                    if self._isChild(self.dfg_graph_generator.graph, node, input_):
                        is_disconnected = False
                if is_disconnected:
                    to_delete.append(node)
            print(f'\rProgress : {num} / {len(self.dfg_graph_generator.graph.nodes())}', end='', flush=True)
        print('\nRemoving nodes not connected to inputs...')
        for num, node in enumerate(to_delete, start=1):
            self.dfg_graph_generator.graph.delete_node(node)
            print(f'\rProgress : {num} / {len(to_delete)}', end='', flush=True)
        print('\nRemoval is complete.\n')

        if draw_graph:
            print(f'Saving graph with {len(self.dfg_graph_generator.graph.nodes())} nodes as a pdf...')
            self.dfg_graph_generator.draw(f'{self.output_directory}input_dependencies.pdf')
            print('Graph saved.\n')

        #cleanup dot file and other residual files generated earlier

    def cleanup_files(self):
        """Removes transient files generated by PyVerilog during analysis.

        Silently ignores missing files. Files removed: file.dot, parser.out,
        parsetab.py, top_state.png.
        """
        for filename in ['file.dot', 'parser.out', 'parsetab.py', 'top_state.png']:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass


class DfgGenerator:
    """Generates a Data Flow Graph (DFG) from RTL (Register Transfer Level) Verilog code.

    Wraps VerilogParser to orchestrate subgraph generation, merging, and JSON export
    for DFG-based analysis pipelines.
    """

    def __init__(self, verilog_file, output_path, code_language='verilog', top_module='top', draw_graph=False):
        """Initialises the DFG generator and runs the full DFG extraction pipeline.

        Args:
            verilog_file (str): Path to the input Verilog source file.
            output_path (str): Directory path for all generated output files.
            code_language (str): Source language identifier. Only 'verilog' is currently
                                 supported. Defaults to 'verilog'.
            top_module (str): Name of the top-level Verilog module. Defaults to 'top'.
            draw_graph (bool): Reserved for future use; not currently applied. Defaults to False.

        Raises:
            IOError: If verilog_file does not exist when code_language is 'verilog'.
        """
        if code_language == "verilog":
            if not os.path.exists(verilog_file):
                raise IOError("file not found: " + verilog_file)

            print("Reading ", verilog_file)
            print(f'Outputting to : {output_path}\n')
            if not os.path.exists(f'{output_path}'):
                os.makedirs(os.path.dirname(f'{output_path}'))
            self.verilog_parser = VerilogParser(verilog_file, output_path, top_module, generate_cfg=False)
            self._generate_dfg()

    def process(self):
        """Runs the subgraph generation and merge steps without exporting to disk.

        Useful for in-memory graph manipulation before calling get_graph_json().
        """
        self.verilog_parser.graph_separate_modules()
        self.verilog_parser.merge_graphs()

    def get_graph_json(self):
        """Returns a dictionary containing all graph components for JSON serialisation.

        Returns:
            dict: Dictionary with keys:
                  'root_nodes' — list of root nodes,
                  'nodes'      — all graph nodes,
                  'edges'      — list of (src, dst, label) triples,
                  'edge_index' — adjacency dict { node: [(label, dst), ...] }.
        """
        graph_json = {}
        graph_json['root_nodes'] = self.verilog_parser.get_root_nodes()
        graph_json['nodes'] = self.verilog_parser.get_nodes()
        graph_json['edges'] = self.verilog_parser.get_edges()
        graph_json['edge_index'] = self.verilog_parser.get_edge_list()

        return graph_json

    def _generate_dfg(self):
        """Runs the full DFG pipeline: subgraph generation, merging, and JSON export."""
        self.verilog_parser.graph_separate_modules()
        self.verilog_parser.merge_graphs()
        self.verilog_parser.export_dfg_graph(output='graph')

    def export_nodes(self):
        """Exports all DFG nodes to all_nodes.json in the output directory."""
        self.verilog_parser.export_dfg_graph(output='nodes')

    def export_edges(self):
        """Exports all DFG edges to all_edges.json in the output directory."""
        self.verilog_parser.export_dfg_graph(output='edges')

    def export_root_nodes(self):
        """Exports all DFG root nodes to root_nodes.json in the output directory."""
        self.verilog_parser.export_dfg_graph(output='roots')

    def check_dependency(self):
        """Filters the DFG to only nodes reachable from primary input ports.

        Delegates to VerilogParser.graph_input_dependencies().
        """
        self.verilog_parser.graph_input_dependencies()


class CfgGenerator:
    """Generates a Control Flow Graph (CFG) from RTL Verilog code.

    Wraps VerilogParser to orchestrate dot-file generation, CFG JSON export,
    and cleanup of transient files.
    """

    def __init__(self, verilog_file, output_path, code_language="verilog", top_module="top", draw_graph=False):
        """Initialises the CFG generator and runs the full CFG extraction pipeline.

        Args:
            verilog_file (str): Path to the input Verilog source file.
            output_path (str): Directory path for all generated output files.
            code_language (str): Source language identifier. Only 'verilog' is currently
                                 supported. Defaults to 'verilog'.
            top_module (str): Name of the top-level Verilog module. Defaults to 'top'.
            draw_graph (bool): Reserved for future use; not currently applied. Defaults to False.

        Raises:
            IOError: If verilog_file does not exist when code_language is 'verilog'.
        """
        if code_language == "verilog":
            if not os.path.exists(verilog_file):
                raise IOError("file not found: " + verilog_file)

            print("Reading ", verilog_file)
            print(f'Outputting to : {output_path}\n')
            if not os.path.exists(f'{output_path}'):
                os.makedirs(os.path.dirname(f'{output_path}'))

            self.parser = VerilogParser(verilog_file, output_path, top_module, generate_cfg=True)
            self._generate_cfg()

    def _generate_cfg(self):
        """Runs the full CFG pipeline: dot-file generation, JSON export, and cleanup."""
        self.parser.generate_dot_file()
        self.parser.export_cfg_graph(output='graph')
        self.parser.cleanup_files()

    def export_nodes(self):
        """Exports all CFG nodes to all_nodes.json in the output directory."""
        self.parser.export_cfg_graph(output='nodes')

    def export_edges(self):
        """Exports all CFG edges to all_edges.json in the output directory."""
        self.parser.export_cfg_graph(output='edges')

    def export_root_nodes(self):
        """Exports all CFG root nodes to root_nodes.json in the output directory."""
        self.parser.export_cfg_graph(output='roots')


class AstGenerator:
    """Generates an Abstract Syntax Tree (AST) in JSON format from a Verilog source file."""

    def __init__(self, verilog_file, output):
        """Initialises the AST generator with a VerilogParser configured for AST mode.

        Args:
            verilog_file (str): Path to the input Verilog source file.
            output (str): Directory path where ast.json will be written.
        """
        self.parser = VerilogParser(verilog_file, output, "top", generate_ast=True)

    def generate_ast_json(self):
        """Converts the parsed AST to a nested dictionary and writes it to ast.json.

        Calls VerilogParser._generate_ast_dict() recursively on the root AST node,
        exports the result via VerilogParser.export_ast(), and removes transient files.
        """
        ast_dict = self.parser._generate_ast_dict(self.parser.ast)
        self.parser.export_ast(ast_dict)
        self.parser.cleanup_files()


class PreprocessVerilog:
    """Provides preprocessing utilities for Verilog files in RTL and Gate-Level Netlist (GLN) formats.

    All static methods operate purely on file paths and do not require an instance.
    """

    def __init__(self, code_language="verilog"):
        """Initialises the preprocessor for the given hardware description language.

        Args:
            code_language (str): Target language. Currently only 'verilog' is supported.
                                 Defaults to 'verilog'.
        """
        if code_language == "verilog":
            pass

    @staticmethod
    def remove_comments(input_path, target_path):
        """Strips inline comments from a Verilog file by truncating each line at the first '/'.

        Args:
            input_path (str): Path to the source Verilog file.
            target_path (str): Path where the comment-stripped file will be written.

        Note:
            This is a simple character-search approach and may incorrectly strip
            content from string literals that contain '/'.
        """
        # read the file into a list of lines
        with open(input_path, 'r') as file_in:
            lines = file_in.read().split("\n")

        file_out = open(target_path, 'w')
        modules_dic = {}
        for i, line in enumerate(lines):
            idx = line.find('/')
            file_out.write(line[0:idx] + '\n')

        file_in.close()
        file_out.close()

    @staticmethod
    def rename_top_module(input_path, target_path):
        """Identifies the top-level module and renames it to 'top' throughout the file.

        Determines the top module by finding the module name that appears exactly once
        in the file (i.e., it is defined but never instantiated elsewhere).

        Args:
            input_path (str): Path to the source Verilog file.
            target_path (str): Path where the renamed file will be written.
        """
        # read the file into a list of lines
        with open(input_path, 'r') as file_in:
            lines = file_in.read().split("\n")

        modules_dic = {}
        # iterate over lines, and list the module names.
        for i, line in enumerate(lines):
            words = line.split()
            for j, word in enumerate(words):
                if word == 'module':
                    module_name = words[j + 1]
                    if '(' in module_name:
                        idx = module_name.find('(')
                        module_name = module_name[:idx]
                        modules_dic[module_name] = 1
                    else:
                        modules_dic[module_name] = 0

        # iterate over file and count the occurance of each module name.
        for i, line in enumerate(lines):
            words = line.split()
            for j, word in enumerate(words):
                for m in modules_dic:
                    if word == m:
                        modules_dic[m] = modules_dic[m] + 1
        print("a dictionary of module names and the number of their occurance: ", modules_dic)

        # find the name of top module
        for m in modules_dic:
            if modules_dic[m] == 1:
                top_module = m
                break

        # rename the top module to 'top'
        file_out = open(target_path, 'w')
        for line in lines:
            file_out.write(line.replace(top_module, 'top') + '\n')
        file_in.close()
        file_out.close()

    @staticmethod
    def flatten(input_path, target_path):
        """Concatenates all .v files in a directory into a single topModule.v file.

        Args:
            input_path (str): Directory containing the .v source files to concatenate.
            target_path (str): Directory where the output topModule.v will be written.
        """
        with open(target_path + "/topModule.v", "wt") as outfile:
            print(input_path, glob(fr'{input_path}/*.v'))
            for verilog_file in glob(fr'{input_path}/*.v'):
                with open(verilog_file, "rt") as infile:
                    outfile.write(infile.read())

    def remove_underscores(self, input_file):
        """Removes all underscore characters from a Verilog file and overwrites the result.

        Args:
            input_file (str): Path to the Verilog file to process.

        Note:
            The output file path is currently hardcoded to the variable 'filename',
            which is not defined in this method (known issue — to be refactored).
        """
        with open(input_file, 'r') as f:
            s = f.read().replace('_', '')
        with open(filename, 'w') as f:
            f.write(s)

    def flatten(self, input_path, output_path, top_module):
        """Concatenates all .v files in input_path, replacing top_module name with 'top'.

        Unlike the static flatten(), this instance method also substitutes the
        top-module name across all concatenated content.

        Args:
            input_path (str): Directory containing the .v source files.
            output_path (str): Ignored — the output path is derived from the location
                               of this source file (__file__).
            top_module (str): The original top-module name to replace with 'top'.
        """
        output_path = os.path.dirname(os.path.abspath(__file__))
        with open(output_path + "/topModule.v", "wt") as outfile:
            for verilog_file in glob(fr'{input_path}/*.v'):
                with open(verilog_file, "rt") as infile:
                    outfile.write(infile.read().replace(top_module, 'top'))