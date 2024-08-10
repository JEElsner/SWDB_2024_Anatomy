from brainglobe_atlasapi import BrainGlobeAtlas
from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm

import json
import k3d
import math
import networkx as nx
import numpy as np
import os

ATLAS = BrainGlobeAtlas('allen_mouse_100um')


def euclidean_distance(node1, node2):
        """
        Calculate the Euclidean distance between two nodes.
    
        Parameters:
        node1, node2 (dict): Nodes with 'pos' key containing x, y, z coordinates.
    
        Returns:
        float: Euclidean distance between node1 and node2.
        """
        pos1 = np.array(node1['pos'])
        pos2 = np.array(node2['pos'])
        return np.linalg.norm(pos1 - pos2)

def add_node_to_graph(graph, node):
        """
        Add a node with attributes to the graph.
    
        Parameters:
        graph (nx.DiGraph): The graph to which the node will be added.
        node (dict): Node data.
        """
        graph.add_node(
            node['sampleNumber'], 
            pos=(node['x'], node['y'], node['z']), 
            radius=node['radius'], 
            structure_id=node['structureIdentifier'],
            allen_id=node['allenId']
        )

def add_edge_to_graph(graph, parent, child):
        """
        Add an edge between parent and child nodes in the graph, with weight as Euclidean distance.
    
        Parameters:
        graph (nx.DiGraph): The graph to which the edge will be added.
        parent, child (int): The sampleNumbers of the parent and child nodes.
        """
        graph.add_edge(
            parent, 
            child, 
            weight=euclidean_distance(
                graph.nodes()[parent],
                graph.nodes()[child]
            )
        )

def json_to_digraph(file_path):
        """
        Load a neuronal reconstruction from a JSON file into a NetworkX graph.
    
        The JSON file contains SWC data with additional brain region information for each node.
        The graph will be a directed tree.
    
        Parameters:
        file_path (str): Path to the JSON file containing reconstruction data.
    
        Returns:
        nx.DiGraph: A directed graph representing the neuronal tree.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except IOError as e:
            print(f"Error opening file: {e}")
            return None
    
        # Certain JSON files may have a single 'neuron' object instead of a 'neurons' array
        neuron_data = data['neuron'] if 'neuron' in data else data['neurons'][0]
    
        axon_graph, dendrite_graph = nx.DiGraph(), nx.DiGraph()
    
        for structure, graph in [('dendrite', dendrite_graph), ('axon', axon_graph)]:
            if structure not in neuron_data:
                # Some reconstructions may be missing an axon or dendrite tracing
                print(f"Missing structure {structure} for {file_path}")
                continue
            for node in sorted(neuron_data[structure], key=lambda x: x['sampleNumber']):
                add_node_to_graph(graph, node)
                if node['parentNumber'] != -1:
                    add_edge_to_graph(graph, node['parentNumber'], node['sampleNumber'])
                    
        if dendrite_graph.nodes() and axon_graph.nodes():
            # Remove duplicate soma node from axon graph
            axon_graph.remove_node(1)  
    
        # The sampleNumber starts at 1 for both axon and dendrite, so 
        # relabel axon nodes to avoid key collisions when merging the graphs,.
        first_axon_label = max(dendrite_graph.nodes()) + 1 if dendrite_graph.nodes() else 1
        joined_graph = nx.union(
            dendrite_graph, 
            nx.convert_node_labels_to_integers(
                axon_graph, 
                first_label=first_axon_label
            )
        )
        roots = [n for n in joined_graph if joined_graph.in_degree(n) == 0]
        # Link the dendrite to the axon
        if len(roots) == 2:
            add_edge_to_graph(joined_graph, roots[0], roots[1])
    
        return joined_graph

# Define a function for filtering the graph based on attribute values
def get_subgraph(G, attribute, values):
        """
        Extract a subgraph from the given graph based on specified attribute values.
    
        Parameters:
        G (nx.Graph): The original graph from which to extract the subgraph.
        attribute (str): The node attribute used for filtering.
        values (tuple): A tuple of attribute values to include in the subgraph.
    
        Returns:
        nx.Graph: A subgraph of G containing only nodes with the specified attribute values.
        """
        filtered_nodes = [node for node, attr in G.nodes(data=True) if attr.get(attribute) in values]
        return G.subgraph(filtered_nodes)


def load_graphs(filepaths):
    """
    Load all JSON files in the given directory as graphs using multiprocessing.

    Parameters:
    directory_path (str): Path to the directory containing JSON files.

    Returns:
    list of nx.Graph: A list of graphs loaded from the JSON files.
    """
    # Use multiprocessing pool to load graphs in parallel
    with Pool() as pool:
        graphs = list(tqdm(pool.imap(json_to_digraph, filepaths), total=len(filepaths)))

    # Remove None values from the list in case there were errors
    return [graph for graph in graphs if graph is not None]


def get_ccf_ids(skel, compartment_type=None, vertex_type=None, depth=None):
    skel = deepcopy(skel)
    vertices = get_vertices(skel, compartment_type, vertex_type)
    ccf_ids = skel.vertex_properties["ccf"][vertices]
    if depth is not None:
        return [get_ccf_id_by_depth(ccf_id, depth) for ccf_id in ccf_ids]
    else:
        return ccf_ids


def get_vertices(skel, compartment_type, vertex_type):
    # Special Cases
    if compartment_type == 1:
        return [skel.root]
    elif not compartment_type and not vertex_type:
        return skel.vertex_properties['compartment'] != -1

    # General Cases
    if compartment_type and not vertex_type:
        return skel.vertex_properties['compartment'] == compartment_type
    else:
        assert vertex_type in ["branch_points", "end_points"]
        verts = skel.end_points if vertex_type == "end_points" else skel.branch_points
        if compartment_type:
            vertex_compartments = skel.vertex_properties['compartment']
            return [v for v in verts if vertex_compartments[v] == compartment_type]
        else:
            return verts


def get_ccf_id_by_depth(ccf_id, depth):
    if ccf_id in ATLAS.structures:
        structures = ATLAS.structures[ccf_id]["structure_id_path"]
        return structures[min(depth, len(structures) - 1)]
    else:
        return ccf_id


def get_connectivity_matrix(skels, binary=False, depth=None):
    # Initializations
    ccf_ids_list = [
        get_ccf_ids(
            skel, compartment_type=2, vertex_type="end_points", depth=depth
        )
        for skel in skels
    ]
    regions, cnts = np.unique(np.concatenate(ccf_ids_list), return_counts=True)
    region_to_idx = dict({r: idx for idx, r in enumerate(regions[cnts > 10])})

    # Populate matrix
    matrix = np.zeros((len(skels), len(region_to_idx)))
    for i, ccf_ids in enumerate(ccf_ids_list):
        ccf_ids, cnts = np.unique(ccf_ids, return_counts=True)
        for j, ccf_id in enumerate(ccf_ids):
            if not math.isnan(ccf_id) and ccf_id in region_to_idx.keys():
                matrix[i, region_to_idx[ccf_id]] = cnts[j]
    idx_to_region = {idx: r for r, idx in region_to_idx.items()}
    return (matrix > 0 if binary else matrix), idx_to_region
