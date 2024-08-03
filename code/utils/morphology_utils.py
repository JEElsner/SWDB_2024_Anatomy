import os
import json
import random
from multiprocessing import Pool
import numpy as np
import networkx as nx
from tqdm import tqdm
import pandas as pd

def get_longest_path(graph):
    longest_path = nx.dag_longest_path(graph, weight='weight')
    longest_path_edges = zip(longest_path, longest_path[1:])
    return longest_path, sum(graph.get_edge_data(*e)['weight'] for e in longest_path_edges)


def find_branch_points(graph):
        """
        Identify all branch points in a graph. A branch point is defined as a node with more than one child.
    
        Parameters:
        graph (nx.Graph): The graph to analyze.
    
        Returns:
        list: A list of nodes that are branch points.
        """
        branch_points = [node for node in graph.nodes() if graph.out_degree(node) > 1]
        return branch_points

def find_terminals(graph):
        """
        Find all terminal nodes in a graph. A terminal node is defined as a node with no children.
    
        Parameters:
        graph (nx.Graph): The graph to analyze.
    
        Returns:
        list: A list of nodes that are terminal nodes.
        """
        terminal_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]
        return terminal_nodes

def find_roots(graph):
        """
        Find the root nodes of a graph. A root node is defined as a node with no parent.
    
        Parameters:
        graph (nx.Graph): The graph to analyze.
    
        Returns:
        list: A list of root nodes.
        """
        roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        return roots

def total_length(graph):
        """
        Calculate the total length of all edges in the graph.
    
        Parameters:
        graph (nx.Graph): The graph with weighted edges.
    
        Returns:
        float: Total length of all edges in the graph.
        """
        total_length = sum(weight for _, _, weight in graph.edges.data("weight"))
        return total_length

def bounding_box(graph):
        """
        Compute the bounding box of the XYZ coordinates of the graph nodes.
    
        Parameters:
        graph (nx.Graph): The graph with nodes containing XYZ coordinates.
    
        Returns:
        tuple: A tuple containing two tuples - the minimum and maximum XYZ coordinates.
               Format: ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        # Initialize min and max coordinates with the first node's coordinates
        first_node = list(graph.nodes(data=True))[0][1]
        min_x, min_y, min_z = first_node['pos']
        max_x, max_y, max_z = first_node['pos']
    
        for _, attr in graph.nodes(data=True):
            x, y, z = attr['pos']
            min_x, min_y, min_z = min(min_x, x), min(min_y, y), min(min_z, z)
            max_x, max_y, max_z = max(max_x, x), max(max_y, y), max(max_z, z)
    
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))

def get_depth(bounding_box):
        """
        Calculate the depth of the graph based on the bounding box.
    
        Parameters:
        bounding_box (tuple): Bounding box of the graph.
    
        Returns:
        float: Depth of the graph.
        """
        min_z, max_z = bounding_box[0][2], bounding_box[1][2]
        return max_z - min_z

def get_height(bounding_box):
        """
        Calculate the height of the graph based on the bounding box.
    
        Parameters:
        bounding_box (tuple): Bounding box of the graph.
    
        Returns:
        float: Height of the graph.
        """
        min_y, max_y = bounding_box[0][1], bounding_box[1][1]
        return max_y - min_y

def get_width(bounding_box):
        """
        Calculate the width of the graph based on the bounding box.
    
        Parameters:
        bounding_box (tuple): Bounding box of the graph.
    
        Returns:
        float: Width of the graph.
        """
        min_x, max_x = bounding_box[0][0], bounding_box[1][0]
        return max_x - min_x


def get_centroid(bounding_box):
        """
        Calculate the centroid of the graph based on the bounding box.
    
        Parameters:
        bounding_box (tuple): Bounding box of the graph.
    
        Returns:
        tuple: The centroid (x, y, z) of the graph.
        """
        min_coords, max_coords = bounding_box
        centroid = tuple((min_coords[i] + max_coords[i]) / 2 for i in range(3))
        return centroid

def get_branches(graph):
    """
    Extract branches from the graph by tracing paths backwards from branch points and terminal nodes.

    Parameters:
    graph (nx.Graph): The graph from which to extract branches.

    Returns:
    list: A list of branches, where each branch is a list of tuples (u, v, edge_data).
          Each tuple represents an edge in the branch.
    """
    # Find all branch points and terminals
    relevant_nodes = find_branch_points(graph) + find_terminals(graph)

    branches = []
    for node in relevant_nodes:
        # Skip root nodes as they don't lead to a previous relevant node
        if graph.in_degree(node) == 0:
            continue

        path = []
        current = node
        # Walk back to the previous relevant node or root
        while graph.in_degree(current) > 0:
            parent = next(graph.predecessors(current))
            edge_data = graph.get_edge_data(parent, current)
            path.insert(0, (parent, current, edge_data))
            
            # Break the loop if a new branch point is reached
            if graph.out_degree(parent) > 1:
                break
            current = parent

        branches.append(path)

    return branches


def calculate_branch_length(branch):
    """
    Calculate the total length of a branch using the edge weights.

    Parameters:
    branch (list of tuples): Each tuple is an edge represented as (u, v, edge_data),
                             where edge_data is a dictionary containing the edge's weight.

    Returns:
    float: The total length of the branch.
    """
    total_length = 0.0
    for _, _, edge_data in branch:
        total_length += edge_data.get('weight', 0.0)  # Add the edge's weight, defaulting to 0 if not present

    return total_length

def get_metrics(graph):
    metrics = {
        'num_branch_points': len(find_branch_points(graph)),
        'num_terminals': len(find_terminals(graph)),
        'total_length': total_length(graph)
    }

    branch_lengths = np.array([calculate_branch_length(branch) for branch in get_branches(graph)])
    metrics.update({
        'mean_branch_length': branch_lengths.mean(),
        'std_branch_length': branch_lengths.std(),
        'max_branch_length': branch_lengths.max()
    })
    
    metrics['longest_path_length'] = get_longest_path(graph)[1] # just the length

    bbox = bounding_box(graph)
    metrics.update({
        'depth': get_depth(bbox),
        'width': get_width(bbox),
        'height': get_height(bbox)
    })

    centroid = get_centroid(bbox)
    metrics.update({
        'centroid_x': centroid[0],
        'centroid_y': centroid[1],
        'centroid_z': centroid[2]
    })

    return metrics

def all_graph_metrics(graphs):
    all_metrics = []
    for graph in tqdm(graphs):
        metrics = get_metrics(graph)
        all_metrics.append(metrics)
    df = pd.DataFrame(all_metrics)
    return df