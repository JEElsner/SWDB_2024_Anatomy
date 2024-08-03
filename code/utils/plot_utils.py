import os
import json
import random
from multiprocessing import Pool

import k3d
import numpy as np
import networkx as nx
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def plot_graph_2d(graph, ax, color="blue", node_size=0.01, width=1):
    """
    Generate a 2D plot of a neuronal tree graph using matplotlib.

    Parameters:
    graph (nx.DiGraph): A directed graph representing the neuronal tree.
    """
    # Extracting node positions
    pos = {node: data['pos'][:2] for node, data in graph.nodes(data=True)}

    nx.draw(
        graph, 
        pos,
        ax=ax,
        arrows=False,
        width=width,
        with_labels=False, 
        node_size=node_size, 
        node_color=color,
        edge_color=color
    )

def plot_graph_branches(graph, branches):
    """
    Plots the branches of the graph in random colors using the 'pos' attribute of nodes for positioning.

    Parameters:
    graph (nx.Graph): The graph from which to plot the branches.
    """

    # Set up the plot
    plt.figure(figsize=(10, 10))

    # Retrieve positions from the 'pos' attribute of the nodes
    pos = {node: graph.nodes[node]['pos'][:2] for node in graph.nodes}

    # Generate random colors for each branch
    colors = [ "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
               for _ in range(len(branches)) ]

    for branch, color in zip(branches, colors):
        # Extract nodes and edges from each branch
        branch_nodes = {u for u, v, d in branch} | {v for u, v, d in branch}
        branch_edges = [(u, v) for u, v, d in branch]

        # Draw nodes and edges for each branch
        nx.draw_networkx_nodes(graph, pos, nodelist=branch_nodes, node_color=[color], node_size=0.01)
        nx.draw_networkx_edges(graph, pos, edgelist=branch_edges, arrows=False, width=1, edge_color=[color])

    plt.axis('off')
    plt.show()

def plot_path_overlay(graph, path, path_color='red', graph_color='blue'):
    """
    Plot the graph with the longest path highlighted in a different color.

    Parameters:
    graph (nx.Graph): The graph to be visualized.
    path_color (str): Color for the longest path. Default is 'red'.
    graph_color (str): Color for the rest of the graph. Default is 'blue'.
    """

    path_edges = set(zip(path, path[1:]))

    # Draw the graph
    pos = {node: data['pos'][:2] for node, data in graph.nodes(data=True)}
    nx.draw(graph, pos, arrows=False, width=1, node_size=1, with_labels=False, node_color=graph_color, edge_color=graph_color)

    # Draw the longest path on top of the graph
    path_edge_colors = [path_color if (u, v) in path_edges or (v, u) in path_edges else graph_color for u, v in graph.edges()]
    nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color=path_color, node_size=1)
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=path_color, width=1, arrows=False)

    plt.show()


def rgb_to_hex(r,g,b):
    # Convert to a hexadecimal string
    hex_color = f'{r:02x}{g:02x}{b:02x}'
    # Convert the hexadecimal string to an integer in base-16
    color_int = int(hex_color, 16)
    return color_int

def random_color_hex():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return rgb_to_hex(red, green, blue)

def graph_to_lines(g,color=None):
    # Extract vertex positions
    g_verts = np.array([g.nodes[n]['pos'] for n in sorted(g.nodes())])
    # Pairs of indices into the vertex array are edges
    # Node keys start at 1
    g_inds = np.array([[u-1, v-1] for u, v in g.edges()])
    if color is None:
        color=random_color_hex()

    
    g_lines = k3d.factory.lines(g_verts, g_inds, indices_type='segment', color=color, width=1, shader='simple')
    return g_lines

def plot_graphs(graphs, plot, color = None):
    for i, g in enumerate(graphs):
        g_lines = graph_to_lines(g,color)
        plot += g_lines

