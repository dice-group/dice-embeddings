import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph_with_highlight(file_path, highlight_node, figsize=(20, 20)):

    df = pd.read_csv(file_path, sep='\t', header=None, names=["head", "relation", "tail"])

    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(row['head'], row['tail'], label=row['relation'])

    highlight_edges = [(u, v) for u, v in G.edges() if u == highlight_node or v == highlight_node]

    pos = nx.spring_layout(G, k=0.5, seed=42)

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    if highlight_node in G.nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=[highlight_node], node_size=800, node_color="orange")

    nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color="red", width=2, arrowsize=15)

    edge_labels = {(u, v): G[u][v]['label'] for u, v in highlight_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8)

    #plt.title(f"Knowledge Graph Highlighting '{highlight_node}'", fontsize=14)
    plt.axis('off')
    plt.show()

    return G



graph = visualize_graph_with_highlight("../KGs/UMLS/train.txt", "animal")
