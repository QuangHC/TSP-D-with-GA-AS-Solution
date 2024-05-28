import networkx as nx
import matplotlib.pyplot as plt

def draw_route(path, type_transport):
    num_cities = len(path) - 1
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for i in range(num_cities):
        G.add_node(path[i])
    
    # Add edges to the graph based on transportation types
    for i in range(num_cities):
        node_i, node_next = type_transport[i], type_transport[i + 1]
        start, end = None, None
        if node_next == 1:
            if node_i == 1 or node_i == 3:
                G.add_edge(path[i], path[i+1], color='blue', arrowsize=40)  # Truck route
            elif node_i == 2:
                start = next((j for j in reversed(range(i+1)) if type_transport[j] == 3), None)
                if start:
                    G.add_edge(path[start], path[i+1], color='blue', arrowsize=40)  # Truck route
        elif node_next == 2:
            start = next((j for j in reversed(range(i+1)) if type_transport[j] == 3), None)
            end = next((k for k in range(i+1, len(type_transport)) if type_transport[k] == 3), None)
            if start != None and end != None:
                G.add_edge(path[start], path[i + 1], color='green', arrowsize=40, style='dashed')  # Drone route
                G.add_edge(path[i + 1], path[end], color='green', arrowsize=40, style='dashed')  # Drone route
            if node_i == 2:
                G.add_edge(path[i], path[i+1], color='green', arrowsize=40, style='dashed')  # Drone route
            # elif node_i == 4:
            #     G.add_edge(path[i - 1], path[i+1], color='green')  # Drone route
        elif node_next == 3:
            if node_i == 1 or node_i == 3:
                G.add_edge(path[i], path[i+1], color='blue', arrowsize=40)  # Truck route
            elif node_i == 2:
                start = next((j for j in reversed(range(i+1)) if type_transport[j] == 3), None)
                if start != None:
                    G.add_edge(path[start], path[i+1], color='blue', arrowsize=40)  # Truck route
        else:
            G.add_edge(path[i], path[i+1], color='red', arrowsize=40, style='dashed')  # Drone route
            G.add_edge(path[i+1], path[i], color='red', arrowsize=40, style='dashed')  # Drone route
            if type_transport[i + 2] == 3 or type_transport[i + 2] == 1:
                G.add_edge(path[i], path[i+2], color='blue', arrowsize=40)  # Truck route
            else:
                G.add_edge(path[i], path[i + 2], color='green', arrowsize=40, style='dashed')  # Drone route  
    # Set node positions using circular layout for better visualization
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=400)
    
    # Draw edges with respective colors and styles
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u, v in edges]
    edge_styles = ['dashed' if G[u][v]['color'] == 'green' else 'solid' for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, style=edge_styles, arrows=True)
    
    # Draw labels (node names)
    nx.draw_networkx_labels(G, pos)
    
    # Highlight depot node
    depot_node = path[0]
    nx.draw_networkx_nodes(G, pos, nodelist=[depot_node], node_size=400, node_color='red')
    
    # Show plot
    plt.title('Route Visualization')
    plt.show()

# Sample data
path = [0, 7, 10, 8, 9, 6, 17, 12, 3, 18, 2, 15, 16, 5, 19, 14, 4, 13, 11, 1, 0]
type_transport = [3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 4, 3, 3, 3, 3, 1, 3]
#   # Sample type of transportation
# Draw the route
draw_route(path, type_transport)
