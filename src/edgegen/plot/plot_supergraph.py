    
def get_supergraph_figure():
    # Plot networkx graph 'super_graph.pkl'
    import networkx as nx
    import matplotlib.pyplot as plt
    from edgegen.design_space.architectures.younger.younger_net import YoungerNet

    # Load super_graph.pkl
    graph = YoungerNet().super_graph

    # Create a new figure with specified size
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Generate positions for all nodes
    pos = nx.spring_layout(graph)
    
    # Draw the graph on the axes
    nx.draw(graph, pos, with_labels=False, node_size=10, font_size=8, ax=ax)
    
    # Return the figure object
    return fig
