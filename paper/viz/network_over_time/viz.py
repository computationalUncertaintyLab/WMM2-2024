#mcandrew

import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import json
from networkx.readwrite import json_graph
import networkx as nx

if __name__ == "__main__":

    with open("./analysis_data/WMM2-2024.json") as f:
        data = json.load(f)
    g = json_graph.node_link_graph(data)

    with open("./viz/network_over_time/node_locations.json") as f:
        from_node_to_location = json.load(f)
    from_node_to_location = { n:(att["x"],att["y"]) for n,att in from_node_to_location.items()}


    gen_interval = pd.read_csv("./analysis_data/time_between_infections.csv")
    
    fig,axs = plt.subplots(1,3)

    ax = axs[0]

    time_1 = gen_interval.loc[ gen_interval.time_they_infected_another<="2024-09-18", ["Infector", "Infectee" ] ]
    time_1_nodes = set(time_1.Infector.unique()) | set(time_1.Infectee.unique())

    
    G_sub  = g.subgraph(time_1_nodes).copy()
    nx.draw(  G_sub
              , pos = from_node_to_location
              ,with_labels=False
              ,node_color="black"
              , node_size=10                    # Small nodes
              ,edge_color="gray"
              , width=0.5                       # Thin edges
              , arrows=True
              , connectionstyle="arc3,rad=0.15" # Curved edges
              , ax = ax
    )
    ax.set_title("Infections up until 9/18",fontdict={"family": "Arial", "size": 11})

    
    ax = axs[1]

    time_2 = gen_interval.loc[ gen_interval.time_they_infected_another<="2024-09-25", ["Infector", "Infectee" ] ]
    time_2_nodes = set(time_2.Infector.unique()) | set(time_2.Infectee.unique())

    G_sub  = g.subgraph(time_2_nodes).copy()
    nx.draw(  G_sub
              , pos = from_node_to_location
              ,with_labels=False
              ,node_color="black"
              , node_size=10                    # Small nodes
              ,edge_color="gray"
              , width=0.5                       # Thin edges
              , arrows=True
              , connectionstyle="arc3,rad=0.15" # Curved edges
              , ax = ax
    )
    ax.set_title("Infections up until 9/25",fontdict={"family": "Arial", "size": 11})

    ax = axs[2]

    nx.draw(  g
              , pos = from_node_to_location
              ,with_labels=False
              ,node_color="black"
              , node_size=10                    # Small nodes
              ,edge_color="gray"
              , width=0.5                       # Thin edges
              , arrows=True
              , connectionstyle="arc3,rad=0.15" # Curved edges
              , ax = ax
    )
    ax.set_title("All Infections",fontdict={"family": "Arial", "size": 11})
    

    fig.set_size_inches( 6.5, (11-2)/3 )
    fig.set_tight_layout(True)
    plt.savefig("./viz/network_over_time/transmission_over_time.pdf")
    plt.savefig("./viz/network_over_time/transmission_over_time.png")
    plt.close()
