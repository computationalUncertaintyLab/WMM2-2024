#mcandrew

import sys
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

import statsmodels.formula.api as smf

from datetime import datetime, timedelta

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

    #--number of infectors
    I = pd.read_csv("./analysis_data/num_infectors.csv")
    T = pd.read_csv("./analysis_data/time_between_infections.csv")

    def ccdf(x):
        N = len(x)
        x, px = np.sort(x), 1.-np.arange(0,N)/N 
        return x,px

    plt.style.use("science")

    fig, axs = plt.subplots(2,1)
    #-----------------------------------------------------------------------

    ax = axs[0]

    bet_centrality = nx.betweenness_centrality(g)
    node_sizes     = [100 * bet_centrality[n] for n in g.nodes()]
    node_colors    = [bet_centrality[n] for n in g.nodes()]
    
    nx.draw(  g
              , pos = from_node_to_location
              ,with_labels=False
              ,node_color=node_colors
              ,node_size=node_sizes
              , edge_color="gray"
              , width=0.5                       # Thin edges
              , arrows=True
              , connectionstyle="arc3,rad=0.15" # Curved edges
              , ax = ax
    )
    

    ax=axs[1]
    degree,centr = zip(*[( len(g[n]), bet_centrality[n]) for n in g.nodes()])

    degree = np.array(degree)
    centr  = np.array(centr)

    degree = degree[centr>0]
    centr  = centr[centr>0] 
    
    
    log_d, log_b = np.log10(degree+0.01), np.log10(centr+0.01)
    
    ax.scatter( log_d, log_b, s=10, color="black")

    ax.set_xlabel("Degree"                , fontsize=10)
    ax.set_ylabel("Betweeness Centrality" , fontsize=10)

    d   = pd.DataFrame({"y":log_b,"logx":log_d})
    mod = smf.ols("y~x", data = d)
    mod = mod.fit()
    
    lower, upper = mod.conf_int().loc["x"]
    mle          = mod.params["x"]

    b0,b1 = mod.params

    a,b = min(log_d), max(log_d)
    ax.plot([a,b], [b0+a*b1,b0+b*b1], color="blue")
    
    plt.show()
