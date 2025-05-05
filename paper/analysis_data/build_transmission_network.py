#mcandrew

import sys
import numpy as np
import pandas as pd

import networkx as nx
from networkx.readwrite import json_graph
import json

if __name__ == "__main__":

    d = pd.read_csv("./analysis_data/WMM2_contacts.csv")

    g = nx.from_pandas_edgelist(df=d, source="Infector", target = "Infectee")

    data = json_graph.node_link_data(g)
    with open("./analysis_data/WMM2-2024.json", "w") as f:
        json.dump(data, f)
   
