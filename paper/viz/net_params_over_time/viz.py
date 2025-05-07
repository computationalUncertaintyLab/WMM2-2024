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

    #--number of infectors
    I = pd.read_csv("./analysis_data/num_infectors.csv")
    T = pd.read_csv("./analysis_data/time_between_infections.csv")

    def ccdf(x):
        N = len(x)
        x, px = np.sort(x), 1.-np.arange(0,N)/N 
        return x,px

    plt.style.use("science")

    fig, ax = plt.subplots()
    #-----------------------------------------------------------------------


    degree           = dict(g.degree())
    avg_neighbor_deg = nx.average_neighbor_degree(g)

    x = np.array([degree[n] for n in g.nodes()])
    y = np.array([avg_neighbor_deg[n] for n in g.nodes()])

    ax.scatter( np.log10(x), np.log10(y) )

    plt.show()
    
    r = nx.degree_assortativity_coefficient(G)











    

    #--Tau over time
    ax = fig.add_subplot(gs[1,:])

    def from_vals_to_tau(x):
        
        def count_infections(y):
            return pd.Series({"num_infections":len(y)})
        I = x.groupby(["Infector"]).apply(count_infections, include_groups=False).reset_index()
        
        x,px = ccdf( I.num_infections.values )

        logx  = np.log10(x)
        logpx = np.log10(px)

        d   = pd.DataFrame({"logpx":logpx,"logx":logx})
        mod = smf.ols("logpx~logx", data = d)
        mod = mod.fit()

        lower, upper = mod.conf_int().loc["logx"]
        mle          = mod.params["logx"]

        return pd.DataFrame({"mle":[2-abs(mle)],"lower":[2-abs(lower)],"upper":[2-abs(upper)]})

    first_time = T.time_they_were_infected.min()
    last_time  = T.time_they_were_infected.max()

    T["elapsed_time"]   = [ (datetime.strptime(x,"%Y-%m-%d %H:%M:%S")-datetime.strptime(first_time,"%Y-%m-%d %H:%M:%S")).total_seconds() for x in T.time_they_were_infected.values] 
    T["elapsed_hours"] = T.elapsed_time/3600
    T["elapsed_days"]  = T.elapsed_hours/24

    all_stats = pd.DataFrame()
    for time in np.arange(0.5,17,0.5):
        subset = T.loc[T.elapsed_days <= time]
        stats  = from_vals_to_tau(subset)
        stats["day"] = time
        
        all_stats = pd.concat([all_stats,stats])

    ax.plot(all_stats.day.values,all_stats.mle.values, color="blue")
    ax.fill_between(all_stats.day.values, all_stats.lower.values,all_stats.upper.values,alpha=0.30,color="blue")
        
    ax.axhline(float(stats.mle),color="black",ls="--")

    ax.text(0.95,0.95,s="C.",fontsize=10,fontweight="bold",ha="right",va="top",transform=ax.transAxes)

    ax.set_ylabel(r"$\tau$")
    ax.set_xlabel("")
    ax.set_xticks(np.arange(0,16+2,2))
    ax.set_xticklabels([])
    
    ax.set_ylim(1,1.5)


    #--time until infection over time 
    ax = fig.add_subplot(gs[2,:])

    def exponential_rate_ci(samples, alpha=0.05):
        from scipy.stats import chi2
        n = len(samples)
        total = np.sum(samples)
        lam_hat = n / total
        q_lower = chi2.ppf(alpha/2, 2 * n)
        q_upper = chi2.ppf(1 - alpha/2, 2 * n)

        ci_lower = q_lower / (2 * total)
        ci_upper = q_upper / (2 * total)

        return pd.DataFrame({"mle":[lam_hat],"lower":[ci_lower], "upper":[ci_upper]})

    all_stats = pd.DataFrame()
    for time in np.arange(0.5,17,0.5):
        subset = T.loc[T.elapsed_days <= time]
        stats  = exponential_rate_ci( subset.hours_between_infection.values)
        stats["day"] = time
        
        all_stats = pd.concat([all_stats,stats])

    ax.plot(all_stats.day.values,all_stats.mle.values, color="green")
    ax.fill_between(all_stats.day.values, all_stats.lower.values,all_stats.upper.values,alpha=0.30,color="green")
        
    ax.axhline(float(stats.mle),color="black",ls="--")

    ax.text(0.95,0.95,s="D.",fontsize=10,fontweight="bold",ha="right",va="top",transform=ax.transAxes)

    ax.set_ylabel(r"$\lambda$")
    ax.set_xlabel("Elapsed days since outbreak")
    ax.set_xticks(np.arange(0,16+2,2))

    #ax.set_ylim(1,1.5)
    
    #fig.set_tight_layout(True)
    fig.set_size_inches( 8.5-2, 11./3 )
    plt.subplots_adjust(hspace=0.5)
    
    plt.savefig("./viz/epi_params_over_time/epi_distributions.pdf")
    plt.savefig("./viz/epi_params_over_time/epi_distributions.png", dpi=300)
    plt.close()

