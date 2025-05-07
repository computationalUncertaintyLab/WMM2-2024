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


if __name__ == "__main__":

    #--number of infectors
    I = pd.read_csv("./analysis_data/num_infectors.csv")
    T = pd.read_csv("./analysis_data/time_between_infections.csv")

    def ccdf(x):
        N = len(x)
        x, px = np.sort(x), 1.-np.arange(0,N)/N 
        return x,px

    plt.style.use("science")


    fig = plt.figure()
    gs = GridSpec(3, 2, height_ratios = [4,1,1])

    #-----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0,0])

    x,px = ccdf( I.num_infections.values )

    logx  = np.log10(x)
    logpx = np.log10(px)

    d   = pd.DataFrame({"logpx":logpx,"logx":logx})
    mod = smf.ols("logpx~logx", data = d)
    mod = mod.fit()

    lower, upper = mod.conf_int().loc["logx"]
    mle          = mod.params["logx"]
    
    
    ax.scatter( logx, logpx, s=10,facecolors='none', edgecolors='black' )
    b1,b0 = np.polyfit(logx,logpx,1)

    x0,x1 = 0,2 
    ax.plot( [x0,x1], [b0+x0*b1, b0+x1*b1], color="blue"   )
    
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["1","10","100"])
    ax.set_xlabel("Number of infections",fontsize=10,labelpad=0.01)


    ax.set_ylim(-3,0)
    ax.set_yticks([0,-1,-2,-3])
    ax.set_yticklabels(["1","1/10","1/100","1/1000"])
    ax.set_ylabel(r"$P(X>x)$",fontsize=10)

    ax.text( 0.05,0.05
             , s = r"$\tau$ = {:.1f} [{:.2f}, {:.2f}]".format(2-abs(mle),2-abs(lower),2-abs(upper))
             , ha="left"
             , va="bottom"
             , transform=ax.transAxes )


    Reff = np.mean( I["num_infections"] )
    ax.axvline( np.log10(Reff),0.625,1,   ls="--" )

    mn,mx = ax.get_ylim()
    
    ax.text( np.log10(Reff)*0.95
             ,-1 
             , s = r"$\text{R}_{\text{eff}}$" + "= {:.1f}".format(Reff)
             , ha="right",va="top"  )
    ax.text(0.95,0.95,s="A.",fontsize=10,fontweight="bold",ha="right",va="top",transform=ax.transAxes)
    
    #---------------------------------------------------------------------
    ax = fig.add_subplot(gs[0,1])
    
    x,px = ccdf( T.hours_between_infection.values)

    logx  = np.log10(x)
    logpx = np.log10(px)

    ax.scatter(x,logpx,s=10,facecolors='none', edgecolors='black')

    d   = pd.DataFrame({"logpx":logpx,"x":x})
    mod = smf.ols("logpx~x-1", data = d)
    mod = mod.fit()

    lower, upper = mod.conf_int().loc["x"]
    mle          = mod.params["x"]

    x0,x1 = 0, max(x) 
    ax.plot( [x0,x1], [x0*mle, x1*mle], color="green" )

    def exponential_rate_ci(samples, alpha=0.05):
        from scipy.stats import chi2
        n = len(samples)
        total = np.sum(samples)
        lam_hat = n / total
        q_lower = chi2.ppf(alpha/2, 2 * n)
        q_upper = chi2.ppf(1 - alpha/2, 2 * n)

        ci_lower = q_lower / (2 * total)
        ci_upper = q_upper / (2 * total)

        return lam_hat, (ci_lower, ci_upper)

    mle,(lower,upper) = exponential_rate_ci(T.hours_between_infection.values)
    
    
    ax.text( 0.05,0.05
             , s = r"$\alpha$ = {:.1f} [{:.2f}, {:.2f}]".format(mle*10**3,lower*10**3,upper*10**3) 
             , ha="left"
             , va="bottom"
             , transform=ax.transAxes )

    #ax.set_ylabel(r"$P(X>x)$")
    ax.set_ylabel("")
    
    ax.set_xlabel(r"Hours between infection",labelpad=0)

    ax.text(0.95,0.95,s="B.",fontsize=10,fontweight="bold",ha="right",va="top",transform=ax.transAxes)

    ax.set_ylim(-3,0)
    ax.set_yticks([0,-1,-2,-3])
    ax.set_yticklabels([])
    
    fig.set_size_inches( 8.5-2, 11./3 )
    fig.set_tight_layout(True)
    #plt.subplots_adjust(hspace=0.5)
    
    plt.savefig("./viz/epi_params_over_time/epi_distributions.pdf")
    plt.savefig("./viz/epi_params_over_time/epi_distributions.png", dpi=300)
    plt.close()

