#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

if __name__ == "__main__":

    time_series_hour = pd.read_csv("./analysis_data/time_series_hourscale.csv")
    yhats            = pickle.load(open("./viz/present_compartmental_wont_capture/yhats.pkl","rb"))
    samples          = pickle.load(open("./viz/present_compartmental_wont_capture/samples.pkl","rb"))

    week_day = time_series_hour[ ["week","dow"] ].drop_duplicates()
    sat_suns = [0 if row.dow in [5,6] else 1 for _,row in week_day.iterrows()]

    fig = plt.figure(constrained_layout=True)
    gs  = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, :])
    ax.vlines( time_series_hour.elpased_hour, 0., time_series_hour.inc, color = "black" )

    lower,middle,upper = np.percentile( yhats[:,0,:], [2.5,50,97.5], axis=0)

    nweeks         = int(np.floor(len(time_series_hour)/len(middle)))
    residual_week  = int(len(time_series_hour) - nweeks*len(middle))

    middle_r = list( np.tile(middle,nweeks)*np.repeat(sat_suns, 24) ) + list( middle[:residual_week] )
    lower_r  = list(np.tile(lower,nweeks)*np.repeat(sat_suns, 24)   ) + list( lower[:residual_week] )
    upper_r  = list(np.tile(upper,nweeks)*np.repeat(sat_suns, 24)   ) + list( upper[:residual_week] )

    
    ax.plot(time_series_hour.elpased_hour , middle_r, lw=2,color="blue"  )
    ax.fill_between(time_series_hour.elpased_hour,lower_r,upper_r,color="blue",alpha=0.50)

    ax      = fig.add_subplot(gs[1, 0])
    twin_ax = plt.twinx(ax)
    
    boxplot = pd.DataFrame({"repo": 5*(samples["repo"]+1), "sus":8000*samples["sus"], "I0": 5*samples["I0"]   })

    boxplot_long = boxplot.melt()
    sns.boxplot( x = "variable", y = "value", ax=ax, data = boxplot_long.loc[boxplot_long.variable=="sus"])
    sns.boxplot( x = "variable", y = "value", ax=twin_ax, data = boxplot_long.loc[boxplot_long.variable!="sus"] )

    # Move twin_ax to the left side
    twin_ax.yaxis.set_label_position('left')
    twin_ax.yaxis.tick_left()
    twin_ax.spines['left'].set_position(('outward', 30))
    twin_ax.spines['left'].set_visible(True)

    ax      = fig.add_subplot(gs[1, 1])

    peak_time      = np.argmax(yhats[:,0,:],1)
    peak_intensity = np.max(yhats[:,0,:],1)

    peak_data = pd.DataFrame({"peak_time":peak_time, "peak_intensity":np.log10(peak_intensity)})

    sns.kdeplot( x="peak_time", y= "peak_intensity", data = peak_data, fill=True, ax=ax )

    analysis_data = time_series_hour.loc[~time_series_hour.dow.isin([5,6])]
    y = pd.pivot_table( index=["week","dow"], columns = ["hour"], values = ["inc"], data = analysis_data )
    y.columns = [y for x,y in y.columns ]

    #change the below back to zero.
    y         = y[ list(np.arange(0,24)) ]
    y         = y.to_numpy()


    peak_time_obs      = np.argmax(y,1)
    peak_intensity_obs = np.max(y,1)
    ax.scatter( peak_time_obs, np.log10(peak_intensity_obs), s=7, color="k" ) 
    
    plt.show()

    



    


    
    











        
    
 
