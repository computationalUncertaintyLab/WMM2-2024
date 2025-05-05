#mcandrew

import sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

if __name__ == "__main__":

    d = pd.read_csv("./data/WMM2-2024.csv")
    d = d[d.Infectee!="thm220"]

    #--NEP225 is developer. Assuming infected NEP at same time of infection plus one minute
    d.loc[ (d.Infector == "thm220") & (d.Infectee=="nep225"), "Timestamp" ] =  "2024-09-17T18:01:00.000"
    
    d = d.to_csv("./analysis_data/WMM2_contacts.csv", index=False)
