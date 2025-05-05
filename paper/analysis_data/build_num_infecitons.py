#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    d = pd.read_csv("./analysis_data/WMM2_contacts.csv")

    def count_infectors(x):
        return pd.Series({"num_infections": len(x) })

    num_infectors = d.groupby(["Infector"]).apply(count_infectors,include_groups=False).reset_index()

    num_infectors.to_csv("./analysis_data/num_infectors.csv",index=False)
