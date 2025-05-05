#mcandrew

import sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

if __name__ == "__main__":

    d = pd.read_csv("./analysis_data/WMM2_contacts.csv")

    d["Timestamp"] = [ datetime.strptime(x,"%Y-%m-%dT%H:%M:%S.%f")  for x in d.Timestamp.values]

    add_row = pd.DataFrame({"Infector":["first"], "Infectee":["thm220"], "Timestamp":[ datetime.strptime("2024-09-17T18:00:00.000","%Y-%m-%dT%H:%M:%S.%f") ] })

    d = pd.concat([add_row,d])

    time_of_infection = d.loc[: , ["Infectee","Timestamp"] ]
    time_of_infection =  time_of_infection.rename(columns = {"Infectee":"Infectee_for_analysis", "Timestamp":"time_they_were_infected"})
    
    gen_interval      = d.merge( time_of_infection, left_on = ["Infector"], right_on = ["Infectee_for_analysis"] )
    gen_interval["delta_obj__time_between_infect"] = [ (row.Timestamp - row.time_they_were_infected)  for _,row in gen_interval.iterrows()  ]
    
    gen_interval["seconds_between_infection"]      = [ row.delta_obj__time_between_infect.total_seconds() for _,row in gen_interval.iterrows() ]
    gen_interval["hours_between_infection"]        = gen_interval.seconds_between_infection/3600
    gen_interval["days_between_infection"]         = gen_interval.hours_between_infection/24

    gen_interval = gen_interval.rename(columns = {"Timestamp":"time_they_infected_another"})

    gen_interval = gen_interval[ ["Infector"
                                  ,"Infectee"
                                  , "time_they_were_infected"
                                  , "time_they_infected_another"
                                  , "seconds_between_infection"
                                  , "hours_between_infection"
                                  ,"days_between_infection"] ]
    gen_interval.to_csv("./analysis_data/time_between_infections.csv", index=False)
