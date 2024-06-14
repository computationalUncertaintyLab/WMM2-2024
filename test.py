#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from st_files_connection import FilesConnection

if __name__ == "__main__":

    # Create connection object and retrieve file contents.
    # Specify input format is a csv and to cache the result for 600 seconds.
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read("testbucket-jrieke/myfile.csv", input_format="csv", ttl=600)

    # Print results.
    for row in df.itertuples():
        st.write(f"{row.Owner} has a :{row.Pet}:")
