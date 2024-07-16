import sys
import numpy as np
import pandas as pd
import io
import os
import re

import streamlit as st
from st_files_connection import FilesConnection
from streamlit_player import st_player

import boto3
from datetime import datetime

# WMM webpage

st.title('Watermelon Meow Meow Host Site')
st.markdown('This website is designed to track infections of the fictious Watermelon Meow Meow outbreak within Lehigh University. If you have viewed this video, you are considered infected by the person who sent you this. Please include your Lehigh email below, as well as the Lehigh email of whoever showed this video to you. We appreciate your help!')

# URL of the YouTube video to embed
st_player('https://www.youtube.com/watch?v=ZSRfbByt4uk')

st.write('**When providing the two Lehigh emails below, please only include the characters and numbers preceding the @lehigh.edu**')
infecteeEmail = st.text_input("Your Lehigh Email credentials: ex. nep225", key = "infecteeEmail")
infectorEmail = st.text_input("Lehigh Email credentials of the person who infected you: ex. thm220", key = "infectorEmail")


# Boto3 stuff and button

AWS_S3_BUCKET = "wmm2-2024"
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# Button and s3 connection

if __name__ == "__main__":

    # Create connection object and retrieve file contents.
    # Specify input format is a csv and to cache the result for 600 seconds.
    conn = st.connection('s3', type=FilesConnection)
    df = pd.read_csv('s3://wmm2-2024/test1.csv', index_col=False)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    def validate_input(email):
    # Regex three chars and three numbers
        pattern1 = r'^[A-Za-z]{3}\d{3}$'
        # Regex two chars and two numbers
        pattern2 = r'^[A-Za-z]{2}\d{2}$'

        # Check if input_string matches either pattern
        if re.match(pattern1, email) or re.match(pattern2, email):
            return email
        else:
            # send back to try again
            st.markdown("Input format is invalid, please follow the format outlined above")
            return None

    if st.button('Submit', on_click=click_button):
        if infecteeEmail and infectorEmail: # not null
            current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            clean_infectee_email = validate_input(infecteeEmail)
            clean_infector_email = validate_input(infectorEmail)

            if clean_infectee_email and clean_infector_email:
                newdata = pd.DataFrame({
                    'Infectee': [clean_infectee_email],
                    'Infector': [clean_infector_email],
                    'Timestamp': [current_date_time]
                })[["Infectee", "Infector", "Timestamp"]]  # Ensuring the order of columns

                df = pd.concat([df, newdata], ignore_index=True) #concat only saves to the local dataframe not to the csv, need to write to csv
                df.to_csv('s3://wmm2-2024/test1.csv', mode='w', header=True, index=False)

                st.success("Thank you for submitting your information to WMM2. Emails have been stored successfully!")
                st.session_state_clicked = True

            else:
                st.session_state_clicked = False
        
        else:
                st.markdown("One or both of the fields is missing input, please be sure to enter the proper format.")
                st.session_state_clicked = False




