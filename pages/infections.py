
import streamlit as st
import pandas as pd
import re
from streamlit_player import st_player
from datetime import datetime, timedelta
import boto3
from pages import contactnetwork, casesovertime  # Import your pages


# Boto3 configuration
AWS_S3_BUCKET = "wmm2-2024"
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Email validation function
def validate_input(email):
    pattern = r'^[A-Za-z]+\d+$'
    pattern2 = r'^[A-Za-z]+\d+[A-Za-z]+$'
    return re.match(pattern, email) or re.match(pattern2, email)

def user_input_page():
    # WMM webpage
    st.title('Add an infectious event')
    st.markdown('''Include your Lehigh username in the box titled **infector** and include the Lehigh username of the person that you infected in the **infectee** box.
    A Lehigh username is the letters and numbers before @lehigh.edu. For example, the Lehigh username for thm220@lehigh.edu is thm220. 
    
''')
    
    # URL of the YouTube video to embed
    st_player('https://www.youtube.com/watch?v=ZSRfbByt4uk')

    infecteeEmail = st.text_input("Infectee (i was infected)", key="infecteeEmail", placeholder = "ABC123", help = "Put your username here if you were infected")
    infectorEmail = st.text_input("Infector (i am the one who infected)", key="infectorEmail", placeholder = "ABC123", help = "Put your username here if you infected someone ")
    st.markdown('By pressing submit, you are aware and consent that your Lehigh username will appear on this public website.')
    infecteeEmail = infecteeEmail.lower().strip()
    infectorEmail = infectorEmail.lower().strip()

    # Initialize session state to track page load time
    if 'page_load_time' not in st.session_state:
        st.session_state.page_load_time = datetime.now()
        st.session_state.submit_enabled = False

    #--Attach dataset to session while they watch video
    st.session_state.dataset = pd.read_csv(f"s3://{AWS_S3_BUCKET}/wmm_live.csv"
                                            ,storage_options={
                                                "key"   : AWS_ACCESS_KEY_ID,
                                                "secret": AWS_SECRET_ACCESS_KEY,
                                            }
                                           )

    # Calculate the elapsed time since the page was loaded
    elapsed_time = datetime.now() - st.session_state.page_load_time

    # Enable submit button after 30 seconds
    if elapsed_time > timedelta(seconds=30):
        st.session_state.submit_enabled = True

    # Main logic
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    if st.session_state.submit_enabled:
        if st.button('Submit', on_click=click_button):
            if infecteeEmail and infectorEmail:  # Check if not null
                if infecteeEmail == infectorEmail:
                    st.error("The infectee and infector emails cannot be the same. Please enter different emails.")
                else:
                    valid_infectee = validate_input(infecteeEmail)
                    valid_infector = validate_input(infectorEmail)

                    if valid_infectee and valid_infector:
                        # Check if the infector is eligible to infect others (must have been an infectee or be 'thm220')
                        if infectorEmail != 'thm220' and 'dataset' in st.session_state:
                            df = st.session_state.dataset
                            if infectorEmail not in df['Infectee'].values:
                                st.error(f"{infectorEmail} is not eligible to infect others as they have not been infected yet.")
                                return

                        # Check if the infectee has already been infected
                        if 'dataset' in st.session_state and not st.session_state.dataset.empty:
                            df = st.session_state.dataset
                            if infecteeEmail in df['Infectee'].values:
                                st.warning(f"{infecteeEmail} has already been infected in this game. Get out there and infect more people!")
                                return
                        else:
                            df = pd.DataFrame(columns=['Infector', 'Infectee', 'Timestamp'])

                        # Add the new infection record
                        current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        newdata = pd.DataFrame({
                            'Infectee': [infecteeEmail],
                            'Infector': [infectorEmail],
                            'Timestamp': [current_date_time]
                        })[["Infectee", "Infector", "Timestamp"]]  # Ensuring the order of columns

                        df = pd.concat([df, newdata], ignore_index=True)

                        # Convert 'Timestamp' to datetime
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

                        # Create a 'Date' column for easier visualization
                        df['Date'] = df['Timestamp'].dt.date

                        st.session_state.dataset = df

                        #--write out
                        st.session_state.dataset.to_csv(f"s3://{AWS_S3_BUCKET}/wmm_live.csv"
                                                        , index=False
                                                        , storage_options={
                                                            "key"   : AWS_ACCESS_KEY_ID,
                                                            "secret": AWS_SECRET_ACCESS_KEY,
                                                        },
                                                    )

                        st.success("Thank you for submitting your information to WMM2. Emails have been stored successfully!")
                    else:
                        if not valid_infectee:
                            st.error("Invalid input for your Lehigh Email credentials. Please follow the specified format.")
                        if not valid_infector:
                            st.error("Invalid input for the infector's Lehigh Email credentials. Please follow the specified format.")
            else:
                st.error("One or both of the fields is missing input. Please ensure both emails are entered correctly.")
    else:
        st.warning("Please wait for 30 seconds after the page loads to submit.")
        st.button('Submit', disabled=True)

if __name__ == "__main__":
    user_input_page()

