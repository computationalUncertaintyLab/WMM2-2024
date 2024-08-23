
import streamlit as st
import pandas as pd
import re
from streamlit_player import st_player
from datetime import datetime, timedelta
import boto3
from pages import contactnetwork, casesovertime  # Import your pages

def show_infections_page():

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
        pattern1 = r'^[A-Za-z]{3}\d{3}$'
        pattern2 = r'^[A-Za-z]{2}\d{2}$'
        return re.match(pattern1, email) or re.match(pattern2, email)

    def user_input_page():
        # WMM webpage
        st.title('Watermelon Meow Meow Host Site')
        st.markdown('This website is designed to track infections of the fictitious Watermelon Meow Meow outbreak within Lehigh University. Please include your Lehigh email below, as well as the Lehigh email of whoever showed this video to you. We appreciate your help!')

        # URL of the YouTube video to embed
        st_player('https://www.youtube.com/watch?v=ZSRfbByt4uk')

        st.write('**When providing the two Lehigh emails below, please only include the characters and numbers preceding the @lehigh.edu**')
        infecteeEmail = st.text_input("Your Lehigh Email credentials (the person who watched the video): ex. nep225", key="infecteeEmail")
        infectorEmail = st.text_input("Lehigh Email credentials of the person who infected you (the person who sent you the video): ex. thm220", key="infectorEmail")
        st.markdown('By pressing submit, you are aware and consent that your Lehigh email will appear on this public website.')

        # Initialize session state to track page load time
        if 'page_load_time' not in st.session_state:
            st.session_state.page_load_time = datetime.now()
            st.session_state.submit_enabled = False

        #--Attach dataset to session while they watch video
        st.session_state.dataset = pd.read_csv(f"s3://{AWS_S3_BUCKET}/wmm_test.csv"
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
                            st.session_state.dataset.to_csv(f"s3://{AWS_S3_BUCKET}/wmm_test.csv"
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
    show_infections_page()

