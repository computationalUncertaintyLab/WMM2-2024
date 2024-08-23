#mcandrew,paras


import streamlit as st
from pages import infections, casesovertime, contactnetwork  # Import your pages
import boto3
import pandas as pd

if __name__ == "__main__":

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "👤 User Input", "🔗 Contact Network Infections", "📈 Cases Over Time"])

    # Home page content
    if page == "🏠 Home":
        st.markdown('''

        ## The Watermelon Meow Meow Outbreak
        * To infect, navigate to "User Input" (*most important page*)
        * To view contacts, navigate to "Contact network infections"
        * To view cases over time, navigate to "Cases over time"


        ### Research Consent
        This research study aims to generate a contact network of successful transmissions of the watermelon-meow-meow pathogen.
        We hypothesize that a student-driven, real-time outbreak of a **fictitious** pathogen will improve student comprehension of course material. 

        **What data is collected from me?**: We collect, and will present on this app, your Lehigh username (i.e. the three letter and three number combination from your Lehigh email address. For example, thm220).

        **How will my data be used?**: We plan to build a contact network over time that will present Lehigh usernames. This will be public and presented on the `Contact Network Infections' page.
        In addition, your connections to others will be used in class to simulate an outbreak to understand the mathematics of the spread of disease over a contact network. 


        **Contacts and Questions**: The Institutional Review Board (IRB) for the protection of human research participants at Lehigh University has reviewed and approved this study. If you have questions about the research study itself, please contact the Principal Investigator, Thomas McAndrew, at mcandrew@lehigh.edu. If you have questions about your rights or would simply like to speak with someone other than the research team about the questions or concerns, please contact the IRB at (610) 758-2871 or inirb@lehigh.edu. All reports or correspondence will be kept confidential.

        **Statement of Consent:** I have read the above information. I have had the opportunity to ask questions and have my questions answered. By adding my Lehigh Username, I consent to participate in this study.

    ''')



        AWS_S3_BUCKET = "wmm2-2024"
        AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        #--Attach dataset to session while they watch video
        if 'dataset' not in st.session_state:
            st.session_state.dataset = pd.read_csv(f"s3://{AWS_S3_BUCKET}/wmm_test.csv"
                                                   ,storage_options={"key"   : AWS_ACCESS_KEY_ID,"secret": AWS_SECRET_ACCESS_KEY})


    # Page routing
    elif page == "👤 User Input":
        infections.user_input_page()  # Call the function from infections.py
    elif page == "🔗 Contact Network Infections":
        contactnetwork.show_contact_network()  # Call the function from contactnetwork.py
    elif page == "📈 Cases Over Time":
        casesovertime.show_cases_over_time()  # Call the function from casesovertime.py
