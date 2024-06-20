import streamlit as st
from streamlit_player import st_player

st.title('Watermelon Meow Meow Host Site')
st.markdown('This website is designed to track infections of the fictious Watermelon Meow Meow outbreak within Lehigh University. If you have viewed this video, you are considered infected. Please include your Lehigh email below, as well as the Lehigh email of whoever showed this video to you.')

# URL of the YouTube video to embed
st_player('https://www.youtube.com/watch?v=ZSRfbByt4uk')

# Embed the YouTube video using an iframe
#st.markdown(f'<iframe width="560" height="315" src="{youtubeUrl}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', unsafe_allow_html=True)

infecteeEmail = st.text_input("Your Lehigh Email: ex. nep225@lehigh.edu", key = "infecteeEmail")
infectorEmail = st.text_input("Lehigh Email of the person who infected you: ex. thm220@lehigh.edu", key = "infectorEmail")

# Is this needed
if st.button("Submit"):
    if infecteeEmail and infectorEmail: # not null
        emails = [infecteeEmail, infectorEmail]
        
        # Write emails to a file
        with open("emails.txt", "a") as file:
            for email in emails:
                file.write(email + "\n")
        st.success("Emails have been stored successfully!")
