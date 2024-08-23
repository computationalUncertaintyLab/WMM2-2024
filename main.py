import streamlit as st
from pages import infections, casesovertime, contactnetwork  # Import your pages

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "👤 User Input", "🔗 Contact Network Infections", "📈 Cases Over Time"])

# Home page content
if page == "🏠 Home":
    st.title("Welcome to the Watermelon Meow Meow Tracking App!")
    st.markdown("This app is designed to track and visualize the spread of the fictitious Watermelon Meow Meow outbreak within Lehigh University.")
    st.markdown("### Use the sidebar to navigate to different sections of the app:")
    st.markdown("- **User Input:** Record new infections.")
    st.markdown("- **Contact Network Infections:** Visualize the contact network of infections.")
    st.markdown("- **Cases Over Time:** View the trend of cases over time.")

# Page routing
elif page == "👤 User Input":
    infections.user_input_page()  # Call the function from infections.py
elif page == "🔗 Contact Network Infections":
    contactnetwork.show_contact_network()  # Call the function from contactnetwork.py
elif page == "📈 Cases Over Time":
    casesovertime.show_cases_over_time()  # Call the function from casesovertime.py
