import streamlit as st
from pages.page1 import page1  # Import the page1 function
from pages.chatbot import chatbot  # Import the chatbot page
from pages.page2 import page2  # Import the page2 function

# Set up the Streamlit application with multiple pages
if "page" not in st.session_state:
    st.session_state.page = "Document Processing"  # Default to "Document Processing" page

# Sidebar navigation for pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ("Document Processing", "Chatbot", "Learn More From Document"))

# Set the current page in session state
st.session_state.page = page

# Render the selected page
if st.session_state.page == "Document Processing":
    page1()  # Call the page1 function
elif st.session_state.page == "Chatbot":
    chatbot()  # Call the chatbot function
elif st.session_state.page == "Learn More From Document":
    page2()  # Call the page2 function
