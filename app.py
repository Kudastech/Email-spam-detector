import streamlit as st
from welcome import main as welcome_main
from spam_detection import main as spam_detection_main

# Initialize the session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Navigation
if st.session_state.page == 'welcome':
    welcome_main()
elif st.session_state.page == 'spam_detection':
    spam_detection_main()
