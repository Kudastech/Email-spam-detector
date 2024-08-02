import streamlit as st

def main():
    st.title("The Federal Polytechnic Ilaro")
    st.title("H/CTE/23/0789")
    # st.title("Welcome to the Spam Detection App")
    # st.title("Welcome to the Spam Detection App")
    st.write("This app helps you detect whether a message is spam or not.")

    if st.button('Go to Spam Detection'):
        st.session_state.page = 'spam_detection'

if __name__ == "__main__":
    main()
