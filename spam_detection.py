import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Stem tokens
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load pre-trained TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def main():
    st.title("Email Spam Detection")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        transformed_sms = transform_text(input_sms)
        # st.write("Transformed input:", transformed_sms)

        vector_input = tfidf.transform([transformed_sms])
        # st.write("Vectorized input shape:", vector_input.shape)

        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

if __name__ == "__main__":
    main()
