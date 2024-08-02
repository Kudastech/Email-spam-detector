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

# Streamlit app title
st.title("Email Spam Detection")

# Text input for user message
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the preprocessed input
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict using the model
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
