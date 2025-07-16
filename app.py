
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

# Input text
user_input = st.text_area("Enter your review:", "Type your movie review here...")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess and predict
        processed_input = preprocess(user_input)
        input_vector = vectorizer.transform([processed_input])
        prediction = model.predict(input_vector)[0]
        sentiment = 'positive' if prediction == 'positive' else 'negative'
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.write("Please enter a review.")
