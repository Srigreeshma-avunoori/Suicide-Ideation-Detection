import streamlit as st
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pickle

# Download NLTK resources
nltk.download('stopwords')

# Define the preprocessing functions
def preprocess_tweet(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text + ' '.join(emoticons).replace('-', '')
    return text

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def custom_tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower())
    text += ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in tokenizer_porter(text) if w not in stop]
    return tokenized

# Load your pre-trained model
def load_model():
    with open('suicidal_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
# Load NLTK stopwords
stop = stopwords.words('english')

# Define and initialize the HashingVectorizer (vect)
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=custom_tokenizer)

# Streamlit app
def main():
    st.title('Suicidal Tendency Detection')

    # Load your pre-trained model
    model = load_model()

    # Input text area for user input
    user_input = st.text_area("Enter a sentence:", "Type Here")

    if st.button("Analyze"):
        # Preprocess the user input
        processed_input = preprocess_tweet(user_input)

        # Vectorize the input
        vectorized_input = vect.transform([processed_input])

        # Make prediction
        prediction = model.predict(vectorized_input)
        #probability = model.predict_proba(vectorized_input)

        # Show prediction result
        if prediction[0] == 0:
            st.write("Prediction: Negative")
        else:
            st.write("Prediction: Positive")

        #st.write(f"Probability: {np.max(probability)*100:.2f}%")

if __name__ == '__main__':
    main()
