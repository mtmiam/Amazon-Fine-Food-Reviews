
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import sys, os
import re



model = open("rf_model.pk", "rb")
loaded_model = pickle.load(model)


vectorizer = open("tfidf_vectorizer.pk", "rb")
loaded_vect = pickle.load(vectorizer)

def stopwords(text):
    text = text.lower()
    stopwords=" ".join([ x for x in text.split() if x not in stop_words])
    return stopwords

def lemmetizer(text):
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(text)
    lemmatized = ' '.join([lemmatizer.lemmatize(w,'v') for w in word_list])
    return lemmatized
    
 
def raw_test(review, model, vectorizer):
    review_c = stopwords(lemmetizer(review))
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"



def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = review
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"
	



def main():
     

    html_temp = """
    <body style="background-color:powderblue;">
    <div style ="background-color: Lavender;padding:15px">
    <h1 style ="color:SteelBlue;text-align:center;">Amazon Fine Food Reviews</h1>
    </div>
    """
   
    st.markdown(html_temp, unsafe_allow_html = True)
      
    your_review = st.text_input("Enter your review")
    result =""
      
    if st.button("Analyze"):
        result = raw_test(your_review,loaded_model,loaded_vect)
    st.success('The review is {}'.format(result))




if __name__ =='__main__':
    main()
