import streamlit as st
from transformers import pipeline

st.title("🚀 Sentiment Analysis App")

text = st.text_area("Enter some text")

if text:
    classifier = pipeline("sentiment-analysis")
    result = classifier(text)
    st.write("Prediction:", result[0]['label'])
    st.write("Confidence:", round(result[0]['score'], 2))
