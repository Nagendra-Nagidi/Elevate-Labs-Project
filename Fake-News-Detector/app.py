import streamlit as st
import joblib

vectorizer = joblib.load('vectorizer.jb')
LR = joblib.load('lr_model.jb')


st.title("Fake News Detector")
st.write("Enter the news headline to check if it's real or fake:")

news_input = st.text_input("News Headline")
if st.button("check news"):
    if news_input.strip():
        transform_input=vectorizer.transform([news_input])
        prediction = LR.predict(transform_input)

        if prediction[0] == 1:
            st.success("The news is real.")
        else:
            st.error("The news is fake.")
    else:
        st.warning("Please enter a valid news headline.")
        
