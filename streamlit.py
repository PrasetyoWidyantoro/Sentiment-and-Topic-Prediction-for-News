import pandas as pd
from joblib import load
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from typing import Optional
import pickle
# Load stopwords for Indonesian
nltk.download('punkt')
nltk.download('stopwords')
stopwords_indonesian = set(nltk.corpus.stopwords.words('indonesian'))

# Function to combine values from specific columns into a single string
def combine_columns(row):
    # Convert each element to string if not already a string
    row = [str(elem) for elem in row]
    return ' '.join(row)

# Function to clean and preprocess text
def preprocess_text(df):
    # Select or extract desired columns
    df = df[['Content']]
    
    # Check if optional columns exist in DataFrame
    if 'SectionName' in df.columns:
        df['SectionName'] = df['SectionName'].fillna('')
    if 'Tag' in df.columns:
        df['Tag'] = df['Tag'].fillna('')
    if 'SiteName' in df.columns:
        df['SiteName'] = df['SiteName'].fillna('')
    
    # Combine 'SiteName', 'SectionName', 'Tag', and 'Content' columns
    df['Content'] = df.apply(lambda row: combine_columns(row), axis=1)
    
    # Remove HTML tags
    df['Content'] = df['Content'].apply(lambda x: re.sub(r'<.*?>', '', x))
    
    # Remove special characters, including 'rdquo', and convert to lowercase
    df['Content'] = df['Content'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).replace('rdquo', '').lower())
    
    # Tokenize the text
    df['Content'] = df['Content'].apply(word_tokenize)
    
    # Remove stopwords
    df['Content'] = df['Content'].apply(lambda tokens: [word for word in tokens if word not in stopwords_indonesian])
    
    # Join the tokens with a single space
    df['Content'] = df['Content'].apply(lambda tokens: ' '.join(tokens))
    
    return df[['Content']]



# Load TfidfVectorizer from file
def load_tfidf_vectorizer(file_path):
    with open(file_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_vectorizer

# Load TfidfVectorizer
tfidf_vectorizer = load_tfidf_vectorizer('tfidf_vectorizer_model.pkl')

# Load final models
model_sentiment = joblib.load("lightgbm_ori_sentiment.pkl")
model_topic = joblib.load("best_logistic_regression_topic.pkl")

# Function to predict sentiment and topic
def predict_sentiment_and_topic(data):
    # Preprocess text
    df = pd.DataFrame(data, index=[0])
    df = preprocess_text(df)
    
    # Transform data using TF-IDF vectorizer
    df_tfidf = tfidf_vectorizer.transform(df['Content'])
    
    # Predict topic classification
    topic_prediction = model_topic.predict(df_tfidf)[0]

    # Predict sentiment analysis
    sentiment_prediction = model_sentiment.predict(df_tfidf)[0]
    
    # Mapping predictions to the appropriate labels
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    topic_mapping = {
        0: "Cuaca dan Lingkungan",
        1: "Kendaraan dan Transportasi",
        2: "Mode, Gaya, Kebudayaan, dan Pariwisata",
        3: "Pemerintahan dan Program",
        4: "Kesehatan dan Gizi",
        5: "Keuangan dan Teknologi",
        6: "Kejahatan, Kriminalitas dan Ketertiban Umum",
        7: "Politik dan Olahraga"
    }
    
    # Store predictions in the desired format
    output = {
        "sentiment_prediction": sentiment_mapping[sentiment_prediction],
        "topic_prediction": topic_mapping[topic_prediction]
    }
    
    return output

# Streamlit
import streamlit as st

st.title("Sentiment and Topic Prediction for News")

st.write("The Sentiment and Topic Prediction for News application allows users to predict sentiment and topic for news articles. Users can input news details via a form or upload a CSV file containing news data. The application then utilizes a prediction function to analyze the input data and provide predictions for sentiment and topic. Results are displayed to the user, allowing for quick analysis and understanding of the news content.")

# Instructions for downloading CSV file
st.write("If you want to try the 'Select Input Method CSV' feature, you can directly download the sample CSV file by clicking the link below:")
# Short link to the CSV file
csv_file_link = "https://drive.google.com/uc?export=download&id=14mQe8rBzHh4M-rtRJC43IfWgQAFMqBKW"
# Create a hyperlink with the download attribute
st.markdown(f'<a href="{csv_file_link}" download="sample.csv">Click here to download the CSV file</a>', unsafe_allow_html=True)

st.write("Choose an input method to enter news details.")

input_method = st.radio("Select Input Method", ("Form", "CSV"))

if input_method == "Form":
    # Creating input fields for the news details
    content = st.text_area("Content")
    url = st.text_input("URL (Optional)")
    site_id = st.number_input("Site ID (Optional)", value=0)
    site_name = st.text_input("Site Name (Optional)")
    section_id = st.number_input("Section ID (Optional)", value=0)
    section_name = st.text_input("Section Name (Optional)")
    published_by = st.text_input("Published By (Optional)")
    sup_title = st.text_input("Sup Title (Optional)")
    title = st.text_input("Title (Optional)")
    sub_title = st.text_input("Sub Title (Optional)")
    description = st.text_input("Description (Optional)")
    author_name = st.text_input("Author Name (Optional)")
    author_id = st.number_input("Author ID (Optional)", value=0)
    photo = st.text_input("Photo (Optional)")
    source_name = st.text_input("Source Name (Optional)")
    video = st.text_input("Video (Optional)")
    embed_social = st.text_input("Embed Social (Optional)")
    tag = st.text_input("Tag (Optional)")
    lipsus = st.text_input("Lipsus (Optional)")
    related = st.text_input("Related (Optional)")
    keyword = st.text_input("Keyword (Optional)")
    url_short = st.text_input("Short URL (Optional)")
    published_date = st.text_input("Published Date (Optional)")

    # When the button is clicked, perform prediction
    if st.button("Predict"):
        # Checking if the content field is not empty
        if content.strip() != "":
            # Creating the data dictionary
            data = {
                "Content": content
            }
            # Calling the function to make predictions
            result = predict_sentiment_and_topic(data)
            # Displaying the prediction results
            st.write("Prediction Results:")
            st.write("Sentiment:", result["sentiment_prediction"])
            st.write("Topic:", result["topic_prediction"])
        else:
            st.warning("Please enter the content first!")

else:  # Input from CSV
    # CSV file input in the sidebar
    st.sidebar.title("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    # Perform prediction from CSV file
    if uploaded_file is not None:
        # Read CSV file
        data = pd.read_csv(uploaded_file)
        # Check if 'content' column exists
        if 'Content' not in data.columns:
            st.error("CSV file must have a 'content' column.")
        else:
            # Predict for each row in CSV
            for idx, row in data.iterrows():
                st.write(f"Predictions for Row {idx+1}:")
                content = row['Content']  # Get the 'content' from the row
                result = predict_sentiment_and_topic({"Content": content})  # Pass content as a dictionary
                st.write("Sentiment:", result["sentiment_prediction"])
                st.write("Topic:", result["topic_prediction"])
