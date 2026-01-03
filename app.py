# =========================================
# SMS SPAM FILTER - STREAMLIT UI
# =========================================

import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from transformers import pipeline

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="SMS Spam Filter",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Filter (AI Integrated)")
st.write("Classifies SMS messages as **SPAM** or **HAM**")

# -----------------------------------------
# Load Dataset
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

# -----------------------------------------
# Text Cleaning Function
# -----------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)
y = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------------------
# Tokenization (TF-IDF)
# -----------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_message'])

# -----------------------------------------
# Train Baseline Models
# -----------------------------------------
nb_model = MultinomialNB()
nb_model.fit(X, y)

svm_model = LinearSVC()
svm_model.fit(X, y)

# -----------------------------------------
# Rule-Based Spam Detection
# -----------------------------------------
spam_keywords = ['free', 'win', 'prize', 'offer', 'click', 'urgent']

def has_url(text):
    return bool(re.search(r'http|www|\.com', text.lower()))

def has_profanity(text):
    return any(word in text.lower() for word in spam_keywords)

# -----------------------------------------
# Load AI Model (Optional)
# -----------------------------------------
@st.cache_resource
def load_ai_model():
    return pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
    )

ai_model = load_ai_model()

def ai_predict(text):
    result = ai_model(text)[0]
    return 1 if result['label'].lower() == 'spam' else 0

# -----------------------------------------
# Final Prediction Logic
# -----------------------------------------
def predict_sms(text):
    # Rule-based check
    if has_url(text) or has_profanity(text):
        return "SPAM"

    clean = clean_text(text)
    vector = vectorizer.transform([clean])

    svm_pred = svm_model.predict(vector)[0]
    ai_pred = ai_predict(text)

    if svm_pred == 1 or ai_pred == 1:
        return "SPAM"
    return "HAM"

# -----------------------------------------
# UI: Single SMS Prediction
# -----------------------------------------
st.subheader("🔍 Test a Single SMS")

sms_input = st.text_area(
    "Enter SMS message",
    height=100,
    placeholder="Type your SMS here..."
)

if st.button("Check Message"):
    if sms_input.strip() == "":
        st.warning("Please enter an SMS message.")
    else:
        result = predict_sms(sms_input)
        if result == "SPAM":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Safe)")

# -----------------------------------------
# UI: CSV Upload for Bulk Prediction
# -----------------------------------------
st.subheader("📂 Bulk SMS Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV file with a 'message' column",
    type=["csv"]
)

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    if 'message' not in input_df.columns:
        st.error("CSV file must contain a 'message' column.")
    else:
        input_df['prediction'] = input_df['message'].apply(predict_sms)
        st.success("Prediction completed!")

        st.dataframe(input_df.head())

        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name="sms_predictions.csv",
            mime="text/csv"
        )

# -----------------------------------------
# Footer
# -----------------------------------------
st.markdown("---")
st.markdown(
    "**Tech Stack:** Python, scikit-learn, regex, Streamlit, Hugging Face"
)
