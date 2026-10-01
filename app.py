import streamlit as st
import re
import joblib
from scipy.sparse import hstack

# Load trained model and vectorizers
model = joblib.load("models/hybrid_model.pkl")
word_vectorizer = joblib.load("models/word_vectorizer.pkl")
char_vectorizer = joblib.load("models/char_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s\u0C80-\u0CFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.set_page_config(
    page_title="Scam Message Detector",
    page_icon="🛡️"
)

st.title("🛡️ Hinglish/Kannada-English Scam Message Detector")

st.markdown(
    """
    ### 🔐 Protect yourself from suspicious messages
    This system uses **Machine Learning, Word TF-IDF and Character TF-IDF**
    to classify messages as **SCAM** or **NORMAL**.
    """
)
st.info(
    "💡 You can enter English, Hinglish, or Kannada-English messages. "
    "The model will analyze the message and predict whether it is likely to be a scam."
)
message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Nimma KYC expire agide, ee link click madi..."
)

if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        cleaned_message = clean_text(message)

        word_features = word_vectorizer.transform([cleaned_message])
        char_features = char_vectorizer.transform([cleaned_message])

        hybrid_features = hstack([
            word_features,
            char_features
        ])

        prediction = model.predict(hybrid_features)[0]
        probabilities = model.predict_proba(hybrid_features)[0]

        confidence = max(probabilities) * 100

        if prediction == 1:
         st.error("🚨 SCAM MESSAGE DETECTED")
         st.metric("Confidence", f"{confidence:.2f}%")

        else:
          st.success("✅ NORMAL MESSAGE")
          st.metric("Confidence", f"{confidence:.2f}%")
    st.markdown("---")

st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Model**")
    st.write("Logistic Regression")

with col2:
    st.write("**Features**")
    st.write("Word + Character TF-IDF")

with col3:
    st.write("**Test Accuracy**")
    st.write("97.43%")
st.caption(
    "⚠️ This tool provides a machine-learning prediction and should not be "
    "treated as a guaranteed confirmation of a scam."
)
st.markdown("---")

st.subheader("⚙️ How It Works")

st.write("1️⃣ **Message Input** — The user enters a message in English, Hinglish, or Kannada-English.")

st.write("2️⃣ **Feature Extraction** — The system converts the message into Word and Character TF-IDF features.")

st.write("3️⃣ **Machine Learning Prediction** — Logistic Regression analyzes the features and classifies the message as SCAM or NORMAL.")
st.markdown("---")
st.markdown("---")

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "97.43%")

with col2:
    st.metric("Scam Precision", "100%")

with col3:
    st.metric("Scam Recall", "87.5%")

with col4:
    st.metric("Scam F1-Score", "93%")
st.subheader("🔎 Confusion Matrix")

st.write("The model was evaluated on 778 test messages.")

cm_data = {
    "": ["Actual Normal", "Actual Scam"],
    "Predicted Normal": [618, 20],
    "Predicted Scam": [0, 140]
}

st.table(cm_data)
st.caption(
    "Hinglish/Kannada-English Scam Message Detector | "
    "Data Mining Project | Machine Learning"
)