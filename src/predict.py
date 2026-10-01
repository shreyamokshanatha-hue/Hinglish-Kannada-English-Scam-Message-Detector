import re
import joblib
from scipy.sparse import hstack


# Load trained model and vectorizers
model = joblib.load("models/hybrid_model.pkl")
word_vectorizer = joblib.load("models/word_vectorizer.pkl")
char_vectorizer = joblib.load("models/char_vectorizer.pkl")


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep English, numbers, Kannada characters and spaces
    text = re.sub(r"[^\w\s\u0C80-\u0CFF]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("===================================")
print(" HINGLISH/KANNADA SCAM DETECTOR")
print("===================================")

message = input("\nEnter your message: ")

cleaned_message = clean_text(message)


# Word TF-IDF
word_features = word_vectorizer.transform([cleaned_message])

# Character TF-IDF
char_features = char_vectorizer.transform([cleaned_message])

# Combine Word + Character features
hybrid_features = hstack([
    word_features,
    char_features
])


# Prediction
prediction = model.predict(hybrid_features)[0]

# Probability
probabilities = model.predict_proba(hybrid_features)[0]

confidence = max(probabilities) * 100


print("\nPrediction:", "SCAM" if prediction == 1 else "NORMAL")
print(f"Confidence: {confidence:.2f}%")