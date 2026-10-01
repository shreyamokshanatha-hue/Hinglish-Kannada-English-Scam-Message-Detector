import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack


data = pd.read_csv("data/final_dataset.csv")


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep English, numbers, Kannada characters and spaces
    text = re.sub(r"[^\w\s\u0C80-\u0CFF]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


data["clean_message"] = data["message"].apply(clean_text)

X = data["clean_message"]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Total messages:", len(data))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# Word TF-IDF
word_vectorizer = TfidfVectorizer()

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)


# Character TF-IDF
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)


print("\nWord TF-IDF features:", X_train_word.shape[1])
print("Character TF-IDF features:", X_train_char.shape[1])


# Combine Word + Character features
X_train_hybrid = hstack([X_train_word, X_train_char])
X_test_hybrid = hstack([X_test_word, X_test_char])


print("Total Hybrid features:", X_train_hybrid.shape[1])


# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_hybrid, y_train)


# Prediction
y_pred = model.predict(X_test_hybrid)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, "models/hybrid_model.pkl")
joblib.dump(word_vectorizer, "models/word_vectorizer.pkl")
joblib.dump(char_vectorizer, "models/char_vectorizer.pkl")

print("\nModel and vectorizers saved successfully!")