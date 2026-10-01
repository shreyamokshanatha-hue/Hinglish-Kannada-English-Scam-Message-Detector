import pandas as pd
import re

from sklearn.model_selection import train_test_split


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


print("Original message:")
print(data["message"].iloc[0])

print("\nCleaned message:")
print(data["clean_message"].iloc[0])


X = data["clean_message"]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTotal messages:", len(data))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining label counts:")
print(y_train.value_counts())

print("\nTesting label counts:")
print(y_test.value_counts())