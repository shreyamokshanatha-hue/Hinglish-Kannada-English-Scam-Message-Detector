import pandas as pd

# Load final dataset
data = pd.read_csv("data/final_dataset.csv")

# Display basic information
print("Total messages:", len(data))

print("\nLabel counts:")
print(data["label"].value_counts())

print("\nLanguage counts:")
print(data["language"].value_counts())

print("\nSource counts:")
print(data["source"].value_counts())

print("\nMissing values:")
print(data.isnull().sum())

print("\nFirst 5 messages:")
print(data[["message", "label", "language", "source"]].head())