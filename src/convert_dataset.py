import json
import pandas as pd


# Open downloaded dataset
with open("data/data.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)


# Convert rows into DataFrame
df = pd.DataFrame(dataset["rows"])


# Rename text column to message
df = df.rename(columns={
    "text": "message"
})


# Add project columns
df["scam_type"] = "Unknown"
df["language"] = "Hinglish/English"
df["source"] = "Public Dataset"


# Create ID
df.insert(0, "id", range(1, len(df) + 1))


# Keep required columns
df = df[
    [
        "id",
        "message",
        "label",
        "scam_type",
        "language",
        "source"
    ]
]


# Save converted dataset
df.to_csv(
    "data/hinglish_dataset.csv",
    index=False,
    encoding="utf-8"
)


# Display results
print("Dataset converted successfully!")

print("\nTotal messages:", len(df))

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nColumns:")
print(df.columns)