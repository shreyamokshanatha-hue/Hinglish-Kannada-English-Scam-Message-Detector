import pandas as pd


# Load your original dataset
original = pd.read_csv("data/messages.csv")

# Load public dataset
public = pd.read_csv("data/hinglish_dataset.csv")


# Combine both datasets
combined = pd.concat(
    [original, public],
    ignore_index=True
)


# Create new IDs
combined["id"] = range(1, len(combined) + 1)


# Save combined dataset
combined.to_csv(
    "data/final_dataset.csv",
    index=False,
    encoding="utf-8"
)


# Display results
print("Datasets combined successfully!")

print("\nTotal messages:", len(combined))

print("\nLabel counts:")
print(combined["label"].value_counts())

print("\nSource counts:")
print(combined["source"].value_counts())

print("\nFinal columns:")
print(combined.columns)