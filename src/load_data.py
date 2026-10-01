import pandas as pd

# Load the dataset
data = pd.read_csv("data/messages.csv")

# Display the first few messages
print("First 5 messages:")
print(data.head())

# Display total number of messages
print("\nTotal messages:", len(data))

# Display scam and normal message counts
print("\nLabel counts:")
print(data["label"].value_counts())

# Display column names
print("\nColumns:")
print(data.columns)