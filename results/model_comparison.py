import pandas as pd

data = {
    "Model": [
        "Word TF-IDF + Logistic Regression",
        "Character TF-IDF + Logistic Regression",
        "Hybrid Word + Character TF-IDF + Logistic Regression"
    ],
    "Accuracy": [
        96.53,
        95.24,
        97.43
    ]
}

df = pd.DataFrame(data)

print("\nMODEL COMPARISON")
print("================")
print(df.to_string(index=False))

df.to_csv("results/model_comparison.csv", index=False)

print("\nComparison saved successfully!")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.bar(df["Model"], df["Accuracy"])

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=20, ha="right")
plt.ylim(90, 100)

plt.tight_layout()

plt.savefig("results/model_accuracy_comparison.png", dpi=300)

print("Graph saved successfully!")