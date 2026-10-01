import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion matrix values from our hybrid model
cm = np.array([
    [618, 0],
    [20, 140]
])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Scam"]
)

disp.plot()

plt.title("Hybrid Model - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

print("Confusion matrix graph saved successfully!")