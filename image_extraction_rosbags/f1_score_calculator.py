import torch
import matplotlib.pyplot as plt

# Provided tensors
precision = torch.tensor([
    0.0952, 0.3830, 0.3913, 0.4000, 0.4091, 0.4186, 0.4286, 0.4390, 0.4500,
    0.4615, 0.4737, 0.4865, 0.5000, 0.5143, 0.5294, 0.5455, 0.5625, 0.5806,
    0.6000, 0.6207, 0.6429, 0.6667, 0.6923, 0.7200, 0.7500, 0.7826, 0.8182,
    0.8571, 0.9000, 0.9474, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000
])

recall = torch.tensor([
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 0.9444, 0.8889, 0.8333, 0.7778, 0.7222,
    0.6667, 0.6111, 0.5556, 0.5000, 0.4444, 0.3889, 0.3333, 0.2778, 0.2222,
    0.1667, 0.1111, 0.0556, 0.0000
])

thresholds = torch.tensor([
    0.0000, 0.0012, 0.0014, 0.0024, 0.0038, 0.0058, 0.0068, 0.0130, 0.0186,
    0.0239, 0.0261, 0.0299, 0.0300, 0.0314, 0.0343, 0.0345, 0.0350, 0.0351,
    0.0381, 0.0407, 0.0473, 0.0481, 0.0494, 0.0499, 0.0535, 0.0536, 0.0553,
    0.0677, 0.1518, 0.1548, 0.4967, 0.5745, 0.5791, 0.6067, 0.6804, 0.7052,
    0.7281, 0.7378, 0.7865, 0.7887, 0.7911, 0.7915, 0.8101, 0.8129, 0.8172,
    0.8511, 0.9266, 0.9313
])

# Calculate F1 scores
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Find the index of the maximum F1 score, avoiding 1.0
best_index = torch.argmax(f1_scores)
best_f1_score = f1_scores[best_index].item()

# Adjust index if F1 score is 1.0
while best_f1_score == 1.0 and best_index < len(f1_scores) - 1:
    best_index += 1
    best_f1_score = f1_scores[best_index].item()

# Get the best threshold, precision, recall, and F1 score
best_threshold = thresholds[best_index].item()
best_precision = precision[best_index].item()
best_recall = recall[best_index].item()
best_f1_score = f1_scores[best_index].item()

print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best Precision: {best_precision:.4f}")
print(f"Best Recall: {best_recall:.4f}")
print(f"Best F1 Score: {best_f1_score:.4f}")

# Plot the Precision-Recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label="Precision-Recall Curve", marker="o")

# Highlight the point with the highest F1 score
plt.scatter(
    best_recall,
    best_precision,
    color="red",
    label=f"Best F1 Score ({best_f1_score:.4f})\nThreshold: {best_threshold:.4f}",
    s=100,
    edgecolors="black",
)

# Add labels and title
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

# Display the plot
plt.grid(True)
plt.show()