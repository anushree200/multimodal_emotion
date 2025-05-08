from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

model = load_model("emotion_rnn_model.h5")
test_sequences = np.load("test_sequences.npy")
test_labels = np.load("test_labels.npy")

le = LabelEncoder()
test_labels_encoded = le.fit_transform(test_labels)

pred_probs = model.predict(test_sequences)
pred_labels = pred_probs.argmax(axis=1)

accuracy = accuracy_score(test_labels_encoded, pred_labels)
precision = precision_score(test_labels_encoded, pred_labels, average='weighted')
recall = recall_score(test_labels_encoded, pred_labels, average='weighted')
f1 = f1_score(test_labels_encoded, pred_labels, average='weighted')
conf_matrix = confusion_matrix(test_labels_encoded, pred_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)

emotion_classes = le.classes_
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
