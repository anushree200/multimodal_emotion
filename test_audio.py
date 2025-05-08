import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

test_spectrograms = np.load('test_spectrograms.npy')
test_labels = np.load('test_labels.npy')

train_labels = np.load('train_labels.npy')
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
emotion_classes = label_encoder.classes_
num_classes = len(emotion_classes)

test_labels_cat = to_categorical(test_labels_encoded, num_classes=num_classes)

model = load_model('audio_cnn_model.h5')

pred_probs = model.predict(test_spectrograms)
pred_labels = np.argmax(pred_probs, axis=1)

accuracy = accuracy_score(test_labels_encoded, pred_labels)
precision = precision_score(test_labels_encoded, pred_labels, average='weighted')
recall = recall_score(test_labels_encoded, pred_labels, average='weighted')
f1 = f1_score(test_labels_encoded, pred_labels, average='weighted')
conf_matrix = confusion_matrix(test_labels_encoded, pred_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_classes, yticklabels=emotion_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
