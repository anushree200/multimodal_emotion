import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

train_padded = np.load('train_sequences.npy')
test_padded = np.load('test_sequences.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

if train_labels.dtype.type is np.str_ or train_labels.dtype.type is np.object_:
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

num_classes = len(np.unique(train_labels))
train_labels_cat = to_categorical(train_labels, num_classes=num_classes)
test_labels_cat = to_categorical(test_labels, num_classes=num_classes)

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_labels),
                                     y=train_labels)
class_weights = dict(enumerate(class_weights))

vocab_size = 10000
input_length = train_padded.shape[1]

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(128)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-3),
              metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

history = model.fit(
    train_padded, train_labels_cat,
    epochs=30,
    batch_size=32,
    validation_data=(test_padded, test_labels_cat),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

model.save("emotion_rnn_model.h5")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
