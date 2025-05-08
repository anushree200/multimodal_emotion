import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_spectrograms = np.load('train_spectrograms.npy')  # (1152, 128, 128, 1)
train_labels = np.load('train_labels.npy')
test_spectrograms = np.load('test_spectrograms.npy')    # (288, 128, 128, 1)
test_labels = np.load('test_labels.npy')

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
num_classes = len(label_encoder.classes_)

print("Label mapping:", dict(zip(label_encoder.classes_, range(num_classes))))

train_labels_cat = to_categorical(train_labels_encoded, num_classes=num_classes)
test_labels_cat = to_categorical(test_labels_encoded, num_classes=num_classes)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(128, 128, 1)),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), padding='same'),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=(3, 3), padding='same'),
    ReLU(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256),
    ReLU(),
    Dropout(0.7),

    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

model.fit(train_spectrograms, train_labels_cat,
          validation_data=(test_spectrograms, test_labels_cat),
          epochs=30,
          batch_size=32,
          callbacks=[early_stop, reduce_lr])

model.save('audio_cnn_model.h5')
print("Training complete! Model saved as 'audio_cnn_model.h5'.")
