import os
import pandas as pd
from sklearn.model_selection import train_test_split

path = r"C:\Users\aanuu\Downloads\EPOCH_MLTASK\archive"

def parse_emotion(filename):
    emotion_code = int(filename.split('-')[2])
    emotions = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    return emotions.get(emotion_code, 'unknown')

data = []
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.wav'):
            emotion = parse_emotion(file)
            file_path = os.path.join(root, file)
            data.append({'file_path': file_path, 'emotion': emotion})

df = pd.DataFrame(data)

df.to_csv('ravdess.csv', index=False)
print("Created ravdess_metadata.csv with", len(df), "files.")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])

train_df.to_csv('ravdess_train.csv', index=False)
test_df.to_csv('ravdess_test.csv', index=False)
print(f"Created ravdess_train.csv ({len(train_df)} samples) and ravdess_test.csv ({len(test_df)} samples).")