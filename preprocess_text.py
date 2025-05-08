import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import whisper
import os
from tqdm import tqdm 
try:
    train_labels = np.load('train_labels.npy')# Shape: (1152,)
    test_labels = np.load('test_labels.npy')# Shape: (288,)
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'train_labels.npy' and 'test_labels.npy' exist.")
    exit(1)

try:
    model = whisper.load_model("base", device="cuda")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    exit(1)

audio_dir = "C:/Users/aanuu/Downloads/EPOCH_MLTASK/archive"

if not os.path.exists(audio_dir):
    print(f"Error: Directory '{audio_dir}' not found.")
    exit(1)

audio_files = []
if any(os.path.isdir(os.path.join(audio_dir, d)) for d in os.listdir(audio_dir)):
    for actor_folder in os.listdir(audio_dir):
        actor_path = os.path.join(audio_dir, actor_folder)
        if os.path.isdir(actor_path):
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith('.wav'):
                    audio_files.append(os.path.join(actor_path, audio_file))
else:
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

audio_files = sorted(audio_files)
print(f"Found {len(audio_files)} audio files.")

expected_total = len(train_labels) + len(test_labels)
if len(audio_files) != expected_total:
    print(f"Warning: Expected {expected_total} audio files, found {len(audio_files)}.")
    test_size = len(audio_files) - len(train_labels)
    if test_size <= 0:
        print("Error: Not enough files.")
        exit(1)
    print(f"Adjusting test label count to {test_size}.")
    test_labels = test_labels[:test_size]
    np.save('test_labels.npy', test_labels)
    try:
        test_labels_encoded = np.load('test_labels_encoded.npy')
        test_labels_encoded = test_labels_encoded[:test_size]
        np.save('test_labels_encoded.npy', test_labels_encoded)
    except FileNotFoundError:
        print("Note: test_labels_encoded.npy not found. Continuing.")

train_audio_files = audio_files[:len(train_labels)]
test_audio_files = audio_files[len(train_labels):len(train_labels) + len(test_labels)]


def transcribe_files(file_list, label=""):
    transcripts = []
    for path in tqdm(file_list, desc=f"Transcribing {label}"):
        try:
            if os.path.exists(path):
                result = model.transcribe(path)
                transcripts.append(result["text"])
            else:
                transcripts.append("")
                print(f"Warning: Missing {path}")
        except Exception as e:
            print(f"Error transcribing {path}: {e}")
            transcripts.append("")
    return transcripts

train_transcripts = transcribe_files(train_audio_files, label="train set")
test_transcripts = transcribe_files(test_audio_files, label="test set")

train_transcripts_clean = [t if t not in ["<EMPTY>", "<ERROR>"] else "neutral" for t in train_transcripts]
test_transcripts_clean = [t if t not in ["<EMPTY>", "<ERROR>"] else "neutral" for t in test_transcripts]

np.save('train_transcripts.npy', train_transcripts)
np.save('test_transcripts.npy', test_transcripts)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_transcripts)

train_sequences = tokenizer.texts_to_sequences(train_transcripts)
test_sequences = tokenizer.texts_to_sequences(test_transcripts)

all_lengths = [len(seq) for seq in train_sequences]
avg_len = int(np.mean(all_lengths))
max_length = int(np.percentile(all_lengths, 95))#use 95th percentile
print(f"Average len: {avg_len}, Max (95%): {max_length}")

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

np.save('train_sequences.npy', train_padded)
np.save('test_sequences.npy', test_padded)

print("Preprocessing complete! Files saved:")
print("- train_transcripts.npy")
print("- test_transcripts.npy")
print("- train_sequences.npy")
print("- test_sequences.npy")
