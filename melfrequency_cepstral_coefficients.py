import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the directory containing the audio files
data_dir = 'dataset'

# Define the labels and their corresponding integer codes
labels = {'A': 0, 'B': 1}

# Define the parameters for MFCC extraction
n_mfcc = 13
hop_length = 512
n_fft = 2048

# Define the maximum sequence length to pad/truncate the MFCC feature sequences
max_length = 306

# Initialize empty lists to store the MFCC feature sequences and their corresponding labels
data = []
targets = []

# Loop through the audio files in the data directory
for label in labels:
    dir_path = os.path.join(data_dir, label)
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        # Load the audio file and extract its MFCC features
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        # Pad or truncate the MFCC feature sequence to the maximum sequence length
        mfcc_padded = pad_sequences([mfcc.T], maxlen=max_length, dtype='float32', padding='post', truncating='post')[0]
        # Append the MFCC feature sequence and its corresponding label to the data and targets lists
        data.append(mfcc_padded)
        targets.append(labels[label])

# Convert the data and targets lists to numpy arrays
data = np.array(data)
targets = np.array(targets)

# Shuffle the data and targets in unison
np.random.seed(123)
shuffle_indices = np.random.permutation(len(data))
data = data[shuffle_indices]
targets = targets[shuffle_indices]

# Split the data and targets into training and validation sets
train_data = data[:int(0.8 * len(data))]
train_targets = targets[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]
val_targets = targets[int(0.8 * len(data)):]

# Define the recurrent neural network model
model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(max_length, n_mfcc)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_targets, validation_data=(val_data, val_targets), epochs=50, batch_size=32)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_data, val_targets)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)