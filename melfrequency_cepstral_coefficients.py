import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.saving.saving_api import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_dir = 'dataset'
labels = {'A': 0, 'B': 1}
n_mfcc = 13
hop_length = 512
n_fft = 2048
audio_data = []
targets = []

for label in labels:
    dir_path = os.path.join(data_dir, label)
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        mfcc_padded = pad_sequences([mfcc.T], maxlen=306, dtype='float32', padding='post', truncating='post')[0]
        audio_data.append(mfcc_padded)
        targets.append(labels[label])

audio_data = np.array(audio_data)
targets = np.array(targets)

np.random.seed(123)
shuffle_indices = np.random.permutation(len(audio_data))
audio_data = audio_data[shuffle_indices]
targets = targets[shuffle_indices]

train_data = audio_data[:int(0.4 * len(audio_data))]
train_targets = targets[:int(0.4 * len(audio_data))]
val_data = audio_data[int(0.8 * len(audio_data)):]
val_targets = targets[int(0.8 * len(audio_data)):]

import matplotlib.pyplot as plt
# Load the audio file
audio_file = 'dataset/A/A1.wav'
y, sr = librosa.load(audio_file)

# Extract the MFCC features
n_mfcc = 13
hop_length = 512
n_fft = 2048
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

# Plot the MFCC features
plt.figure(figsize=(10, 4))
plt.imshow(mfcc, cmap='hot', interpolation='nearest')
plt.title('MFCC')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.colorbar()
plt.show()
exit(1)


model = Sequential()
model.add(Bidirectional(LSTM(64), input_shape=(306, n_mfcc)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# callbacks = [ModelCheckpoint('modelMFCC.h5', monitor='val_accuracy', save_best_only=True),EarlyStopping(verbose=20, patience=10)]
history = model.fit(train_data, train_targets, validation_data=(val_data, val_targets), epochs=100, batch_size=32)


model = load_model('modelMFCC.h5')
loss, accuracy = model.evaluate(val_data, val_targets)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)


# 2/2 [==============================] - 5s 573ms/step - loss: 0.5755 - accuracy: 0.7619 - val_loss: 0.6772 - val_accuracy: 0.5455
# Epoch 2/100
# 2/2 [==============================] - 0s 101ms/step - loss: 0.6123 - accuracy: 0.6905 - val_loss: 0.6520 - val_accuracy: 0.5455
# Epoch 3/100
# 2/2 [==============================] - 0s 116ms/step - loss: 0.5742 - accuracy: 0.6905 - val_loss: 0.6504 - val_accuracy: 0.5455
# Epoch 4/100
# 2/2 [==============================] - 0s 115ms/step - loss: 0.5445 - accuracy: 0.7381 - val_loss: 0.6470 - val_accuracy: 0.6364
# Epoch 5/100
# 2/2 [==============================] - 0s 96ms/step - loss: 0.5179 - accuracy: 0.8095 - val_loss: 0.6374 - val_accuracy: 0.7273
# Epoch 6/100
# 2/2 [==============================] - 0s 101ms/step - loss: 0.5001 - accuracy: 0.7857 - val_loss: 0.6186 - val_accuracy: 0.7273
# Epoch 7/100
# 2/2 [==============================] - 0s 79ms/step - loss: 0.5346 - accuracy: 0.7619 - val_loss: 0.6098 - val_accuracy: 0.7273
# Epoch 8/100
# 2/2 [==============================] - 0s 120ms/step - loss: 0.4663 - accuracy: 0.8571 - val_loss: 0.6018 - val_accuracy: 0.8182
# Epoch 9/100
# 2/2 [==============================] - 0s 80ms/step - loss: 0.4853 - accuracy: 0.7857 - val_loss: 0.5940 - val_accuracy: 0.8182
# Epoch 10/100
# 2/2 [==============================] - 0s 80ms/step - loss: 0.5154 - accuracy: 0.8095 - val_loss: 0.5865 - val_accuracy: 0.8182
# Epoch 11/100
# 2/2 [==============================] - 0s 85ms/step - loss: 0.4478 - accuracy: 0.8095 - val_loss: 0.5765 - val_accuracy: 0.8182
# Epoch 12/100
# 2/2 [==============================] - 0s 100ms/step - loss: 0.4988 - accuracy: 0.7857 - val_loss: 0.5722 - val_accuracy: 0.8182
# Epoch 13/100
# 2/2 [==============================] - 0s 115ms/step - loss: 0.4258 - accuracy: 0.9048 - val_loss: 0.5556 - val_accuracy: 0.9091
# Epoch 14/100
# 2/2 [==============================] - 0s 83ms/step - loss: 0.4872 - accuracy: 0.7857 - val_loss: 0.5553 - val_accuracy: 0.8182
# Epoch 15/100
# 2/2 [==============================] - 0s 84ms/step - loss: 0.4109 - accuracy: 0.9048 - val_loss: 0.5562 - val_accuracy: 0.7273
# Epoch 16/100
# 2/2 [==============================] - 0s 69ms/step - loss: 0.4532 - accuracy: 0.8571 - val_loss: 0.5441 - val_accuracy: 0.8182
# Epoch 17/100
# 2/2 [==============================] - 0s 73ms/step - loss: 0.4568 - accuracy: 0.8571 - val_loss: 0.5299 - val_accuracy: 0.9091
# Epoch 18/100
# 2/2 [==============================] - 0s 76ms/step - loss: 0.4046 - accuracy: 0.9524 - val_loss: 0.5216 - val_accuracy: 0.9091
# Epoch 19/100
# 2/2 [==============================] - 0s 83ms/step - loss: 0.4172 - accuracy: 0.8810 - val_loss: 0.5228 - val_accuracy: 0.9091
# Epoch 20/100
# 2/2 [==============================] - 0s 85ms/step - loss: 0.4185 - accuracy: 0.8810 - val_loss: 0.5194 - val_accuracy: 0.9091
# Epoch 21/100
# 2/2 [==============================] - 0s 101ms/step - loss: 0.3841 - accuracy: 0.9048 - val_loss: 0.4999 - val_accuracy: 0.9091
# Epoch 22/100
# 2/2 [==============================] - 0s 98ms/step - loss: 0.3977 - accuracy: 0.8571 - val_loss: 0.4792 - val_accuracy: 1.0000
# Epoch 23/100
# 2/2 [==============================] - 0s 92ms/step - loss: 0.4022 - accuracy: 0.9048 - val_loss: 0.4790 - val_accuracy: 0.9091
# Epoch 24/100
# 2/2 [==============================] - 0s 78ms/step - loss: 0.3833 - accuracy: 0.9524 - val_loss: 0.4778 - val_accuracy: 0.9091
# Epoch 25/100
# 2/2 [==============================] - 0s 89ms/step - loss: 0.3575 - accuracy: 0.9048 - val_loss: 0.4778 - val_accuracy: 0.9091
# Epoch 26/100
# 2/2 [==============================] - 0s 69ms/step - loss: 0.4146 - accuracy: 0.8333 - val_loss: 0.4639 - val_accuracy: 0.9091
# Epoch 27/100
# 2/2 [==============================] - 0s 71ms/step - loss: 0.4220 - accuracy: 0.8333 - val_loss: 0.4559 - val_accuracy: 0.9091
# Epoch 28/100
# 2/2 [==============================] - 0s 75ms/step - loss: 0.3755 - accuracy: 0.8810 - val_loss: 0.4454 - val_accuracy: 0.9091
# Epoch 29/100
# 2/2 [==============================] - 0s 113ms/step - loss: 0.3047 - accuracy: 0.9524 - val_loss: 0.4388 - val_accuracy: 0.8182
# Epoch 30/100
# 2/2 [==============================] - 0s 75ms/step - loss: 0.3454 - accuracy: 0.9048 - val_loss: 0.4294 - val_accuracy: 0.9091
# Epoch 31/100
# 2/2 [==============================] - 0s 78ms/step - loss: 0.3727 - accuracy: 0.8571 - val_loss: 0.4010 - val_accuracy: 0.9091
# Epoch 32/100
# 2/2 [==============================] - 0s 92ms/step - loss: 0.3360 - accuracy: 0.9524 - val_loss: 0.3862 - val_accuracy: 0.9091
# Epoch 33/100
# 2/2 [==============================] - 0s 88ms/step - loss: 0.3153 - accuracy: 0.9048 - val_loss: 0.3770 - val_accuracy: 0.9091
# Epoch 34/100
# 2/2 [==============================] - 0s 74ms/step - loss: 0.2903 - accuracy: 0.9524 - val_loss: 0.3751 - val_accuracy: 1.0000
# Epoch 35/100
# 2/2 [==============================] - 0s 113ms/step - loss: 0.3354 - accuracy: 0.8810 - val_loss: 0.3898 - val_accuracy: 0.9091
# Epoch 36/100
# 2/2 [==============================] - 0s 75ms/step - loss: 0.2583 - accuracy: 0.9524 - val_loss: 0.3803 - val_accuracy: 0.9091
# Epoch 37/100
# 2/2 [==============================] - 0s 78ms/step - loss: 0.3122 - accuracy: 0.9048 - val_loss: 0.3533 - val_accuracy: 1.0000
# Epoch 38/100
# 2/2 [==============================] - 0s 68ms/step - loss: 0.2462 - accuracy: 0.9762 - val_loss: 0.3557 - val_accuracy: 0.9091
# Epoch 39/100
# 2/2 [==============================] - 0s 77ms/step - loss: 0.2226 - accuracy: 1.0000 - val_loss: 0.3691 - val_accuracy: 0.9091
# Epoch 40/100
# 2/2 [==============================] - 0s 90ms/step - loss: 0.2836 - accuracy: 0.9762 - val_loss: 0.3784 - val_accuracy: 0.9091
# Epoch 41/100
# 2/2 [==============================] - 0s 106ms/step - loss: 0.2777 - accuracy: 0.9286 - val_loss: 0.3830 - val_accuracy: 0.9091
# Epoch 42/100
# 2/2 [==============================] - 0s 73ms/step - loss: 0.2447 - accuracy: 0.9524 - val_loss: 0.3828 - val_accuracy: 0.9091
# Epoch 43/100
# 2/2 [==============================] - 0s 78ms/step - loss: 0.2618 - accuracy: 0.9524 - val_loss: 0.3800 - val_accuracy: 0.9091
# Epoch 44/100
# 2/2 [==============================] - 0s 74ms/step - loss: 0.2659 - accuracy: 0.9762 - val_loss: 0.3738 - val_accuracy: 0.9091
# Epoch 45/100
# 2/2 [==============================] - 0s 91ms/step - loss: 0.2222 - accuracy: 0.9762 - val_loss: 0.3590 - val_accuracy: 0.9091
# Epoch 46/100
# 2/2 [==============================] - 0s 68ms/step - loss: 0.2738 - accuracy: 0.9286 - val_loss: 0.3530 - val_accuracy: 0.9091
# Epoch 47/100
# 2/2 [==============================] - 0s 79ms/step - loss: 0.2664 - accuracy: 0.9762 - val_loss: 0.3390 - val_accuracy: 1.0000
# Epoch 48/100
# 2/2 [==============================] - 0s 122ms/step - loss: 0.2416 - accuracy: 1.0000 - val_loss: 0.3382 - val_accuracy: 1.0000
# Epoch 49/100
# 2/2 [==============================] - 0s 62ms/step - loss: 0.2292 - accuracy: 1.0000 - val_loss: 0.3171 - val_accuracy: 1.0000
# Epoch 50/100
# 2/2 [==============================] - 0s 68ms/step - loss: 0.2694 - accuracy: 0.9048 - val_loss: 0.3277 - val_accuracy: 0.9091
# Epoch 51/100
# 2/2 [==============================] - 0s 72ms/step - loss: 0.2080 - accuracy: 0.9762 - val_loss: 0.3619 - val_accuracy: 0.9091
# Epoch 52/100
# 2/2 [==============================] - 0s 91ms/step - loss: 0.2024 - accuracy: 0.9762 - val_loss: 0.3588 - val_accuracy: 0.9091
# Epoch 53/100
# 2/2 [==============================] - 0s 95ms/step - loss: 0.1967 - accuracy: 0.9762 - val_loss: 0.3366 - val_accuracy: 0.9091
# Epoch 54/100
# 2/2 [==============================] - 0s 92ms/step - loss: 0.2490 - accuracy: 0.9286 - val_loss: 0.3169 - val_accuracy: 0.9091
# Epoch 55/100
# 2/2 [==============================] - 0s 75ms/step - loss: 0.1821 - accuracy: 1.0000 - val_loss: 0.2963 - val_accuracy: 0.9091
# Epoch 56/100
# 2/2 [==============================] - 0s 97ms/step - loss: 0.2634 - accuracy: 0.9524 - val_loss: 0.2819 - val_accuracy: 0.9091
# Epoch 57/100
# 2/2 [==============================] - 0s 73ms/step - loss: 0.1704 - accuracy: 1.0000 - val_loss: 0.2674 - val_accuracy: 1.0000
# Epoch 58/100
# 2/2 [==============================] - 0s 74ms/step - loss: 0.2234 - accuracy: 0.9762 - val_loss: 0.2669 - val_accuracy: 1.0000
# Epoch 59/100
# 2/2 [==============================] - 0s 93ms/step - loss: 0.1967 - accuracy: 1.0000 - val_loss: 0.2641 - val_accuracy: 0.9091
# Epoch 60/100
# 2/2 [==============================] - 0s 92ms/step - loss: 0.1989 - accuracy: 0.9762 - val_loss: 0.2630 - val_accuracy: 0.9091
# Epoch 61/100
# 2/2 [==============================] - 0s 64ms/step - loss: 0.1690 - accuracy: 1.0000 - val_loss: 0.2683 - val_accuracy: 0.9091
# Epoch 62/100
# 2/2 [==============================] - 0s 70ms/step - loss: 0.1769 - accuracy: 1.0000 - val_loss: 0.2859 - val_accuracy: 0.9091
# Epoch 63/100
# 2/2 [==============================] - 0s 70ms/step - loss: 0.1893 - accuracy: 1.0000 - val_loss: 0.2939 - val_accuracy: 0.9091
# Epoch 64/100
# 2/2 [==============================] - 0s 77ms/step - loss: 0.1795 - accuracy: 0.9762 - val_loss: 0.2998 - val_accuracy: 0.9091
# Epoch 65/100
# 2/2 [==============================] - 0s 86ms/step - loss: 0.1705 - accuracy: 1.0000 - val_loss: 0.2949 - val_accuracy: 0.9091
# Epoch 66/100
# 2/2 [==============================] - 0s 75ms/step - loss: 0.1753 - accuracy: 0.9762 - val_loss: 0.2855 - val_accuracy: 0.9091
# Epoch 67/100
# 2/2 [==============================] - 0s 74ms/step - loss: 0.1611 - accuracy: 1.0000 - val_loss: 0.2626 - val_accuracy: 0.9091
# Epoch 68/100
# 2/2 [==============================] - 0s 80ms/step - loss: 0.1676 - accuracy: 0.9762 - val_loss: 0.2622 - val_accuracy: 0.9091
# Epoch 69/100
# 2/2 [==============================] - 0s 78ms/step - loss: 0.1679 - accuracy: 1.0000 - val_loss: 0.2558 - val_accuracy: 0.9091
# Epoch 70/100
# 2/2 [==============================] - 0s 87ms/step - loss: 0.1684 - accuracy: 1.0000 - val_loss: 0.2437 - val_accuracy: 0.9091
# Epoch 71/100
# 2/2 [==============================] - 0s 70ms/step - loss: 0.1634 - accuracy: 0.9524 - val_loss: 0.2412 - val_accuracy: 0.9091
# Epoch 72/100
# 2/2 [==============================] - 0s 77ms/step - loss: 0.1754 - accuracy: 1.0000 - val_loss: 0.2494 - val_accuracy: 0.9091
# Epoch 73/100
# 2/2 [==============================] - 0s 81ms/step - loss: 0.1569 - accuracy: 0.9762 - val_loss: 0.2575 - val_accuracy: 0.9091
# Epoch 74/100
# 2/2 [==============================] - 0s 94ms/step - loss: 0.1574 - accuracy: 0.9762 - val_loss: 0.2450 - val_accuracy: 0.9091
# Epoch 75/100
# 2/2 [==============================] - 0s 80ms/step - loss: 0.1486 - accuracy: 1.0000 - val_loss: 0.2354 - val_accuracy: 0.9091
# Epoch 76/100
# 2/2 [==============================] - 0s 83ms/step - loss: 0.1126 - accuracy: 1.0000 - val_loss: 0.2312 - val_accuracy: 0.9091
# Epoch 77/100
# 2/2 [==============================] - 0s 75ms/step - loss: 0.1474 - accuracy: 0.9762 - val_loss: 0.2223 - val_accuracy: 0.9091
# Epoch 78/100
# 2/2 [==============================] - 0s 84ms/step - loss: 0.1526 - accuracy: 1.0000 - val_loss: 0.2114 - val_accuracy: 0.9091
# Epoch 79/100
# 2/2 [==============================] - 0s 83ms/step - loss: 0.1595 - accuracy: 0.9762 - val_loss: 0.2197 - val_accuracy: 0.9091
# Epoch 80/100
# 2/2 [==============================] - 0s 71ms/step - loss: 0.1188 - accuracy: 1.0000 - val_loss: 0.2144 - val_accuracy: 0.9091
# Epoch 81/100
# 2/2 [==============================] - 0s 91ms/step - loss: 0.1112 - accuracy: 1.0000 - val_loss: 0.2114 - val_accuracy: 0.9091
# Epoch 82/100
# 2/2 [==============================] - 0s 83ms/step - loss: 0.1410 - accuracy: 0.9762 - val_loss: 0.2079 - val_accuracy: 0.9091
# Epoch 83/100
# 2/2 [==============================] - 0s 86ms/step - loss: 0.1532 - accuracy: 0.9762 - val_loss: 0.2150 - val_accuracy: 0.9091
# Epoch 84/100
# 2/2 [==============================] - 0s 84ms/step - loss: 0.1238 - accuracy: 1.0000 - val_loss: 0.2370 - val_accuracy: 0.9091
# Epoch 85/100
# 2/2 [==============================] - 0s 90ms/step - loss: 0.1235 - accuracy: 1.0000 - val_loss: 0.2370 - val_accuracy: 0.9091
# Epoch 86/100
# 2/2 [==============================] - 0s 96ms/step - loss: 0.1373 - accuracy: 1.0000 - val_loss: 0.2139 - val_accuracy: 0.9091
# Epoch 87/100
# 2/2 [==============================] - 0s 74ms/step - loss: 0.1076 - accuracy: 1.0000 - val_loss: 0.2023 - val_accuracy: 0.9091
# Epoch 88/100
# 2/2 [==============================] - 0s 101ms/step - loss: 0.1022 - accuracy: 1.0000 - val_loss: 0.1971 - val_accuracy: 0.9091
# Epoch 89/100
# 2/2 [==============================] - 0s 71ms/step - loss: 0.1204 - accuracy: 1.0000 - val_loss: 0.1936 - val_accuracy: 0.9091
# Epoch 90/100
# 2/2 [==============================] - 0s 101ms/step - loss: 0.1321 - accuracy: 1.0000 - val_loss: 0.2152 - val_accuracy: 0.9091
# Epoch 91/100
# 2/2 [==============================] - 0s 73ms/step - loss: 0.1375 - accuracy: 1.0000 - val_loss: 0.2041 - val_accuracy: 0.9091
# Epoch 92/100
# 2/2 [==============================] - 0s 91ms/step - loss: 0.1272 - accuracy: 0.9762 - val_loss: 0.2063 - val_accuracy: 0.9091
# Epoch 93/100
# 2/2 [==============================] - 0s 87ms/step - loss: 0.1073 - accuracy: 1.0000 - val_loss: 0.1990 - val_accuracy: 0.9091
# Epoch 94/100
# 2/2 [==============================] - 0s 83ms/step - loss: 0.0956 - accuracy: 1.0000 - val_loss: 0.1806 - val_accuracy: 0.9091
# Epoch 95/100
# 2/2 [==============================] - 0s 63ms/step - loss: 0.0860 - accuracy: 1.0000 - val_loss: 0.1735 - val_accuracy: 0.9091
# Epoch 96/100
# 2/2 [==============================] - 0s 73ms/step - loss: 0.1074 - accuracy: 1.0000 - val_loss: 0.1679 - val_accuracy: 0.9091
# Epoch 97/100
# 2/2 [==============================] - 0s 90ms/step - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.1594 - val_accuracy: 1.0000
# Epoch 98/100
# 2/2 [==============================] - 0s 106ms/step - loss: 0.0897 - accuracy: 1.0000 - val_loss: 0.1560 - val_accuracy: 1.0000
# Epoch 99/100
# 2/2 [==============================] - 0s 78ms/step - loss: 0.0790 - accuracy: 1.0000 - val_loss: 0.1547 - val_accuracy: 1.0000
# Epoch 100/100
# 2/2 [==============================] - 0s 63ms/step - loss: 0.1066 - accuracy: 1.0000 - val_loss: 0.1309 - val_accuracy: 1.0000