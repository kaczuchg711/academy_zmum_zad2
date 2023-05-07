import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Conv1D, LSTM, BatchNormalization, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

print('Numpy version:', np.__version__)
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)
for i in range(1):
    data_amplitudes = []
    data_sampling_rate = []
    labels = []
    audio_paths = []

    for dir in os.listdir('dataset'):
        for data_file in os.listdir('dataset/' + dir):
            audio_paths.append('dataset/' + dir + "/" + data_file)
            amplitude, sr = librosa.load(audio_paths[-1], sr=30000)
            data_sampling_rate.append(sr)
            data_amplitudes.append(amplitude)
            labels.append(dir)

    # Plot amplitude, time representation
    # amplitude, sr = librosa.load('dataset/A/A13.wav')
    # librosa.display.waveplot(amplitude, sr=sr, x_axis='time', color='purple', offset=0.0)
    # plt.show()

    # Model STFTS function
    def modelSTFTS(file_name):
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        stfts = librosa.stft(y=audio)
        stfts_processed = np.mean(stfts.T, axis=0)
        return stfts_processed

    # Extract features
    features = []
    for file, label in zip(audio_paths, labels):
        data = modelSTFTS(file)
        features.append([data, label])

    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.3)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build and compile the model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='LeakyReLU'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='LeakyReLU'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='LeakyReLU'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1.0e-11, epsilon=1.00e-08)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    cb = [ModelCheckpoint('modelSTFTS.h5', monitor='val_accuracy', save_best_only=True), EarlyStopping(verbose=5, patience=100),] # tf.keras.callbacks.ReduceLROnPlateau(patience=3)

    num_epochs = 5
    num_batch_size = 32

    history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1, callbacks=cb)

    # os.system(f'echo "LERN RATE {str(1 * pow(0.1, i))}" "val_accuracy {max(history.history["val_accuracy"])}">> result_lerning_rate')




    #
    # best Model: "sequential_7"
    # _________________________________________________________________
    #  Layer (type)                Output Shape              Param #
    # =================================================================
    #  lstm_14 (LSTM)              (None, 1025, 256)         264192
    #
    #  dropout_35 (Dropout)        (None, 1025, 256)         0
    #
    #  lstm_15 (LSTM)              (None, 128)               197120
    #
    #  dropout_36 (Dropout)        (None, 128)               0
    #
    #  dense_28 (Dense)            (None, 64)                8256
    #
    #  dropout_37 (Dropout)        (None, 64)                0
    #
    #  dense_29 (Dense)            (None, 32)                2080
    #
    #  dropout_38 (Dropout)        (None, 32)                0
    #
    #  dense_30 (Dense)            (None, 16)                528
    #
    #  dropout_39 (Dropout)        (None, 16)                0
    #
    #  dense_31 (Dense)            (None, 2)                 34
    #
    # =================================================================
    # Total params: 472,210
    # Trainable params: 472,210
    # Non-trainable params: 0
    # _________________________________________________________________
    # Epoch 1/5
    # 2/2 [==============================] - 4s 864ms/step - loss: 0.6930 - accuracy: 0.4865 - val_loss: 0.6912 - val_accuracy: 0.6875
    # Epoch 2/5
    # 2/2 [==============================] - 0s 230ms/step - loss: 0.6936 - accuracy: 0.5676 - val_loss: 0.6859 - val_accuracy: 0.6875
    # Epoch 3/5
    # 2/2 [==============================] - 0s 234ms/step - loss: 0.6941 - accuracy: 0.5135 - val_loss: 0.6776 - val_accuracy: 0.6875
    # Epoch 4/5
    # 2/2 [==============================] - 0s 189ms/step - loss: 0.6925 - accuracy: 0.5405 - val_loss: 0.6706 - val_accuracy: 0.6875
    # Epoch 5/5
    # 2/2 [==============================] - 0s 211ms/step - loss: 0.6972 - accuracy: 0.5135 - val_loss: 0.6674 - val_accuracy: 0.6875