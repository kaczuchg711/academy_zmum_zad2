# -*- coding: utf-8 -*-


"""ZMUM_S_Zadanie_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XjuOh3uMqbvMgYa7Nc75PNFYla0u_cEM

<center>
    <img src='http://torus.uck.pk.edu.pl/~amarsz/images/zmum/naglowek_power2.png', alt="Logo: Fundusze Europejskie Wiedza Edukacja Rozwój, Politechnika Krakowska, Unia Europejska Europejski Fundusz Spełeczny">
</center>
<center>
    <font size="5"> Zaawansowane Metody Uczenia Maszynowego<br/>
        <small><em>Studia stacjonarne II stopnia 2021/2022</em><br/>Kierunek: Informatyka<br>Specjalność: Informatyka Stosowana</small>
    </font>
</center>
<br><center><small>
Projekt„Programowanie doskonałości – PK XXI 2.0. Program rozwoju Politechniki Krakowskiej<br/> na lata 2018-22” dofinansowany z Europejskiego Funduszu Społecznego<br/>
Umowa nr POWR.03.05.00-00-z224/17
    </small>
</center>

# Zadanie nr 2:

Głównym zadaniem jest stworzenie i wytrenowanie głębokiej rekurencyjnej sieci neuronowej zdolnej poprawnie rozwiązać jedno z zagadnień rozpoznawania mowy a mianowicie zagadnienie identyfikacji rozmówcy.

Dane uczące zawierają nagrania audio pochodzące z rozmowy pomiędzy osobą A i B (plik `dataset.zip` dołączony do zadania na elf'ie).

W trakcie realizacji zadania należy wykonań następujące podzadania:
1. Podzielić próbki na dane uczące i testowe.
2. Zdefiniować __dwa__ modele głebokiej rekurencyjnej sieci neuronowej według własnego pomysłu. W pierwszym modelu dzwięk reprezentowany powinien być za pomocą szeregu czasowego, a w drugim za pomocą spektogramu MFCC (warto wykorzystać pakiet [librosa](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html). Oba modele powinny rozwiązywać problem klasyfikacji, czy dana próbka pochodzi od osoby A czy B.
3. Wytrenować zdefiniowane sieci na danych uczących.
4. Ocenić skuteczność i porównać działanie modeli na danych testowych.
5. Najlepsze ze stworzonych modeli zapisać do pliku i przesłać na elf'a razem z notatnikiem.

__Polecam też rozszerzyć zbiór danych o nagrania własnego głosu.__

## Import bibliotek
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import librosa.display
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Conv1D, LSTM, BatchNormalization, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

print('Numpy version:', np.__version__)
print('Tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

import sys

path_nb = r''
sys.path.append(path_nb)

"""## Przygotowanie danych"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

tf.get_logger().setLevel(logging.ERROR)
files = []
folders = []
for (path, dirnames, filenames) in os.walk(path_nb + 'dataset'):
    folders.extend(os.path.join(path, name) for name in dirnames)
    files.extend(os.path.join(path, name) for name in filenames)

print(files)
print(folders)

print(os.listdir(path_nb + 'dataset/A'))

data_y = []
data_sr = []
labels = []
for dir in os.listdir(path_nb + 'dataset'):
    for data_file in os.listdir(path_nb + 'dataset/' + dir):
        a = path_nb + 'dataset/' + dir + "/" + data_file
        print(a)
        y, sr = librosa.load(path_nb + 'dataset/' + dir + "/" + data_file)
        data_sr.append(sr)
        data_y.append(y)
        labels.append(dir)

print(data_y)
print(data_sr)
print(labels)

librosa.display.waveplot(data_y[10], sr=data_sr[0], x_axis='time', color='purple', offset=0.0)

y = data_y[0]
hop_length = 512  # the default spacing between frames
n_fft = 255  # number of samples
# cut the sample to the relevant times
MFCCs = librosa.feature.mfcc(y, sr=22050, n_mfcc=30)
print(MFCCs.shape)
fig, ax = plt.subplots(figsize=(20, 7))
librosa.display.specshow(MFCCs, sr=22050, hop_length=hop_length)
ax.set_xlabel('Time', fontsize=15)
ax.set_title('MFCC', size=20)
plt.colorbar()
plt.show()

"""MFCC"""

pathToDataset = path_nb + 'dataset'
class_names = os.listdir(pathToDataset)
audio_paths = []
labels = []
for label, name in enumerate(class_names):
    dir_path = Path(pathToDataset) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

shuffleSeed = 10
np.random.RandomState(shuffleSeed).shuffle(audio_paths)
np.random.RandomState(shuffleSeed).shuffle(labels)


def modelMFCC(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccsProcessed = np.mean(mfccs.T, axis=0)
    return mfccsProcessed


features = []
for file, label in zip(audio_paths, labels):
    data = modelMFCC(file)
    features.append([data, label])

featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

"""Podział na dane trneujące i testowe

"""

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

"""Model

"""

model = Sequential()
model.add(Input(shape=(x_train.shape[1], x_train.shape[2],)))
from keras.layers import LSTM
from keras import backend as K



model.add(LSTM(256, return_sequences=False))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

cb = [ModelCheckpoint(path_nb + 'modelMFCC.h5', monitor='val_accuracy', save_best_only=True)]
model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
num_epochs = 15
num_batch_size = 3
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1,
          callbacks=cb)

"""Zastosowanie modelu na danych testowych

"""

fmodel = tf.keras.models.load_model(path_nb + 'modelMFCC.h5')
fmodel.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
result = fmodel.evaluate(x_test, y_test, verbose=0)
print("Accuracy = {0:.0%}".format(result[1]))

"""Reprezentacja szeregiem czasowym"""


def modelSTFTS(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    stfts = librosa.stft(y=audio)
    stfts_processed = np.mean(stfts.T, axis=0)
    return stfts_processed


features = []
for file, label in zip(audio_paths, labels):
    data = modelSTFTS(file)
    features.append([data, label])

featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(Input(shape=(x_train.shape[1], x_train.shape[2],)))
model.add(Conv1D(filters=64, kernel_size=2))
model.add(MaxPooling1D(pool_size=16, strides=4))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
model.summary()

cb = [ModelCheckpoint(path_nb + 'modelSTFTS.h5', monitor='val_accuracy', save_best_only=True)]
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
num_epochs = 50
num_batch_size = 30
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1,
          callbacks=cb)

fmodel = tf.keras.models.load_model(path_nb + 'modelSTFTS.h5')
fmodel.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
result = fmodel.evaluate(x_test, y_test, verbose=0)
print("Accuracy = {0:.0%}".format(result[1]))