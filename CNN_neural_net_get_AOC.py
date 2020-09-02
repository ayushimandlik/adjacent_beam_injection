from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import concatenate
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import argparse
import locale
import h5py

def getting_data(h5_file):
    f = h5py.File(h5_file)
    print(h5_file)
    freq_time_data = np.array(f['data_freq_time'])[:, ::-1].T
    dm_time_data = np.array(f['data_dm_time']).T
    return freq_time_data, dm_time_data



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("weights.best.5D_FINAL_learning_rate_0.001_batch_150.hdf5")

training_set = pd.read_csv("/fred/oz002/amandlik/20_beams_RFI_adj_beam_candidates/TEST_SET_first_run.csv")

X_freq_time_test = []
X_dm_time_test = []
y = []
for i,r in training_set.iterrows():
    freq_time_data, dm_time_data = getting_data(r[0])
    X_freq_time_test.append(freq_time_data)
    X_dm_time_test.append(dm_time_data)
    y.append(r[1])

le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

def correct_dim_cnn(array):
    output=[]
    for i in array:
        output.append(i.reshape(np.shape(i)[1], np.shape(i)[2],5))
    return output

x_freq_time_test = correct_dim_cnn(X_freq_time_test)
x_dm_time_test = correct_dim_cnn(X_dm_time_test)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

score = model.evaluate([x_freq_time_test, x_dm_time_test], yy, verbose=1) 
accuracy = 100*score[1]

print("Accuracy on test set: %.4f%%" % accuracy)

y_pred_keras = model.predict([x_freq_time_test, x_dm_time_test], verbose=1)

pred = pd.DataFrame(y_pred_keras, header = None, index = None)
predictions = pd.concat([training_set, pred], axis = 1)
predictions.to_csv("predictions_output.csv")

fpr_keras, tpr_keras, thresholds_keras = roc_curve(yy, y_pred_keras[:,1])

auc_keras = auc(fpr_keras, tpr_keras)
print("AOC score: " + str(auc_keras))









