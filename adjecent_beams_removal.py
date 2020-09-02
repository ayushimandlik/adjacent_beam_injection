import pandas as pd
import numpy as np
import h5py
import sys
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
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import concatenate
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import argparse
import locale
from tensorflow import keras
from tensorflow.keras import layers

def getting_data(h5_file):
    f = h5py.File(h5_file)
    print(h5_file)
    freq_time_data = np.array(f['data_freq_time'])[:, ::-1].T
    dm_time_data = np.array(f['data_dm_time']).T
    return freq_time_data, dm_time_data

def cnn_model(training_file):
    training_set = pd.read_csv(training_file, header = None)
    X_freq_time = []
    X_dm_time = []
    y = []
    for i,r in training_set.iterrows():
        freq_time_data, dm_time_data = getting_data(r[0])
        X_freq_time.append(freq_time_data)
        X_dm_time.append(dm_time_data)
        y.append(r[1])

    print(np.shape(X_freq_time))
    print(np.shape(X_dm_time))
    print(np.shape(y))

    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    # the dataset is slit into training and validation sets.
    x_freq_time_train, x_freq_time_test, x_dm_time_train, x_dm_time_test, y_train, y_test = train_test_split(X_freq_time, X_dm_time, yy, test_size=0.2, random_state = 42)

    def correct_dim_cnn(array):
        output=[]
        for i in array:
            output.append(i.reshape(np.shape(i)[1], np.shape(i)[2],5))
        return output
 
    x_freq_time_train = correct_dim_cnn(x_freq_time_train)
    x_freq_time_test = correct_dim_cnn(x_freq_time_test)
    x_dm_time_train = correct_dim_cnn(x_dm_time_train)
    x_dm_time_test = correct_dim_cnn(x_dm_time_test)

    print(np.shape(x_freq_time_train))
    def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
        inputShape = (height, width, depth)
        chanDim = -1
        #Initialising the input layer:
        inputs = Input(shape=inputShape)

        for (i, f) in enumerate(filters):
            # for the first filter, the input will be the first input layer. After that, the inputs are obtaineed from the preceding layer.
            if i == 0:
                x = inputs
            # the convolution layer goes through each of the filter elements (16,32,64).
            # the strides were (3,3). This can be changed - increased- to improve inference speeds. (Large scale features only). I went with 3,3 because I did not see the point in 
            # reducing the resolution too much. 
            x = Conv2D(f, (3, 3), padding="same")(x)
            # rectified linear unit: f(x) = max(0,x). Takes values above 0, and puts the values to 0 if the number is negative.
            x = Activation("relu")(x)
            # normalisation after every conv layer is being done.
            x = BatchNormalization(axis=chanDim)(x)
            # basically just downsampling. Taking the maximum in every pool. This is something tahn can be increased to improve the speed and load.
            x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # once all the convolutions are done, the pooled layer is flattened.
        x = Flatten()(x)
        # fully connected layer.
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        # To prevent overfitting. This can also be added several times. 0.5 is fraction of units to drop. Values from 0 to 1.
        x = Dropout(0.5)(x)
        
        # another dense layer.
        x = Dense(4)(x)
        x = Activation("relu")(x)

    # This is where the final 'model' is constructed, and returned as the output for the different inputs.
        model = Model(inputs, x)
        return model 
    
    # this is where the model building starts. 
    freq_time_cnn = create_cnn(np.shape(freq_time_data)[1], np.shape(freq_time_data)[2], 5, regress=False)
    dm_time_cnn = create_cnn(np.shape(dm_time_data)[1], np.shape(dm_time_data)[2], 5, regress=False)
    
    # the outputs from the layers are concatenated in this step.
    combinedInput = concatenate([freq_time_cnn.output, dm_time_cnn.output])

    # Do not do this. You should never hard wire. The number or labels represent 0 and 1. 1 is frb, 0 is RFI.
    num_labels = 2
    x = Dense(4, activation="relu")(combinedInput)i
    # this is when the output from the dense layer is converted to a probability. 
    x = Dense(num_labels, activation="softmax")(x)

    #The model has been constructed. The rest of the code is setting inputs of model and then its compilation.
    ###########################################################################

    model = Model(inputs=[freq_time_cnn.input, dm_time_cnn.input], outputs=x)

#    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

    # Compiling the model
#    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.summary()

    # LET THE TRAININGS BEGIN!!!!!!!!
    num_epochs = 100
    num_batch_size = 150

    checkpointer = ModelCheckpoint(filepath='weights.best.5D_FINAL_learning_rate_0.001_batch_150_epochs_100.hdf5',
                               verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit([x_freq_time_train, x_dm_time_train], y_train, validation_data=([x_freq_time_test, x_dm_time_test], y_test), batch_size=num_batch_size, epochs=num_epochs, callbacks=[checkpointer], verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Calculate the final accuracy on the validation set
    score = model.evaluate([x_freq_time_test, x_dm_time_test], y_test, verbose=1)
    accuracy = 100*score
#    accuracy = 100*score[1]

    print("Accuracy on validation set: %.4f%%" % accuracy)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('-t', '--training_file', help='training file with labels', type=str)
    values = a.parse_args()
    cnn_model(values.training_file)

