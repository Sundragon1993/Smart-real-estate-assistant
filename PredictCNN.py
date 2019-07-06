import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from bokeh.io import output_notebook
from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure, show
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


pri_test = pd.read_csv("private_test.csv")
pri_train = pd.read_csv("private_train.csv")
print(pri_test.shape)
print(pri_train.shape)
pri_train_price = pri_train['price']
pri_train_x, pri_val_x, pri_train_y, pri_val_y = train_test_split(pri_train, pri_train_price, test_size=0.15,
                                                                  random_state=123)
print(pri_train_x.shape)
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

main_input = Input(shape=(21,), name='main_input')
emb = Embedding(256 * 8, output_dim=64, input_length=21)(main_input)
conv1d = Conv1D(filters=32, kernel_size=3, padding='valid')(emb)
bn = BatchNormalization()(conv1d)
sgconv1d = Activation('sigmoid')(bn)
conv1d_2 = Conv1D(filters=32, kernel_size=3, padding='valid')(sgconv1d)
bn2 = BatchNormalization()(conv1d_2)
sgconv1d_2 = Activation('sigmoid')(bn2)
# conv = Multiply()([conv1d, sgconv1d])
# pool = MaxPooling1D(pool_size = 32)(conv)
out = Flatten()(sgconv1d_2)
out = Dense(512, activation='relu')(out)
out = Dense(256, activation='relu')(out)

loss = Dense(1, activation='linear')(out)

model = Model(inputs=[main_input], outputs=[loss])
model.compile(loss='mean_absolute_percentage_error', optimizer='Adam', \
              metrics=['mean_squared_error', 'mean_absolute_percentage_error'])
model.summary()

pri_train_x = pri_train_x.values
pri_train_y = pri_train_y.values
pri_val_x = pri_val_x.values
pri_val_y = pri_val_y.values

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=5,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=5,
                               verbose=1,
                               mode='auto')
history = model.fit(pri_train_x, pri_train_y, validation_data=(pri_val_x, pri_val_y), epochs=1000, batch_size=128,
                    callbacks=[learning_rate_reduction])
print(history.history.keys())

plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('Mean Absolute Percentage Error')
plt.ylabel('MAPE')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

