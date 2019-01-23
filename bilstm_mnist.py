from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Lambda, Concatenate, Dense
import numpy as np
import pandas as pd
import keras.backend as K


def get_mnist():
    df = pd.read_csv('data/train_minist.csv')
    data = df.values
    np.random.shuffle(data)
    x1 = data[:, 1:]
    x = data[:, 1:].reshape(-1, 28, 28) / 255.0
    print(x.shape)
    y = data[:, 0]
    return x, y


x_data, y_data = get_mnist()

d = 28
m = 15

input = Input(shape=(d, d))
rnn1 = Bidirectional(LSTM(m, return_sequences=True))
x1 = rnn1(input) # output is N x D x 2M
x1 = GlobalMaxPooling1D()(x1) # output is N x 2M

rnn2 = Bidirectional(LSTM(m, return_sequences=True))

permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))

x2 = permutor(input)
x2 = rnn2(x2) # output is N x D x 2M
x2 = GlobalMaxPooling1D()(x2) # output is N x 2M

concatenator = Concatenate(axis=1)
x = concatenator([x1, x2])

output = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(x_data, y_data, batch_size=32, epochs=10, validation_split=0.3)


