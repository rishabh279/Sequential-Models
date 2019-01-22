from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np
import keras.backend as K

t = 8
d = 2
m = 3

X = np.random.randn(1, t, d)


def lstm1():
    input = Input(shape=(t, d))
    rnn = LSTM(m, return_state=True)
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    print(model.summary())
    o, h, c = model.predict(X)
    print('o:', o)
    print('h:', h)
    print('c:', c)


def lstm2():
    input = Input(shape=(t, d))
    rnn = LSTM(m, return_state=True, return_sequences=True)
    x = rnn(input)

    model = Model(inputs=input, outputs=x)
    o, h, c = model.predict(X)
    print('o:', o)
    print('h:', h)
    print('c:', c)


lstm1()
#lstm2()
