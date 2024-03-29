from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional
import numpy as np

T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)


input = Input(shape=(T, D))
#rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=False))
rnn = Bidirectional(LSTM(M, return_state=True, return_sequences=True))
x = rnn(input)

model = Model(inputs=input, outputs=x)
o, h1, c1, h2, c2 = model.predict(X)
print("o:", o)
print("o.shape", o.shape)
print('h1:', h1)
print('c1:', c1)
print('h2:', h2)
print('c2:', c2)

