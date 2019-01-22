from keras.models import Sequential
from keras.layers import Dense, Activation
from utility import get_normalized_data, y2indicator

xtrain, ytrain, xtest, ytest = get_normalized_data()

n, d = xtrain.shape
k = len(set(ytrain))

ytrain = y2indicator(ytrain)
ytest = y2indicator(ytest)

model = Sequential()
print(d)
model.add(Dense(units=500, input_dim=d))
model.add(Activation('relu'))
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dense(units=k))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

r = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=1, batch_size=32)

print(r.history_keys(()))
