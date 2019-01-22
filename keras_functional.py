from keras.models import Model
from keras.layers import Dense, Input
from utility import get_normalized_data, y2indicator

xtrain, ytrain, xtest, ytest = get_normalized_data()

n, d = xtrain.shape
k = len(set(ytrain))

ytrain = y2indicator(ytrain)
ytest = y2indicator(ytest)

i = Input(shape=(d,))
x = Dense(500, activation='relu')(i)
x = Dense(300, activation='relu')(x)
x = Dense(k, activation='softmax')(x)
model = Model(inputs=i, outputs=x)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
print(model.summary())
r = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=15, batch_size=32)
