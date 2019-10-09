from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

dataset = pd.read_csv('Testing_Dataset.csv', header=None)

inputs = dataset.iloc[:,1:5]
outputs = dataset.iloc[:,5]

print(inputs)
print(outputs)

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(inputs,outputs,epochs=150, batch_size=10)

_, accuracy = model.evaluate(inputs, outputs)
print('Accuracy: %.2f' %(accuracy*100))

predictions = model.predict_classes(inputs)

for i in range(26):
    print('%d (expected %d)' % (predictions[i], outputs[i]))

