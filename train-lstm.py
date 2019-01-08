from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import random
import numpy as np
from numpy import array

batch_size=1
x_t=[]
y_t=[]
x_te=[]
y_te=[]

for i in range (100):
 subdataset=[]
 
 for j in range (random.randint(15,25)):
  subdataset.append(random.randint(0,255))
 if random.randint(0,9)==4:
  y_t.append(True)
  if subdataset[10]!=4:
   subdataset[10]=4 #str(random.randint(5,255))
 else:
  y_t.append(False) 
 x_t.append(subdataset)

for i in range (40):
 subdataset=[]
 
 for j in range (random.randint(15,25)):
  subdataset.append(random.randint(0,255))
 if random.randint(0,9)==4:
  y_te.append(True)
  if subdataset[10]!=4:
   subdataset[10]=4 #str(random.randint(5,255))
 else:
  y_te.append(False) 
 x_te.append(subdataset)
 
x_train=array(x_t)
y_train=array(y_t)

x_test=array(x_te)
y_test=array(y_te)
 
model = Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(None, 5)))
model.add(LSTM(8, return_sequences=True))
model.add(Dense(2, activation='sigmoid'))

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
              optimizer='adam')


		
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
							
print('Test score:', score)
print('Test accuracy:', acc)
