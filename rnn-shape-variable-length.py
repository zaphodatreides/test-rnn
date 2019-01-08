import numpy as np
import random
from numpy import array
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM

dataset=[]
dataset_y=[]

for i in range(4000):
 subset=[]
 for j in range(random.randint(20,20)):
  subset.append(random.randint(20,40))
 if random.randint(0,9) == 4:
  subset[0]=255
  dataset_y.append(True)
 else:
  if subset[4]== 4:
   subset[4]=5
  dataset_y.append(False)
 dataset.append(subset)
 
train_x=array(dataset).reshape(4000,20,1)
train_y=array(dataset_y)


dataset=[]
dataset_y=[]
for i in range(200):
 subset=[]
 for j in range(random.randint(20,20)):
  subset.append(random.randint(20,40))
 if random.randint(0,9) == 4:
  subset[0]=255
  dataset_y.append(True)
 else:
  if subset[4]== 4:
   subset[4]=5
  dataset_y.append(False)
 dataset.append(subset)
test_x=array(dataset).reshape(200,20,1)
test_y=array(dataset_y)


final_t=[255,1,24,0,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
final=array(final_t).reshape (1,20,1)

model = Sequential()

model.add(LSTM(40, input_shape=(20,1), return_sequences=True)) #,activation='sigmoid',recurrent_activation='hard_sigmoid'
model.add(Dropout(0.2))

#model.add(LSTM(32,return_sequences=False))
#model.add(Dropout(0.2))

model.add(TimeDistributed(Dense(1,activation='sigmoid')))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) #binary output

model.fit(train_x, train_y, batch_size=1000, epochs=100, validation_split=0.05,validation_data=(test_x,test_y))

final_y=model.predict_classes(final)
print (final_y)
final_y=model.predict_proba(final)
print (final_y)

