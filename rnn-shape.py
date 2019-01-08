import numpy as np
import random
from numpy import array
from keras.models import Sequential
from keras.layers import TimeDistributed,Flatten
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

dataset=[]
dataset_y=[]


filepath = "weights-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,        
    save_best_only=True,        
    mode='min'
)    


for i in range(2000):
 subset=[]
 for j in range(random.randint(20,20)):
  subset.append(random.randint(0,20))
 if random.randint(0,9) == 4:
  subset[4]=4
  dataset_y.append(True)
 else:
  if subset[4]== 4:
   subset[4]=5
  dataset_y.append(False)
 dataset.append(subset)
train_x=array(dataset).reshape(2000,20,1)
train_y=array(dataset_y)


dataset=[]
dataset_y=[]
for i in range(200):
 subset=[]
 for j in range(random.randint(20,20)):
  subset.append(random.randint(0,255))
 if random.randint(0,9) == 4:
  subset[4]=4
  dataset_y.append(True)
 else:
  if subset[4]== 4:
   subset[4]=5
  dataset_y.append(False)
 dataset.append(subset)
test_x=array(dataset).reshape(200,20,1)
test_y=array(dataset_y)


final_t=[0,1,24,5,6,8,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
final_t[4]=4
final=array(final_t).reshape (1,20,1)

model = Sequential()
model.add(LSTM(32,return_sequences=True, input_shape=(20,1)))
model.add(LSTM(16, return_sequences=False))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='categorical_crossentropy',optimizer='adam')
			  
#model.add(LSTM(4, input_shape=(20,1),activation='sigmoid',recurrent_activation='hard_sigmoid', return_sequences=False))
#model.add(Dense(10, input_dim=4,activation='sigmoid'))
#model.add(Dense(10,input_dim=10,activation='sigmoid'))
#model.add(Dense(1, input_dim=5,activation='sigmoid'))
model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
callbacks_list = [checkpoint]     

model.fit(train_x, train_y, batch_size=2000, epochs=300, validation_split=0.05,validation_data=(test_x,test_y), callbacks=callbacks_list)

final_y=model.predict_classes(final)
print (final_y)
final_y=model.predict_proba(final)
print (final_y)

