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
 dataset.append(subset)
 dataset_y.append(subset[4])


#train_x=array(dataset).reshape(2000,1,20)
#train_y=array(dataset_y)


dataset=[]
dataset_y=[]
for i in range(200):
 subset=[]
 for j in range(random.randint(20,20)):
  subset.append(random.randint(0,20))
 dataset_y.append(subset[4])
 dataset.append(subset)

test_x=array(dataset).reshape(200,1,20)
test_y=array(dataset_y)

#for data,y in zip(dataset,dataset_y):
# print (y)
# print (data)
#quit(0)

final_t=[0,1,24,5,6,8,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
final_t[4]=4
final=array(final_t).reshape (1,1,20)

model = Sequential()
model.add(LSTM(20, input_shape=(1,20)))
model.add(Dense(21, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
			  

callbacks_list = [checkpoint]     
for i in range (2000):
 train_x=array(dataset[i]).reshape(1,1,20)
 train_y=array(dataset_y[i])
 model.fit(train_x, train_y, batch_size=1, epochs=1,callbacks=callbacks_list)

final_y=model.predict_classes(final)
print (final_y)
final_y=model.predict_proba(final)
print (final_y)

