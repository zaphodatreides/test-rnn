import random
import pandas as pd

dataset=[]
for i in range (1000):
 subdataset=[]
 sub1=[]
 
 for j in range (random.randint(200,800)):
  subdataset.append(random.randint(0,255))
 if random.randint(0,9)==4:
  sub1.append('True')
  if subdataset[10]!=4:
   subdataset[10]=4 #str(random.randint(5,255))
 else:
  sub1.append('False')
 sub1.append(subdataset) 
 dataset.append(subdataset)
 
#with open('./dataset/YES-'+str(i)+'.csv', 'w') as csvfile:
#print (dataset[0])
output=pd.DataFrame(dataset)
output.to_csv('./dataset/dataset.csv', index=False, header=False)