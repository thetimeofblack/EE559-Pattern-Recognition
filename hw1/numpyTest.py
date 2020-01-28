import numpy as np 
from python3.plotDecBoundaries import plotDecBoundaries
train_data_1 = np.genfromtxt('python3/synthetic1_train.csv',delimiter=',') 
#print(train_data_1)
print(train_data_1[0:2])
sumClass1 = np.array([0,0])
sumClass2 = np.array([0,0])
countClass1 = 0 
countClass2 = 0 
train_labels = train_data_1[:,[2]]

for data in train_data_1:
	if data[2] == 1 : 
		sumClass1[0] += data[0]
		sumClass1[1] += data[1]
		countClass1+=1
	else: 
		sumClass2[0] += data[0]
		sumClass2[1] += data[1]
		countClass2+=1
print(sumClass1/countClass1)
print(sumClass2/countClass2)
mean_class1 = np.append(sumClass1/countClass1,[1]) 
mean_class2 = np.append(sumClass2/countClass2,[2])
mean_sample = np.array([mean_class1,mean_class2])
print(mean_sample)
plotDecBoundaries(train_data_1,train_labels,mean_sample)