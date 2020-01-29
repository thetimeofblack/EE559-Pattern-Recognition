import numpy as np 
from python3.plotDecBoundaries import plotDecBoundaries
train_data = np.genfromtxt('python3/synthetic1_train.csv',delimiter=',')
test_data =  np.genfromtxt('python3/synthetic1_test.csv',delimiter = ',')
#train_data_2 = np.genfromtxt('python3/synthetic2_train.csv', delimiter = ',')
#print(train_data_1)

sumClass1 = np.array([0,0])
sumClass2 = np.array([0,0])
countClass1 = 0 
countClass2 = 0
train_labels = []
test_labels = [] 
for data in train_data_1:
    train_labels.append(data[2])

for data in train_data:
	if data[2] == 1 : 
		sumClass1[0] += data[0]
		sumClass1[1] += data[1]
		countClass1+=1
	else: 
		sumClass2[0] += data[0]
		sumClass2[1] += data[1]
		countClass2+=1
for data in test_data

#print(sumClass2/countClass2)
#print(train_labels)
mean_class1 = sumClass1/countClass1 
mean_class2 = sumClass2/countClass2
mean_class1_label = np.append(mean_class_1,1)
mean_class2_label = np.append(mean_class_2,2)
mean_sample = np.array([mean_class1,mean_class2])

print(mean_sample)
plotDecBoundaries(train_data_1[:,[0,1]],train_labels,mean_sample)






