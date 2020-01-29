import numpy as np 
from python3.plotDecBoundaries import plotDecBoundaries
from tools.nearest_centroid_classifier import *
def computeErrorRate(labelset1, labelset2):
    LabelCount = 0 
    ErrorCount = 0
    for i in range(len(labelset1)):
        if(labelset1[i]!=labelset2[i]):
            ErrorCount+=1
        LabelCount+=1
    result = ErrorCount/LabelCount
    print("The error rate: ",result,"  The total test data:  ",LabelCount)
    return result



train_data = np.genfromtxt('python3/synthetic2_train.csv',delimiter=',')
test_data =  np.genfromtxt('python3/synthetic2_test.csv',delimiter = ',')
#train_data_2 = np.genfromtxt('python3/synthetic2_train.csv', delimiter = ',')
#print(train_data_1)



sumClass1 = np.array([0,0])
sumClass2 = np.array([0,0])
countClass1 = 0 
countClass2 = 0
train_labels = []
test_labels = [] 
for data in train_data:
    train_labels.append(data[2])
for data in test_data:
    test_labels.append(data[2])

for data in train_data:
	if data[2] == 1 : 
		sumClass1[0] += data[0]
		sumClass1[1] += data[1]
		countClass1+=1
	else: 
		sumClass2[0] += data[0]
		sumClass2[1] += data[1]
		countClass2+=1
estimate_labels = [] 
test_data_unlabelled = test_data[:,[0,1]]
mean_class1 = sumClass1/countClass1 
mean_class2 = sumClass2/countClass2
mean_class1_label = np.append(mean_class1,1)
mean_class2_label = np.append(mean_class2,2)
mean_sample = np.array([mean_class1,mean_class2])
for data in test_data_unlabelled:
	if computeVectorEuclideanDistance(data,mean_class1)> computeVectorEuclideanDistance(data,mean_class2):
		estimate_labels.append(2)
	else:
		estimate_labels.append(1)

errorrate = computeErrorRate(estimate_labels,test_labels) 
plotDecBoundaries(train_data[:,[0,1]],train_labels,mean_sample)






