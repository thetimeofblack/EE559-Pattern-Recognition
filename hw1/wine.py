import numpy as np 
from python3.plotDecBoundaries import plotDecBoundaries
from tools.nearest_centroid_classifier import *

def find_features(train_data_set, test_data_set ):
    print(train_data)


train_data = np.genfromtxt('python3/wine_train.csv',  delimiter = ',')
test_data =  np.genfromtxt('python3/wine_test.csv' ,  delimiter = ',')

train_data_labels = []
test_data_labels = []

getLabels(train_data, train_data_labels)
getLabels(test_data, test_data_labels)
#print(train_data_labels)
train_error_set_for_features = []
test_error_set_for_features = []
feature1 = 1
feature2 = 2
mean_set_label1 = train_classifer_label(train_data,1,feature1,feature2)
mean_set_label2 = train_classifer_label(train_data,2,feature1,feature2)
mean_set_label3 = train_classifer_label(train_data,3,feature1,feature2)
mean_set_label1 = np.append(mean_set_label1,1)
mean_set_label2 = np.append(mean_set_label2,2)
mean_set_label3 = np.append(mean_set_label3,3)
sample_mean_set = [mean_set_label1,mean_set_label2,mean_set_label3]
print(mean_set_label1)
estimate_label_set = []
estimate_label_set= nearest_classifier(train_data,sample_mean_set,1,5)
print(estimate_label_set)
TrainDataErrorRate = computeErrorRate(estimate_label_set,train_data_labels)

'''
for i in range(len(train_data)-1):
    for j in range(i+1,len(train_data)-1):
        mean_set =  []
        train_mean_set_nearest_classifier(train_data,mean_set,i,j)
        train_error_set_for_features.append(computeErrorRate())
'''