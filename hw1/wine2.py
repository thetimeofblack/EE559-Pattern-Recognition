import numpy as np
from python3.plotDecBoundaries import plotDecBoundaries
from tools.nearest_centroid_classifier import *

train_data = np.genfromtxt('python3/wine_train.csv',  delimiter = ',')
test_data =  np.genfromtxt('python3/wine_test.csv' ,  delimiter = ',')

feature1 = 0
feature2 = 1
train_data_labels= getLabels(train_data)
test_data_labels= getLabels(test_data)


TrainDataErrorRate,sample_mean_set_unlabelled , sample_mean_set_labelled = searchFeature(train_data,test_data, feature1 , feature2)

estimatedTestDataLabel = nearest_classifier(test_data,sample_mean_set_labelled,0,1)
computeErrorRate(test_data_labels,estimatedTestDataLabel)