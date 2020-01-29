import numpy as np 
from python3.plotDecBoundaries import plotDecBoundaries
from tools.nearest_centroid_classifier import *


train_data = np.genfromtxt('python3/wine_train.csv',  delimiter = ',')
test_data =  np.genfromtxt('python3/wine_test.csv' ,  delimiter = ',')


train_data_labels = []
test_data_labels = []

getLabels(train_data, train_data_labels)
getLabels(test_data, test_data_labels)
print(train_data_labels)


