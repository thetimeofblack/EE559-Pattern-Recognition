import numpy as np
from math import *
def computeVectorEuclideanDistance(v1 ,v2):
    sum = 0
    for feature1, feature2 in v1,v2:
        sum += (feature1-feature2)*(feature1-feature2)

    return sqrt(sum)

def nearest_centroid_classifier(test_data, sample_mean_label):
    result = []  
    for data in test_data:
        minClass =  0 
        minDistance = 99
        for mean in sample_mean_label:
            mean_feature_set = mean[:,[0,1]]
            if computeVectorEuclideanDistance(mean_feature_set,data)<minDistance :
                 minDistance = computeVectorEuclideanDistance(mean_feature_set,data)
                 minClass = mean[2]
        result.append(minClass) 
    return result 
	
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


def getLabels(dataset,labelset):
    for data in dataset:
        labelset.append(data[13])




    
