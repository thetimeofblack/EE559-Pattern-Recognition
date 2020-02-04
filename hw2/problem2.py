import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from python3.plotDecBoundaries import plotDecBoundaries
from tools.nearest_centroid_classifier import *

def getDataByClass(DataSet , classlabel , f1, f2):
    result = []
    for data in DataSet:
        if data[13] == classlabel:
            result.append([data[f1],data[f2]])
    return result

def getRestClassData(DataSet, classlabel, f1 ,f2):
    result = []
    for data in DataSet:
        if data[13] != classlabel:
            result.append([data[f1],data[f2]])
    return np.array(result)

def getMeanVector(DataSet, f1,f2):
    result = np.array([0.0,0.0])
    count = 0
    for data in DataSet:
        result += data
        count += 1
    return np.array(result/count)

def getLabelledSampleMeanSet(mean1 ,mean2 ):
    labelledmean1 = np.append(mean1,1)
    labelledmean2 = np.append(mean2,2)
    result = np.array([labelledmean1,labelledmean2])
    return result

def getUnlabelledSampleMeanSet(mean1,mean2):
    return np.array([mean1,mean2])


def getLabelSetOne2Rest(training,classlabel):
    LabelSet = []
    for data in training :
        if data[13] != classlabel:
            LabelSet.append(2)
        else:
            LabelSet.append(1)
    return np.array(LabelSet)

def showOne2RestDecisionBoundary(DataSet,classlabel, feature1,feature2):
    DataClassOne = getDataByClass(DataSet, classlabel, feature1, feature2)
#   print(DataClassOne)
    DataClassOneRest = getRestClassData(DataSet, classlabel, feature1, feature2)
    SampleMeanClassOne = getMeanVector(DataClassOne, feature1, feature2)
    SampleMeanClassOneRest = getMeanVector(DataClassOneRest, feature1, feature2)
#    print(SampleMeanClassOne)
#    SampleMeanSetClassOne = getLabelledSampleMeanSet(SampleMeanClassOne, SampleMeanClassOneRest)

    SampleMeanUnlabelledSetClassOne = getUnlabelledSampleMeanSet(SampleMeanClassOne, SampleMeanClassOneRest)

#    print(SampleMeanSetClassOne)


    TrainDataLabelSetOne2Rest = getLabelSetOne2Rest(DataSet, classlabel)
#    print(TrainDataLabelSetOne2Rest)
#    print(DataSet)
    plotDecBoundaries(DataSet[:, [feature1, feature2]], TrainDataLabelSetOne2Rest, SampleMeanUnlabelledSetClassOne)


def getSampleMeanSetForAllClasses(DataSet,feature1,feature2):
    
    return 0

def showOne2RestDecisionBoundaryForAllClasses(DataSet,feature1,feature2):
    return 0








train_data = np.genfromtxt('python3/wine_train.csv' , delimiter = ',')
test_data = np.genfromtxt('python3/wine_test.csv' , delimiter = ',')
test_data_labels = getLabels(test_data)

feature1 = 0
feature2 =  1

TrainDataErrorRate , sample_mean_set_unlabelled ,sample_mean_set_labelled= searchFeature(train_data,test_data,feature1, feature2)
estimatedTestDataLabel = nearest_classifier(test_data,sample_mean_set_labelled,feature1,feature2)
computeErrorRate(test_data_labels,estimatedTestDataLabel)


showOne2RestDecisionBoundary(train_data,1,0,1)
showOne2RestDecisionBoundary(train_data,2,0,1)
showOne2RestDecisionBoundary(train_data,3,0,1)

