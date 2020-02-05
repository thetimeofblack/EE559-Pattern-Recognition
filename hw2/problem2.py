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

def predictLabelSetOne2Rest(DataSet,SampleMeanSet) :
    DistanceMatrix = cdist(DataSet,SampleMeanSet)
    print(DistanceMatrix)
    LabelSet = []
    for data in DistanceMatrix:
        if data[0] > data[1]:
            LabelSet.append(2)
        else:
            LabelSet.append(1)
    return np.array(LabelSet)


def showOne2RestDecisionBoundary(DataSet,DataSetTest, TrainLabel,TestLabel,ClassType, feature1,feature2):
    DataClassOne = getDataByClass(DataSet, ClassType, feature1, feature2)
#   print(DataClassOne)
    DataClassOneRest = getRestClassData(DataSet, ClassType, feature1, feature2)
    SampleMeanClassOne = getMeanVector(DataClassOne, feature1, feature2)
    SampleMeanClassOneRest = getMeanVector(DataClassOneRest, feature1, feature2)
#    print(SampleMeanClassOne)
#    SampleMeanSetClassOne = getLabelledSampleMeanSet(SampleMeanClassOne, SampleMeanClassOneRest)
    TrainLabel = getLabelSetOne2Rest(DataSet,ClassType)
    TestLabel = getLabelSetOne2Rest(DataSetTest,ClassType)
    SampleMeanUnlabelledSetClassOne = getUnlabelledSampleMeanSet(SampleMeanClassOne, SampleMeanClassOneRest)

  # print(SampleMeanSetClassOne)


    PredictTrainLabelSet = predictLabelSetOne2Rest(DataSet[:,[feature1,feature2]], SampleMeanUnlabelledSetClassOne)

    PredictTestLabelSet = predictLabelSetOne2Rest(DataSetTest[:,[feature1,feature2]],SampleMeanUnlabelledSetClassOne)

    print(SampleMeanUnlabelledSetClassOne)
#    print(DataSet)
    TrainErrorRate = computeErrorRate(PredictTrainLabelSet,TrainLabel)
    TestErrorRate = computeErrorRate(PredictTestLabelSet,TestLabel)
    TrainDataErrorRate = 'The error rate of Train Data: ' + str(round(TrainErrorRate, 3))
    TestDataErrorRate = 'The error rate of Test Data: ' + str(round(TestErrorRate, 3))
    caption = TestDataErrorRate+'\n' + TrainDataErrorRate
    plotDecBoundaries(DataSet[:, [feature1, feature2]], caption,TrainLabel, SampleMeanUnlabelledSetClassOne)


def getSampleMeanSetForAllClasses(DataSet,feature1,feature2):
    DataClassOne   =  getDataByClass(DataSet,1,feature1,feature2)
    DataClassOneRest = getRestClassData(DataSet,1,feature1,feature2)
    DataClassTwo = getDataByClass(DataSet, 2, feature1, feature2)
    DataClassTwoRest = getRestClassData(DataSet, 2, feature1, feature2)
    DataClassThree = getDataByClass(DataSet, 3, feature1, feature2)
    DataClassThreeRest = getRestClassData(DataSet, 3, feature1, feature2)


    SampleMeanClassOne = getMeanVector(DataClassOne, feature1, feature2)
    SampleMeanClassOneRest = getMeanVector(DataClassOneRest, feature1, feature2)
    SampleMeanClassTwo = getMeanVector(DataClassTwo, feature1, feature2)
    SampleMeanClassTwoRest = getMeanVector(DataClassTwoRest, feature1, feature2)
    SampleMeanClassThree = getMeanVector(DataClassThree, feature1, feature2)
    SampleMeanClassThreeRest = getMeanVector(DataClassThreeRest, feature1, feature2)

    return np.array([SampleMeanClassOne,SampleMeanClassOneRest,SampleMeanClassTwo,
                     SampleMeanClassTwoRest,SampleMeanClassThree,SampleMeanClassThreeRest])


def predictDataLabelOne2RestForAllClasses(DataSet,SampleMeanSet) :
    DistanceMatrix = cdist(DataSet,SampleMeanSet)
    pred_label = []
    for data in DistanceMatrix:
        if data[0] < data[1] and data[2] > data[3] and data[4] > data[5]:
            pred_label.append(1)
        elif data[0] > data[1] and data[2] < data[3] and data[4] > data[5]:
            pred_label.append(2)
        elif data[0] > data[1] and data[2] > data[3] and data[4] < data[5]:
            pred_label.append(3)
        else:
            pred_label.append(0)
    return np.array(pred_label)

def plotDecisionBoundaryOne2Rest(DataSet,DataSetTest,DataLabelTrain,DataLabelTest,SampleMeanSet):
    # Plot the decision boundaries and data points for minimum distance to
    # class mean classifier
    #
    # training: traning data
    # label_train: class lables correspond to training data
    # sample_mean: mean vector for each class
    #
    # Total number of classes
    nclass = max(np.unique(DataLabelTrain))

    # Set the feature range for ploting
    max_x = np.ceil(max(DataSet[:, 0])) + 1
    min_x = np.floor(min(DataSet[:, 0])) - 1
    max_y = np.ceil(max(DataSet[:, 1])) + 1
    min_y = np.floor(min(DataSet[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005
    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                         np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                    y.reshape(y.shape[0] * y.shape[1], 1, order='F')))  # make (x,y) pairs as a bunch of row vectors.

    # distance measure evaluations for each (x,y) pair.
    dist_mat = cdist(xy, SampleMeanSet)
    pred_label = []
    for data in dist_mat:
        if data[0]<data[1] and data[2]>data[3] and data[4]>data[5] :
            pred_label.append(1)
        elif data[0]>data[1] and data[2]<data[3] and data[4]>data[5] :
            pred_label.append(2)
        elif data[0]>data[1] and data[2]>data[3] and data[4]<data[5] :
            pred_label.append(3)
        else :
            pred_label.append(0)
    pred_label = np.array(pred_label)
    # predict data
    TrainDataPredictLabelSet = predictDataLabelOne2RestForAllClasses(DataSet[:,[0,1]],SampleMeanSet)
    TestDataPredictLabelSet = predictDataLabelOne2RestForAllClasses(DataSetTest[:,[0,1]],SampleMeanSet)
    # TestDataPredictLabel = predictDataLabelNearestClassifierOne2Rest(test_data)

    TrainDataErrorRate=computeErrorRate(TrainDataPredictLabelSet,DataLabelTrain)
    TestDataErrorRate = computeErrorRate(TestDataPredictLabelSet,DataLabelTest)
    TrainDataErrorRate =  'The error rate of Train Data: ' + str(round(TrainDataErrorRate,3))
    TestDataErrorRate = 'The error rate of Test Data: ' + str(round(TestDataErrorRate,3))

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    # show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')
    # plot the class training data.
    plt.plot(DataSet[DataLabelTrain == 1, 0], DataSet[DataLabelTrain == 1, 1], 'rx')
    plt.plot(DataSet[DataLabelTrain == 2, 0], DataSet[DataLabelTrain == 2, 1], 'go')
    if nclass == 3:
        plt.plot(DataSet[DataLabelTrain == 3, 0], DataSet[DataLabelTrain == 3, 1], 'b*')

    # include legend for training data
    if nclass == 3:
        l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    else:
        l = plt.legend(('Class 1', 'Class 2'), loc=2)
    plt.gca().add_artist(l)

    # plot the class mean vector.
    m1, = plt.plot(SampleMeanSet[0, 0], SampleMeanSet[0, 1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(SampleMeanSet[2, 0], SampleMeanSet[2, 1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    if nclass == 3:
        m3, = plt.plot(SampleMeanSet[4, 0], SampleMeanSet[4, 1], 'bd', markersize=12, markerfacecolor='b',
                       markeredgecolor='w')

    # include legend for class mean vector
    if nclass == 3:
        l1 = plt.legend([m1, m2, m3], ['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
    else:
        l1 = plt.legend([m1, m2], ['Class 1 Mean', 'Class 2 Mean'], loc=4)

    plt.gca().add_artist(l1)

    # plot text for error rate of test data and train data
    caption = TestDataErrorRate+'\n' + TrainDataErrorRate
    plt.title(caption)
    #erl = plt.title((TestDataErrorRate,TrainDataErrorRate),loc=3)
    #plt.gca().add_artist(erl)


    plt.show()
    return 0








train_data = np.genfromtxt('python3/wine_train.csv' , delimiter = ',')
test_data = np.genfromtxt('python3/wine_test.csv' , delimiter = ',')
test_data_labels = getLabels(test_data)
train_data_labeles = getLabels(train_data)
feature1 = 0
feature2 =  1

TrainDataErrorRate , sample_mean_set_unlabelled ,sample_mean_set_labelled= searchFeature(train_data,test_data,feature1, feature2)
estimatedTestDataLabel = nearest_classifier(test_data,sample_mean_set_labelled,feature1,feature2)
computeErrorRate(test_data_labels,estimatedTestDataLabel)


showOne2RestDecisionBoundary(train_data,test_data,train_data_labeles,test_data_labels,1,0,1)
showOne2RestDecisionBoundary(train_data,test_data,train_data_labeles,test_data_labels,2,0,1)
showOne2RestDecisionBoundary(train_data,test_data,train_data_labeles,test_data_labels,3,0,1)

SampleMeanSetOne2Rest = getSampleMeanSetForAllClasses(train_data,feature1,feature2)

plotDecisionBoundaryOne2Rest(train_data,test_data,train_data_labeles,test_data_labels,SampleMeanSetOne2Rest)