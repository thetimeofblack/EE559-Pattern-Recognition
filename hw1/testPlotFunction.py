import numpy as np 
from python3.plotDecBoundaries import plotDecBoundaries
train_data =np.array( [[1,2], [2,3]]) 
train_labels = np.array([0,1])
mean_sample = np.array([[1,2],[2,3]])   
plotDecBoundaries(train_data, train_labels, mean_sample)