import numpy as np
from python3.plotDecBoundaries import plotDecBoundaries
from tools.nearest_centroid_classifier import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches

def G12(x):
    return np.array(-x[0]- x[1] +5)
def G13(x):
    return np.array(-x[0]+3)
def G23(x):
    return np.array(-x[0]+x[1]-1)


def judgeClassOne(pixel):
    return 0
def judgeClassTwo(pixel):
    return 0
def judgeClassThree(pixel):
    return 0

'''
train_data =np.array( [[1,2], [2,3]])
train_labels = np.array([0,1])
mean_sample = np.array([[0,0],[5,5]])
plotDecBoundaries(train_data, train_labels, mean_sample)
'''

max_x = 10.0
min_x =-10.0
max_y = 10.0
min_y = -10.0

xrange = (min_x , max_x)
yrange = (min_y ,max_y)

inc = 0.05

(x, y) = np.meshgrid(np.arange(xrange[0],xrange[1]+inc/100 ,inc),np.arange(yrange[0],yrange[1]+inc/100 ,inc))
image_size = x.shape
xy = np.hstack((x.reshape(x.shape[0]*x.shape[1],1,order='F'), y.reshape(y.shape[0]*y.shape[1] , 1, order='F')))


pred_label = []
for pixel in xy :
    if (-pixel[0] - pixel[1] +5 >0) and (-pixel[0] +3 > 0) :
        pred_label.append(1)
    elif (-pixel[0] +pixel[1] - 1 > 0) and (-pixel[0]-pixel[1]+5 <0) :
        pred_label.append(2)
    elif  (-pixel[0] + 3 <0) and (-pixel[0] + pixel[1] -1<0)  :
        pred_label.append(3)
    else:
        pred_label.append(0)

pred_label = np.array(pred_label)
decisionmap = pred_label.reshape(image_size,order='F')
print(decisionmap)

plt.imshow(decisionmap, extent = [xrange[0],xrange[1],yrange[0],yrange[1]], origin = 'lower' )
plt.plot(4,1 , 'rx')
plt.plot(1,5 , 'go')
plt.plot(0,0 , 'b*')
l1=plt.legend(( '[4,1] Class 3' , '[1,5] Class 2' , '[0,0] Class 1') , loc = 3)
plt.gca().add_artist(l1)
TestErrorRate = 'Error rate' + str(122.32)
l2=plt.legend((TestErrorRate, 'Error rate'),loc=2)
plt.gca().add_artist(l2)
plt.plot()
plt.show()
