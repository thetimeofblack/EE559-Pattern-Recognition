import numpy as np
from python3.plotDecBoundaries import plotDecBoundaries
from tools.nearest_centroid_classifier import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def f1(x,y):
    return -x-y+5
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
print(xrange)

(x, y) = np.meshgrid(np.arange(xrange[0],xrange[1]+inc/100 ,inc),np.arange(yrange[0],yrange[1]+inc/100 ,inc))
print(x)
image_size = x.shape
#print(x.shape)
#print(x.reshape(x.shape[0]*x.shape[1],1,order='F'))
xy = np.hstack((x.reshape(x.shape[0]*x.shape[1],1,order='F'), y.reshape(y.shape[0]*y.shape[1] , 1, order='F')))
print(xy.shape)
#print(xy)

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
print(pred_label)
decisionmap = pred_label.reshape(image_size,order='F')
print(decisionmap)
plt.imshow(decisionmap, extent = [xrange[0],xrange[1],yrange[0],yrange[1]], origin = 'lower' )
plt.plot(4,1 , 'rx')
plt.plot(1,5 , 'go')
plt.plot(0,0 , 'b*')
plt.legend(( 'Class 1' , 'Class 2' , 'Class 3') , loc = 3)
plt.show()
