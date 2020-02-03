
def extract2FeatureArray(training,f1,f2) :
    return training[:,[f1,f2]]



def comp2DSampleMean2Array(training) :
    sum =np.array( [0 , 0])
    count = 0
    for data in training :
        sum += data
        count += 1
    return sum/count