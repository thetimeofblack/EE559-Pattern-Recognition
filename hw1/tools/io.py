import numpy as np 
def readCSVFile(filename,data):
	data = np.genfromtxt(filename,delimiter=',') 
	