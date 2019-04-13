import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

input_file = "numerical_features.csv"
df = pd.read_csv(input_file, header = 0)
numpy_array = df.as_matrix()
np.set_printoptions(linewidth = 200)

#print 'Numpy Matrix: '
#print numpy_array
#print('\n')

x = numpy_array[:,3]
y = numpy_array[:,4]

print x
print y

print np.corrcoef(x,y)[0,1]

