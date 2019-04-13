import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

input_file = "numerical_features_moreErrors.csv"
df = pd.read_csv(input_file, header = 0)
#df = (df - df.min())/(df.max() - df.min())   # Data Normalization
original_headers = list(df.columns.values)
numpy_matrix = df.as_matrix()
np.set_printoptions(linewidth = 200)
pd.set_option('expand_frame_repr',False)
pd.set_option('display.max_rows',365)
print 'Numpy Matrix: '
print df
print('\n')
'''
x = numpy_matrix[:,[3]]
print 'x-values: '
print(x)
print('\n')
y = numpy_matrix[:,[4]]
print 'y-values: '
print(y)
print('\n')
'''

c = 0
sum_of_coeff = 0
array_of_coeff = []
for item in numpy_matrix[:5]:
    coeff = np.corrcoef(numpy_matrix[:, c], numpy_matrix[:, 5])[0, 1]
    array_of_coeff.append(coeff)
    print 'Correlation coeff of feature ',c,': ',coeff
    sum_of_coeff = sum_of_coeff + coeff
    c = c + 1
print '\n'

f = 0
array_of_weight = []
for coeff in array_of_coeff:
    weight = (coeff/sum_of_coeff)
    if weight < 0:
        print 'Weight of feature ', f, ': ', abs(weight)
    else:
        print 'Weight of feature ',f,': ',weight
    array_of_weight.append(weight)
    f = f + 1
print '\n'


a = 0
weighted_sum_of_features = []
for feature in numpy_matrix[:5]:
    feature_a = numpy_matrix[:,a]
    b = 0
    for item in feature_a:
        weighted_item = item * array_of_weight[a]
        #print weighted_item
        if a == 0:
            weighted_sum_of_features.append(weighted_item)
        else:
            weighted_sum_of_features[b] = weighted_sum_of_features[b] + weighted_item
            b = b + 1
    a = a + 1

print 'Weighted sum of features: '
for item in weighted_sum_of_features:
    print item

x = np.array(weighted_sum_of_features)
y = np.array(numpy_matrix[:,5])

x = x.reshape(-1,1)
y = y.reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(x,y)
print 'Mean Squared Error: ',np.mean((regr.predict(x) - y)**2)
fig = plt.figure()
fig.suptitle('Training by linear regression', fontsize = 16)
plt.scatter(x, y,  color='black')
plt.plot(x,regr.predict(x), color='blue',linewidth=3)
plt.xlabel('Sum of weighted features')
plt.ylabel('Score')
plt.show()



