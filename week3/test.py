import datasets
import numpy as np
X,Y = datasets.load_linear_example1()
print(X)
print(X[0])
print(Y)

Z= np.dot(Y,X)
print(Z)

import regression
model = regression.LinearRegression()
model.fit(X,Y)
print(model.theta)

print(model.predict(X))