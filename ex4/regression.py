import numpy as np

class LinearRegression:
    x = None
    theta = None
    y= None

    def fit(self, x, y):#1.9 シータの本体
       temp = np.linalg.inv(np.dot(x.T,x))
       self.theta = np.dot(np.dot(temp,x.T),y)

    def predict(self, x):#予測値そのもの
        return np.dot(x,self.theta)

    def score(self, x, y):#誤差計算
        pass