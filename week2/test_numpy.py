import numpy as np

# 行列の作成
a = np.array([[1,2,3], [4,5,6]])

#行の参照
a[0]
#転置行列
a.T

#(1) 5個の要素を持つ列ベクトルを作成せよ。値は全て1とする。
b = np.array([1,1,1,1,1], dtype=float)
print(b)
#(2) 1で作成した列ベクトルのうち、2番目の要素を3.14に更新せよ。なおインデックスは0から数える（0番目、1番目、2番目、、）ものとする。
b[2]=3.14
print(b)
#(3)
b.T
print(b)
#(4)
print(np.dot(b, b.T))
#(5)
print(np.random.rand(1,10))
#(6)
mu, sigma = 10, 2 # mean and standard deviation
s = np.random.normal(mu, sigma, size=(5, 2))
print(s)
#(7)
print(s[1])
#(8)
print(s[2:4])
#(10)
t=np.random.rand(2,5)
print(np.dot(s, t))