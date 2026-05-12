from dataset1 import load_dataset_from_tsv, true_function
import regression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# データ読み込み
df = load_dataset_from_tsv()

x = df["観測点"].values.reshape(-1, 1)
y = df["観測値"].values

# 8割を学習用
n = len(x)
idx = np.random.permutation(n)

train_size = int(0.8 * n)
train_idx = idx[:train_size]
test_idx = idx[train_size:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

import regression
model = regression.LinearRegression()
model.fit(x_train, y_train)

#print(model.theta)

# 予測
y_pred = model.predict(x_test)

#print(y_pred)

x_ = np.linspace(-1, 1, 100)
y_ = true_function(x_)
#真の関数を表示するだけのx_,y_
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_plot)
print(y_pred)

#MEAテスト
MAE_score = model.MAE_score(x_plot,y_)
print(f"{MAE_score=}")


plt.plot(x_, y_, label="y = sin(pi * 0.8 * x) * 10")
plt.scatter(df["観測点"], df["観測値"], label="Observed (Noisy)", alpha=0.7)
plt.scatter(x_plot, y_pred, label="predicted", alpha=0.7)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True Function")
plt.savefig("ex4/ex1.10.png")
plt.show()

'''
考察
MAE_score=2.446822274808775
から、平均で ±2.45 くらい外してると読み取れる。
計算したモデルが一次関数によるものなので、真の関数に対しては右肩上がりの傾向を示すことはできたものの
やはり曲線による出力結果とはずれが大きくなる。
'''
