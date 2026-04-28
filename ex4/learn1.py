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

x = np.linspace(-1, 1, 100)
y = true_function(x)
x_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_plot)

plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
plt.scatter(df["観測点"], df["観測値"], label="Observed (Noisy)", alpha=0.7)
plt.scatter(x_plot, y_pred, label="predicted", alpha=0.7)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True Function")
plt.savefig("ex4/ex1.10.png")
plt.show()
