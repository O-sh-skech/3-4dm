from dataset1 import load_dataset_from_tsv
import regression
import numpy as np


# データ読み込み
df = load_dataset_from_tsv()

x = df["観測点"].values.reshape(-1, 1)
y = df["観測値"].values

# 8割を学習用
n = len(x)
idx = np.random.permutation(n)

train_size = int(0.8 * n)
train_idx = idx[:train_size]

x_train, y_train = x[train_idx], y[train_idx]

import regression
model = regression.LinearRegression()
model.fit(x_train, y_train)

print(model.theta)