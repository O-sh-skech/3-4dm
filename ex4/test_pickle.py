import pickle
from dataset1 import load_dataset_from_tsv, true_function
import matplotlib.pyplot as plt
import numpy as np


# モデル読み込み


with open("ex4/model.pkl", "rb") as f:
    model = pickle.load(f)


# データ読み込み

df = load_dataset_from_tsv()

x_data = df["観測点"].values.reshape(-1, 1)
y_data = df["観測値"].values




# 描画


x_plot = np.linspace(-1, 1, 100)

# 真の関数
y_true = true_function(x_plot)

# モデル予測
x_plot_2d = x_plot.reshape(-1, 1)
y_pred = model.predict(x_plot_2d)

# MAE確認


mae = model.MAE_score(x_plot_2d, y_true)
#2.482程度。ほぼ一致
print(f"MAE = {mae:.3f}")

# グラフ表示


plt.plot(x_plot, y_true, label="True Function")

plt.scatter(
    df["観測点"],
    df["観測値"],
    label="Observed Data",
    alpha=0.7
)

plt.plot(x_plot, y_pred, label="Predicted")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"MAE = {mae:.3f}")

plt.show()