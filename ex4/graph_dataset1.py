import matplotlib.pyplot as plt
import numpy as np
from dataset1 import true_function
from dataset1 import sample_x
import pandas as pd

import os
import sys

filename = "ex1.3.png"

# 線用データ
x = np.linspace(-1, 1, 100)
y = true_function(x)

# サンプル点
sam_x = sample_x()
sam_y = true_function(sam_x)

df = pd.DataFrame({
    "観測点": sam_x,
    "真値": sam_y
})

# ノイズ生成（平均0, 分散2 → 標準偏差は sqrt(2)）
noise = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=len(sam_x))
noise = noise / 2
y_observed = sam_y + noise

df["観測値"] = y_observed

df.to_csv("/Users/uta/datamining/ex4/data.tsv", sep="\t", index=False)
df = pd.read_csv("/Users/uta/datamining/ex4/data.tsv", sep="\t")

#-----------グラフ処理---------------

plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
plt.scatter(sam_x, sam_y, color="red", label="Samples")
plt.scatter(sam_x, y_observed, label="Observed (Noisy)", alpha=0.7)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True Function")

if os.path.exists(f"/Users/uta/datamining/ex4/{filename}"):
    print("ファイルが存在するので終了")
    sys.exit()
    

plt.savefig(f"/Users/uta/datamining/ex4/{filename}")
plt.show()

