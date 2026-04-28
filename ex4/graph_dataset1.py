import matplotlib.pyplot as plt
import numpy as np
from dataset1 import true_function
from dataset1 import sample_x

import os
import sys

filename = "ex1.2.png"

# 線用データ
x = np.linspace(-1, 1, 100)
y = true_function(x)
# サンプル点
sam_x = sample_x()
sam_y = true_function(sam_x)

plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
plt.scatter(sam_x, sam_y, color="red", label="Samples")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True Function")

if os.path.exists(f"/Users/uta/datamining/ex4/{filename}"):
    print("ファイルが存在するので終了")
    sys.exit()
    

plt.savefig(f"/Users/uta/datamining/ex4/{filename}")
plt.show()