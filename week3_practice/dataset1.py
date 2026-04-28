
import matplotlib.pyplot as plt
import numpy as np

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10


x = np.linspace(-1, 1, 100)
y = true_function(x)
plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True Function")
plt.savefig("week3_practice/ex1.1.png")
plt.show()