
import matplotlib.pyplot as plt
import numpy as np

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

def sample_x(n=20, seed=0):
    np.random.seed(seed)
    return np.random.uniform(-1, 1, n)

x = sample_x()
y = true_function(x)

import pandas as pd

df = pd.DataFrame({
    "観測点": x,
    "真値": y
})