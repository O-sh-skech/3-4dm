from pathlib import Path

# このファイルの場所
base_dir = Path(__file__).resolve().parent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def true_function(x):
    return np.sin(np.pi * x * 0.8) * 10

def sample_x(n=20, seed=0):
    np.random.seed(seed)
    return np.random.uniform(-1, 1, n)


x = np.linspace(-1, 1, 100)
y = true_function(x)
sam_x = sample_x()
sam_y = true_function(sam_x)
noise = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=len(sam_x))
noise = noise / 2
y_observed = sam_y + noise

df = pd.DataFrame({
    "観測点": sam_x,
    "真値": sam_y
})

df["観測値"] = y_observed

def plot_true_function():#1.1
    # 線用データ
    plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Function")
    plt.savefig(f"{base_dir}/ex1.1.png")
    plt.show()



def plot_samples():#1.2
    plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
    plt.scatter(sam_x, sam_y, color="red", label="Samples")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Function")
    plt.savefig(f"{base_dir}/ex1.2.png")
    plt.show()



def plot_noisy_samples():
    plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
    plt.scatter(sam_x, sam_y, color="red", label="Samples")
    plt.scatter(sam_x, y_observed, label="Observed (Noisy)", alpha=0.7)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Function")
    plt.savefig(f"{base_dir}/ex1.3.png")
    plt.show()



def save_dataset_to_tsv():
    df.to_csv(f"{base_dir}/data.tsv", sep="\t", index=False)




def load_dataset_from_tsv():
    return pd.read_csv(f"{base_dir}/data.tsv", sep="\t")



if __name__ == "__main__":
    plot_true_function()
    plot_samples()
    plot_noisy_samples()
    save_dataset_to_tsv()
    print(load_dataset_from_tsv())