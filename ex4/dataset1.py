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


def create_dataset(n=20, seed=0):
    sam_x = sample_x(n, seed)
    sam_y = true_function(sam_x)

    noise = np.random.normal(0.0, np.sqrt(2.0), size=len(sam_x)) / 2
    y_observed = sam_y + noise

    df = pd.DataFrame({
        "観測点": sam_x,
        "真値": sam_y,
        "観測値": y_observed
    })
    return df

def plot_true_function():#1.1
    # 線用データ
    x = np.linspace(-1, 1, 100)
    y = true_function(x)
    plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Function")
    plt.savefig(f"{base_dir}/ex1.1.png")
    plt.show()



def plot_samples(df):#1.2
    x = np.linspace(-1, 1, 100)
    y = true_function(x)

    plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
    plt.scatter(df["観測点"], df["真値"], color="red", label="Samples")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Function")
    plt.savefig(f"{base_dir}/ex1.2.png")
    plt.show()



def plot_noisy_samples(df):
    x = np.linspace(-1, 1, 100)
    y = true_function(x)

    plt.plot(x, y, label="y = sin(pi * 0.8 * x) * 10")
    plt.scatter(df["観測点"], df["真値"], color="red", label="Samples")
    plt.scatter(df["観測点"], df["観測値"], label="Observed (Noisy)", alpha=0.7)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Function")
    plt.savefig(f"{base_dir}/ex1.3.png")
    plt.show()



def save_dataset_to_tsv(df):
    df.to_csv(f"{base_dir}/data.tsv", sep="\t", index=False)




def load_dataset_from_tsv():
    return pd.read_csv(f"{base_dir}/data.tsv", sep="\t")



if __name__ == "__main__":
    df = create_dataset(n=20, seed=0)
    plot_true_function()
    plot_samples(df)
    plot_noisy_samples(df)
    save_dataset_to_tsv(df)
    print(load_dataset_from_tsv())