"""
File: main.py
Author: Chuncheng Zhang
Date: 2024-04-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Demo of James Stein estimation

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-01 ------------------------
# Requirements and constants
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------
dim = 5
n_samples = 10

# %% ---- 2024-04-01 ------------------------
# Function and class


def js_estimate(x):
    alpha = dim-2
    # k = (1-alpha / np.power(np.linalg.norm(x), 2))
    k = (1-alpha / np.dot(x, x))  # (np.linalg.norm(x), 2))
    return k * x


def generate_multiple_normal_dist(dim: int = dim, n_samples: int = n_samples, k: float = 10.0):
    mean = np.random.random((dim,)) * k
    cov = np.eye(dim)
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    return mean, samples


def measure_risks(mean, samples, estimation):
    x2 = np.dot(mean, mean)
    risk_mle = np.mean([np.dot(e, e) for e in samples-mean])
    risk_js = np.mean([np.dot(e, e) for e in estimation-mean])
    return dict(
        mle=risk_mle,
        js=risk_js,
        x2=x2
    )


# %% ---- 2024-04-01 ------------------------
# Play ground
mean, samples = generate_multiple_normal_dist()
estimation = np.array([js_estimate(e) for e in samples])
risk = measure_risks(mean, samples, estimation)
print(risk)

# %%
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
ax = axs[0]
sns.heatmap(samples-mean, cmap='RdBu', ax=ax)
ax = axs[1]
sns.heatmap(estimation-mean, cmap='RdBu', ax=ax)
plt.show()


# %% ---- 2024-04-01 ------------------------
# Pending
# print(np.mean([np.dot(e, e) for e in samples-mean]))
# print(np.mean([np.dot(e, e) for e in estimation-mean]))


# %% ---- 2024-04-01 ------------------------
# Pending

# %%
