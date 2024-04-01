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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------
dim = 5
n_samples = 10

# %% ---- 2024-04-01 ------------------------
# Function and class


def js_estimate(x):
    alpha = dim-2
    k = (1-alpha / np.dot(x, x))
    return k * x


def _generate_multiple_normal_dist(dim: int = dim, n_samples: int = n_samples, k: float = 10.0):
    mean = np.random.random((dim,)) * k
    cov = np.eye(dim)
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    return mean, samples


def generate_multiple_normal_dist(dim: int = dim, n_samples: int = n_samples, x2: float = 10.0):
    mean = np.random.random((dim,))
    mean /= np.linalg.norm(mean)
    mean *= np.sqrt(x2)

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
results = []
x2_values = np.concatenate([np.linspace(.1, 1, 10), np.linspace(1, 10, 10)])
for x2 in x2_values:
    for j in range(100):
        mean, samples = generate_multiple_normal_dist(x2=x2)
        estimation = np.array([js_estimate(e) for e in samples])
        risk = measure_risks(mean, samples, estimation)
        for est in ['mle', 'js']:
            results.append(dict(risk=risk[est], estimate=est, x2=risk['x2']))
results = pd.DataFrame(results)
print(results)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.boxplot(results, x='x2', y='risk', hue='estimate', notch=True, ax=ax)
labels = [e for e in ax.get_xticklabels() if len(e.get_text()) < 5]
ax.set_xticklabels([e.get_text() for e in labels])
ax.set_xticks([e.get_position()[0] for e in labels])
plt.show()

# %%


# %% ---- 2024-04-01 ------------------------
# Pending
# print(np.mean([np.dot(e, e) for e in samples-mean]))
# print(np.mean([np.dot(e, e) for e in estimation-mean]))


# %% ---- 2024-04-01 ------------------------
# Pending

# %%
