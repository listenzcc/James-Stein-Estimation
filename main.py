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


# %% ---- 2024-04-01 ------------------------
# Function and class


def js_estimate(x):
    """
Estimates the James-Stein estimator for a given input vector.

Args:
    x: The input vector.

Returns:
    The estimated James-Stein estimator.

Examples:
    >>> js_estimate([1, 2, 3])
    array([-0.5, -1. , -1.5])
"""

    alpha = dim-2
    k = (1-alpha / np.dot(x, x))
    return k * x


def generate_multiple_normal_dist(dim: int = 5, n_samples: int = 100, x2: float = 10.0):
    """
Generates multiple samples from a multivariate normal distribution.

Args:
    dim (int): The dimension of the distribution. Defaults to 5.
    n_samples (int): The number of samples to generate. Defaults to 100.
    x2 (float): The variance of the distribution. Defaults to 10.0.

Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the mean vector and the generated samples.

Examples:
    >>> generate_multiple_normal_dist(dim=3, n_samples=10, x2=5.0)
    (array([-0.123, 0.456, -0.789]), array([[ 0.123, -0.456,  0.789],
                                            [ 0.321, -0.654,  0.987],
                                            ...
                                            [ 0.789, -0.123,  0.456]]))
"""

    mean = np.random.random((dim,))
    mean /= np.linalg.norm(mean)
    mean *= np.sqrt(x2)

    cov = np.eye(dim)
    samples = np.random.multivariate_normal(mean, cov, size=n_samples)
    return mean, samples


def measure_risks(mean, samples, estimation):
    """
Measures the risks of different estimators.

Args:
    mean (np.ndarray): The mean vector.
    samples (np.ndarray): The generated samples.
    estimation (np.ndarray): The estimated values.

Returns:
    dict: A dictionary containing the risks of different estimators, including MLE, JS, and X2.

Examples:
    >>> mean = np.array([1, 2, 3])
    >>> samples = np.array([[1, 2, 3], [4, 5, 6]])
    >>> estimation = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
    >>> measure_risks(mean, samples, estimation)
    {'MLE': 0.5, 'JS': 0.5, 'X2': 14}
"""

    x2 = np.dot(mean, mean)
    risk_mle = np.mean([np.dot(e, e) for e in samples-mean])
    risk_js = np.mean([np.dot(e, e) for e in estimation-mean])
    return dict(
        MLE=risk_mle,
        JS=risk_js,
        X2=x2
    )


# %% ---- 2024-04-01 ------------------------
# Play ground
if __name__ == '__main__':

    # --------------------
    dim = 10
    n_repeat = 100

    results = []
    x2_values = np.concatenate(
        [np.linspace(.2, 1, 5), np.linspace(2, 10, 5)])
    for x2 in x2_values:
        for _ in range(100):
            mean, samples = generate_multiple_normal_dist(dim, n_repeat, x2=x2)
            estimation = np.array([js_estimate(e) for e in samples])
            risk = measure_risks(mean, samples, estimation)
            results.extend(
                dict(Risk=risk[est], Estimate=est, X2=risk['X2'])
                for est in ['MLE', 'JS']
            )
    results = pd.DataFrame(results)
    print(results)

    # --------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.boxplot(results, x='X2', y='Risk', hue='Estimate', notch=True, ax=ax)
    labels = [e for e in ax.get_xticklabels() if len(e.get_text()) < 5]
    sns.move_legend(ax, "lower right")
    ax.set_ylim([0, dim*1.5])
    ax.set_xticklabels([e.get_text() for e in labels])
    ax.set_xticks([e.get_position()[0] for e in labels])
    ax.grid()
    ax.set_title(f'MLE & JS Estimation at dim: {dim} | repeats: {n_repeat}')
    fig.tight_layout()
    plt.show()

# %% ---- 2024-04-01 ------------------------
# Pending


# %% ---- 2024-04-01 ------------------------
# Pending

# %%
