import math
import matplotlib.pyplot as pyplt
import numpy as np
from scipy import stats
from scipy import optimize


def get_table_func(sample):
    seq = sorted(sample)
    n = len(seq)
    acc = 0.
    table = {'x': seq, 'y': []}
    for value in seq:
        acc += 1 / n
        table['y'].append(acc)
    return table


def data_mean(data: list):
    n = len(data)
    acc = 0.
    for value in data:
        acc += value / n
    return acc


def data_sqr_mean(data: list):
    n = len(data)
    acc = 0.
    for value in data:
        acc += (value ** 2) / n
    return acc


def data_dispersion(data: list = None, *, mean=None, sqr_mean=None):
    if (mean is None or sqr_mean is None) and data is None:
        raise ValueError('Not enough data for dispersion')
    if mean is None:
        mean = data_mean(data)
    if sqr_mean is None:
        sqr_mean = data_sqr_mean(data)
    return sqr_mean - mean ** 2


def data_median(data: list):
    seq = sorted(data)
    n = len(seq)
    if n % 2 == 1:
        return seq[n // 2]
    else:
        return (seq[n // 2 - 1] + seq[n // 2]) / 2


def seq_median(seq: list):
    n = len(seq)
    if n % 2 == 1:
        return seq[n // 2]
    else:
        return (seq[n // 2 - 1] + seq[n // 2]) / 2


def sample_z_r(sample: list):
    return (min(sample) + max(sample)) / 2


def seq_quartile(seq, p):
    n_p = p * len(seq)
    if p.is_integer():
        return seq[math.floor(n_p) - 1]
    else:
        return seq[math.floor(n_p)]


def seq_z_q(seq):
    return (seq_quartile(seq, 1. / 4) + seq_quartile(seq, 3. / 4)) / 2


def seq_z_tr(seq):
    size = len(seq)
    r = int(size / 4)
    sum = 0
    for i in range(r, size - r):
        sum += seq[i]
    return sum / (size - 2 * r)


def outliers_bounds(data):
    q_1, q_3 = np.quantile(data, [0.25, 0.75])
    return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)


def number_of_outliers(data):
    x1, x2 = outliers_bounds(data)
    filtered = [x for x in data if x > x2 or x < x1]
    return len(filtered)


def pearson_correlation_coefficient(x, y):
    return stats.pearsonr(x, y)[0]


def spearman_rank_correlation_coefficient(x, y):
    return stats.spearmanr(x, y)[0]


def quadrant_correlation_coefficient(x, y):
    size = len(x)
    med_x = data_median(x)
    med_y = data_median(y)
    n = sum(map(lambda _x, _y: 1 if (_x - med_x)*(_y - med_y) >= 0 else -1, x, y))
    return n / size


def absolute_deviations_from_linear(linear_params: tuple[float, float], x, y):
    beta_0, beta_1 = linear_params
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - beta_0 - beta_1 * x[i])
    return sum


def least_squares_regression(x, y) -> tuple[float, float]:
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def least_absolute_deviations_regression(x, y) -> tuple[float, float]:
    start_params = least_squares_regression(x, y)
    result = optimize.minimize(absolute_deviations_from_linear, start_params, args=(x, y), method='SLSQP')
    alpha_0, alpha_1 = result.x
    return alpha_0, alpha_1
