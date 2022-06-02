import numpy as np
import math
import matplotlib.pyplot as pyplt
import seaborn as sb


def upscale_range_to_sample(sample: list, min_range):
    low = min([min(sample), min_range[0]])
    high = max([max(sample), min_range[1]])
    return low, high


def plot_sample_hist(axis: pyplt.Axes, plot_range: tuple[float, float], sample: list, *, bins_width: float = -1,
                     bins_count: int = 10, **kwargs):
    plot_range_len = plot_range[1] - plot_range[0]
    if bins_width != -1:
        bins_count = math.ceil(plot_range_len / bins_width)
    axis.hist(sample, range=plot_range, bins=bins_count, density=True, edgecolor='blue',
              facecolor=(0.3, 0.3, 1.0, 0.1), linewidth=0.4, **kwargs)


def plot_func(axis: pyplt.Axes, plot_range: tuple[float, float], points_count: int, func):
    if points_count == -1:
        x = np.arange(math.floor(plot_range[0]), math.ceil(plot_range[1]) + 1)
    else:
        x = np.linspace(plot_range[0], plot_range[1], points_count)
    y = func(x)
    axis.plot(x, y, 'r')


def plot_step_func(axis: pyplt.Axes, plot_range: tuple[float, float], func):
    x = np.arange(math.floor(plot_range[0]), math.ceil(plot_range[1]) + 1)
    y = func(x)
    axis.step(x, y, 'r', where='post')


def draw_boxplot(axis, data, labels):
    axis.boxplot(x=data, vert=False, labels=labels, widths=0.8)
    return


def plot_kde(axis, sample, factor):
    sb.kdeplot(data=sample, bw_method='silverman', bw_adjust=factor, ax=axis, fill=True, common_norm=False,
               palette="crest", alpha=0, linewidth=2, label='kde')


def plot_table_func(axis: pyplt.Axes, plot_range: tuple[float, float], table, low_value, high_value):
    x_grid = []
    y_grid = []
    i = 0
    while i < len(table['x']) and table['x'][i] < plot_range[0]:
        i += 1
    if i == len(table['x']):
        return
    x_grid.append(plot_range[0])
    if i == 0:
        y_grid.append(low_value)
    else:
        y_grid.append(table['y'][i - 1])
    while i < len(table['x']) and table['x'][i] < plot_range[1]:
        x_grid.append(table['x'][i])
        y_grid.append(table['y'][i])
        i += 1
    x_grid.append(plot_range[1])
    if i == len(table['x']):
        y_grid.append(high_value)
    else:
        y_grid.append(table['y'][i])

    axis.step(x_grid, y_grid, 'b', where='post')
