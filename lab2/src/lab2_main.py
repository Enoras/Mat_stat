import math

from tabulate import tabulate
from matplotlib import pyplot as pyplt, transforms
from matplotlib.patches import Ellipse
import numpy as np
import scipy.stats as stats
import distributions as dist
import characteristics_utils as ch_utils
from plot_utils import plot_sample_hist, upscale_range_to_sample


def lab_2_part_correlation_coefficient():
    print_lab_log('* part_correlation_coefficient:\n')

    repeat_count = 1000

    sample_size_list = (20, 60, 100)
    cor_list = (0, 0.5, 0.9)
    cor_estimate_list = (
        {
            'name': 'Pearson',
            'designation': '$r$',
            'func': ch_utils.pearson_correlation_coefficient
        },
        {
            'name': 'Spearman',
            'designation': '$r_{S}$',
            'func': ch_utils.spearman_rank_correlation_coefficient
        },
        {
            'name': 'Quadrant',
            'designation': '$r_{Q}$',
            'func': ch_utils.quadrant_correlation_coefficient
        },
    )

    for sample_size in sample_size_list:
        print_lab_log('- experiment \'Normal 2d, size = {}\':'.format(sample_size))
        table_rows = []
        for rho in cor_list:
            cor_estimates_data = [[] for cor_estimate in cor_estimate_list]
            for j in range(repeat_count):
                sample = dist.get_normal_2d_sample(sample_size, rho)
                x = list(map(lambda i: sample[i][0], range(sample_size)))
                y = list(map(lambda i: sample[i][1], range(sample_size)))
                for i in range(len(cor_estimate_list)):
                    cor_estimates_data[i].append(cor_estimate_list[i]['func'](x, y))
            header_row = ['${{\\rho}} = {0}$'.format(rho)]
            for cor_estimate in cor_estimate_list:
                header_row.append(cor_estimate['designation'])
            table_rows.append(header_row)
            row_mean = ['$E(z)$']
            row_sqr_mean = ['$E(z^{2})$']
            row_dispersion = ['$D(z)$']
            row_empty = ['' for i in range(len(cor_estimate_list) + 1)]
            for i in range(len(cor_estimate_list)):
                mean = ch_utils.data_mean(cor_estimates_data[i])
                sqr_mean = ch_utils.data_sqr_mean(cor_estimates_data[i])
                dispersion = ch_utils.data_dispersion(mean=mean, sqr_mean=sqr_mean)
                row_mean.append('{: .4f}'.format(mean))
                row_sqr_mean.append('{: .4f}'.format(sqr_mean))
                row_dispersion.append('{: .4f}'.format(dispersion))
            table_rows.append(row_mean)
            table_rows.append(row_sqr_mean)
            table_rows.append(row_dispersion)
            table_rows.append(row_empty)
        print_lab_log(tabulate(table_rows, tablefmt="latex_raw"))
        print_lab_log('')

    print_lab_log('- experiment \'Normal mix 2d\':')
    table_rows = []
    cor_estimates_data = [[] for cor_estimate in cor_estimate_list]
    for sample_size in sample_size_list:
        for j in range(repeat_count):
            sample = dist.get_mix_normal_2d_sample(sample_size)
            x = list(map(lambda i: sample[i][0], range(sample_size)))
            y = list(map(lambda i: sample[i][1], range(sample_size)))
            for i in range(len(cor_estimate_list)):
                cor_estimates_data[i].append(cor_estimate_list[i]['func'](x, y))
        header_row = ['$n = {0}$'.format(sample_size)]
        for cor_estimate in cor_estimate_list:
            header_row.append(cor_estimate['designation'])
        table_rows.append(header_row)
        row_mean = ['$E(z)$']
        row_sqr_mean = ['$E(z^{2})$']
        row_dispersion = ['$D(z)$']
        row_empty = ['' for i in range(len(cor_estimate_list) + 1)]
        for i in range(len(cor_estimate_list)):
            mean = ch_utils.data_mean(cor_estimates_data[i])
            sqr_mean = ch_utils.data_sqr_mean(cor_estimates_data[i])
            dispersion = ch_utils.data_dispersion(mean=mean, sqr_mean=sqr_mean)
            row_mean.append('{: .4f}'.format(mean))
            row_sqr_mean.append('{: .4f}'.format(sqr_mean))
            row_dispersion.append('{: .4f}'.format(dispersion))
        table_rows.append(row_mean)
        table_rows.append(row_sqr_mean)
        table_rows.append(row_dispersion)
        table_rows.append(row_empty)
    print_lab_log(tabulate(table_rows, tablefmt="latex_raw"))
    print_lab_log('')


def get_scattering_ellipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ellipse


def lab_2_part_scattering_ellipse():
    sample_size_list = (20, 60, 100)
    cor_list = (0, 0.5, 0.9)
    for sample_size in sample_size_list:
        fig_file_name = 'lab_2/part_scattering_ellipse/figures/n_{}.png'.format(sample_size)
        fig, ax = pyplt.subplots(1, 3)
        size_str = "n = " + str(sample_size)
        titles = [size_str + r', $ \rho = 0$', size_str + r', $\rho = 0.5 $', size_str + r', $ \rho = 0.9$']
        for i in range(len(cor_list)):
            num, ro = i, cor_list[i]
            sample = dist.get_normal_2d_sample(sample_size, ro)
            x, y = sample[:, 0], sample[:, 1]
            ellipse = get_scattering_ellipse(x, y, ax[num], edgecolor='green')
            ax[num].add_patch(ellipse)
            ax[num].grid()
            ax[num].scatter(x, y, s=5)
            ax[num].set_title(titles[num])
        fig.savefig(fig_file_name)


def lab_2_part_linear_regression():
    print_lab_log('* part_linear_regression:\n')

    regression_methods_list = (
        {
            'name': 'Least squares',
            'func': ch_utils.least_squares_regression,
            'format': 'r'
        },
        {
            'name': 'Least absolute deviations',
            'func': ch_utils.least_absolute_deviations_regression,
            'format': 'g'
        },
    )
    experiment_list = (
        {
            'name': 'no_error',
            'get_dist_sample': dist.get_norm_sample,
            'func_range': (-1.8, 2.),
            'points_count': 20,
            'a': 2.,
            'b': 2.,
            'errors_list': []
        },
        {
            'name': 'ends_error',
            'get_dist_sample': dist.get_norm_sample,
            'func_range': (-1.8, 2.),
            'points_count': 20,
            'a': 2.,
            'b': 2.,
            'errors_list': [(0, +10.), (19, -10.)]
        },
    )

    for experiment in experiment_list:
        print_lab_log('- experiment \'{0}\':'.format(experiment['name']))
        fig_file_name = 'lab_2/part_linear_regression/figures/' + experiment['name'] + '.png'
        print_lab_log('  figure: \'{0}\''.format(fig_file_name))
        fig, axis = pyplt.subplots(1, 1, figsize=(6, 4))
        func_range = experiment['func_range']
        points_count = experiment['points_count']
        x = np.linspace(func_range[0], func_range[1], points_count)
        err = experiment['get_dist_sample'](points_count)
        a = experiment['a']
        b = experiment['b']
        clear_y = [a * x[i] + b for i in range(points_count)]
        y = [clear_y[i] + err[i] for i in range(points_count)]
        for error_point in experiment['errors_list']:
            y[error_point[0]] += error_point[1]

        print_lab_log('  clear dependency: a = {0}; b = {1};'.format(a, b))

        axis.plot(x, clear_y, ':b', label='clear dependency')

        axis.plot(x, y, 'ob', label='noisy data', markerfacecolor='none')

        for method in regression_methods_list:
            reg_a, reg_b = method['func'](x, y)
            reg_y = [reg_a * x[i] + reg_b for i in range(points_count)]
            print_lab_log('  {2}: a = {0}; b = {1};'.format(reg_a, reg_b, method['name']))
            axis.plot(x, reg_y, method['format'], label=method['name'])
        axis.grid()
        axis.legend()
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        fig.savefig(fig_file_name)
        print_lab_log('')


def get_k(size):
    return math.ceil(1.72 * (size) ** (1 / 3))


def calculate(distribution, p, k):
    mu = np.mean(distribution)
    sigma = np.std(distribution)

    print_lab_log('mu = ' + str(np.around(mu, decimals=2)))
    print_lab_log('sigma = ' + str(np.around(sigma, decimals=2)))

    limits = np.linspace(-1.1, 1.1, num=k - 1)
    chi_2 = stats.chi2.ppf(p, k - 1)
    print_lab_log('chi_2 = ' + str(chi_2))
    return limits


def get_n_and_p(distribution, limits, size):
    p_list = np.array([])
    n_list = np.array([])

    for i in range(-1, len(limits)):
        if i != -1:
            prev_cdf_val = dist.norm_func(limits[i])
        else:
            prev_cdf_val = 0
        if i != len(limits) - 1:
            cur_cdf_val = dist.norm_func(limits[i + 1])
        else:
            cur_cdf_val = 1
        p_list = np.append(p_list, cur_cdf_val - prev_cdf_val)
        if i == -1:
            n_list = np.append(n_list, len(distribution[distribution <= limits[0]]))
        elif i == len(limits) - 1:
            n_list = np.append(n_list, len(distribution[distribution >= limits[-1]]))
        else:
            n_list = np.append(n_list, len(distribution[(distribution <= limits[i + 1]) & (distribution >= limits[i])]))

    result = np.divide(np.multiply((n_list - size * p_list), (n_list - size * p_list)), p_list * size)
    return n_list, p_list, result


def create_table(n_list, p_list, result, size, limits):
    cols = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "/frac{(n_i-np_i)^2}{np_i}"]
    rows = []
    for i in range(0, len(n_list)):
        if i == 0:
            boarders = ['-inf', np.around(limits[0], decimals=2)]
        elif i == len(n_list) - 1:
            boarders = [np.around(limits[-1], decimals=2), 'inf']
        else:
            boarders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]

        rows.append(
            [i + 1, boarders, n_list[i], np.around(p_list[i], decimals=4), np.around(p_list[i] * size, decimals=2),
             np.around(n_list[i] - size * p_list[i], decimals=2), np.around(result[i], decimals=2)])

    rows.append([len(n_list), "-", np.sum(n_list), np.around(np.sum(p_list), decimals=4),
                 np.around(np.sum(p_list * size), decimals=2),
                 -np.around(np.sum(n_list - size * p_list), decimals=2),
                 np.around(np.sum(result), decimals=2)])
    print_lab_log(tabulate(rows, cols, tablefmt="latex"))


def task3Solver(size, distribution, p, alpha):
    k = get_k(size)
    limits = calculate(distribution, p, k)
    n_list, p_list, result = get_n_and_p(distribution, limits, size)
    create_table(n_list, p_list, result, size, limits)


def lab_2_part_least_squares():
    sizes = [20, 100]
    alpha = 0.05
    p = 1 - alpha

    # for normal ditribution
    task3Solver(sizes[1], np.random.normal(0, 1, size=sizes[1]), p, alpha)

    # for laplace ditribution
    task3Solver(sizes[0], dist.get_laplace_sample(size=sizes[0], scale=1 / math.sqrt(2), loc=0), p, alpha)

    # for uniform ditribution
    task3Solver(sizes[0], dist.get_uniform_sample(size=sizes[0], loc=-math.sqrt(3), scale=2 * math.sqrt(3)), p, alpha)


def task4(x_set: list, n_set: list):
    alpha = 0.05
    m_all = list()
    s_all = list()
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = ch_utils.data_mean(x)
        s = math.sqrt(ch_utils.data_dispersion(x, mean=m))

        m1 = [m - s * (stats.t.ppf(1 - alpha / 2, n - 1)) / math.sqrt(n - 1),
              m + s * (stats.t.ppf(1 - alpha / 2, n - 1)) / math.sqrt(n - 1)]
        s1 = [s * math.sqrt(n) / math.sqrt(stats.chi2.ppf(1 - alpha / 2, n - 1)),
              s * math.sqrt(n) / math.sqrt(stats.chi2.ppf(alpha / 2, n - 1))]

        m_all.append(m1)
        s_all.append(s1)

        print_lab_log("t: %i" % (n))
        print_lab_log("m: %.2f, %.2f" % (m1[0], m1[1]))
        print_lab_log("sigma: %.2f, %.2f" % (s1[0], s1[1]))
    return m_all, s_all


def task4_asymp(x_set: list, n_set: list):
    alpha = 0.05
    m_all = list()
    s_all = list()
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = ch_utils.data_mean(x)
        s = math.sqrt(ch_utils.data_dispersion(x, mean=m))

        m_as = [m - stats.norm.ppf(1 - alpha / 2) / math.sqrt(n), m + stats.norm.ppf(1 - alpha / 2) / math.sqrt(n)]
        e = (sum(list(map(lambda el: (el - m) ** 4, x))) / n) / s ** 4 - 3
        s_as = [s / math.sqrt(1 + stats.norm.ppf(1 - alpha / 2) * math.sqrt((e + 2) / n)),
                s / math.sqrt(1 - stats.norm.ppf(1 - alpha / 2) * math.sqrt((e + 2) / n))]

        m_all.append(m_as)
        s_all.append(s_as)

        print_lab_log("m asymptotic :%.2f, %.2f" % (m_as[0], m_as[1]))
        print_lab_log("sigma asymptotic: %.2f, %.2f" % (s_as[0], s_as[1]))
    return m_all, s_all


def draw_result(x_set: list, m_all, s_all):
    fig, (ax1, ax2, ax3, ax4) = pyplt.subplots(1, 4, figsize=(15, 5), gridspec_kw={'left': 0.05, 'right': 0.95 })

    # draw hystograms
    ax1.set_ylim(0, 1)
    hist_range = upscale_range_to_sample(x_set[0], [0, 0])
    plot_sample_hist(ax1, hist_range, x_set[0], label='N(0, 1) hyst n=20')
    m_max = max(m_all[0][0])
    m_min = min(m_all[0][0])
    s_max = max(s_all[0][0])
    sigma_range = [m_min - s_max, m_max + s_max]
    ax1.plot([m_min, m_min, m_max, m_max], [0.0, 0.7, 0.7, 0.0], 'r', label='$[m_{min}, m_{max}]$')
    ax1.plot([sigma_range[0]] + sigma_range + [sigma_range[-1]], [0.0, 0.4, 0.4, 0.0], 'g', label='$[m_{min} - \\sigma_{max}, m_{max} + \\sigma_{max}]$')
    ax1.legend(loc='best', frameon=True)
    ax2.set_ylim(0, 1)
    hist_range = upscale_range_to_sample(x_set[1], [0, 0])
    plot_sample_hist(ax2, hist_range, x_set[1], label='N(0, 1) hyst n=100')
    m_max = max(m_all[0][1])
    m_min = min(m_all[0][1])
    s_max = max(s_all[0][1])
    sigma_range = [m_min - s_max, m_max + s_max]
    ax2.plot([m_min, m_min, m_max, m_max], [0.0, 0.7, 0.7, 0.0], 'r', label='$[m_{min}, m_{max}]$')
    ax2.plot([sigma_range[0]] + sigma_range + [sigma_range[-1]], [0.0, 0.4, 0.4, 0.0], 'g', label='$[m_{min} - \\sigma_{max}, m_{max} + \\sigma_{max}]$')
    ax2.legend(loc='best', frameon=True)

    # draw intervals of m
    ax3.set_ylim(0.9, 1.8)
    ax3.plot([0, 0], [0.95, 1.35], 'g--', label='true "m"')
    ax3.plot(m_all[0][0], [1.3, 1.3], 'ro-', label='"m" interval n = 20')
    ax3.plot(m_all[0][1], [1.2, 1.2], 'bo-', label='"m" interval n = 100')
    ax3.plot(m_all[1][0], [1.1, 1.1], 'ro:', label='"m" interval asymp n = 20')
    ax3.plot(m_all[1][1], [1.0, 1.0], 'bo:', label='"m" interval asymp n = 100')
    ax3.legend()

    # draw intervals of sigma
    ax4.set_ylim(0.9, 1.8)
    ax4.plot([1, 1], [0.95, 1.35], 'g--', label='true $\\sigma$')
    ax4.plot(s_all[0][0], [1.3, 1.3], 'ro-', label='$\\sigma$ interval n = 20')
    ax4.plot(s_all[0][1], [1.2, 1.2], 'bo-', label='$\\sigma$ interval n = 100')
    ax4.plot(s_all[1][0], [1.1, 1.1], 'ro:', label='$\\sigma$ interval asymp n = 20')
    ax4.plot(s_all[1][1], [1.0, 1.0], 'bo:', label='$\\sigma$ interval asymp n = 100')
    ax4.legend()
    return fig


def lab_2_part_asymp_normal_estimates():
    n_set = [20, 100]
    x_20 = dist.get_norm_sample(20)
    x_100 = dist.get_norm_sample(100)
    x_set = [x_20, x_100]
    m_all = [None, None]
    s_all = [None, None]
    m_all[0], s_all[0] = task4(x_set, n_set)
    m_all[1], s_all[1] = task4_asymp(x_set, n_set)
    fig = draw_result(x_set, m_all, s_all)
    fig.savefig('lab_2/part_asymp_normal_estimates/figures/norm_asymp.png')
    return


lab_log = None


def print_lab_log(value, sep=' ', end='\n'):
    print(value, sep=sep, end=end, file=lab_log)


def run_lab2():
    lab_2_part_correlation_coefficient()
    lab_2_part_scattering_ellipse()
    lab_2_part_linear_regression()
    lab_2_part_least_squares()
    lab_2_part_asymp_normal_estimates()


if __name__ == '__main__':
    lab_log = open('lab_2/log.txt', 'w')
    run_lab2()
