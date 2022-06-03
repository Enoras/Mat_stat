from scipy.stats import cauchy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

import utils as u

def plot_cauchy(sizes : list, x_label : str, y_label : str, clr : str):
    for size in sizes:
        histogram = cauchy.rvs(size = size)
        rv = cauchy()

        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, density=True, histtype='stepfilled', color=clr, alpha = 0.55)

        x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
        ax.plot(x, rv.pdf(x), 'k--', lw=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("SIZE: " + str(size))
        plt.grid()
        #plt.show()
        plt.savefig(u.SAVE_PATH + "cauchy" + str(size) + ".png")
    return

def print_table_cauchy(sizes : list, repeats : int):
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
            distr = cauchy.rvs(size = size)
            distr.sort()
            means.append(u.mean(distr))
            meds.append(u.median(distr))
            zRs.append(u.zR(distr))
            zQs.append(u.zQ(distr))
            zTRs.append(u.zTR(distr))
        for column in table:
            E.append(round(u.mean(column), u.ROUND_SIGNS))
            D.append(round(u.dispersion(column), u.ROUND_SIGNS))
        #print("size: " + str(size))
        u.print_table_rows(E, D, "Cauchy E(z) " + str(size), "Cauchy D(z) " + str(size))
    return

def boxplot_Tukey_cauchy(sizes : list, repeats : int):
    tips, result, count = [], [], 0
    for size in sizes:
        for i in range(repeats):
            distr = cauchy.rvs(size = size)
            distr.sort()
            count += u.number_of_emissions(distr)
        result.append(count/(size * repeats))

        distr = cauchy.rvs(size = size)
        distr.sort()
        tips.append(distr)
    u.draw_boxplot_Tukey(tips, "Cauchy Tukey")
    u.print_emissions(sizes, result)
    return

def draw_emp_func_cauchy(sizes : list, left_border : float, right_border : float):
    sns.set_style('whitegrid')
    figures, axs = plt.subplots(ncols=3, figsize=(15,5))
    for size in range(len(sizes)):
        x = np.linspace(left_border, right_border, 10000)
        y = cauchy.cdf(x)
        sample = cauchy.rvs(size = sizes[size])
        sample.sort()
        ecdf = ECDF(sample)
        axs[size].plot(x, y, color='blue', label='cdf')
        axs[size].plot(x, ecdf(x), color='black', label='ecdf')
        axs[size].legend(loc='lower right')
        axs[size].set(xlabel='x', ylabel='$F(x)$')
        axs[size].set_title("Cauchy" + ' n = ' + str(sizes[size])) 
    figures.savefig(u.SAVE_PATH + "CauchyEmp.jpg")
    return

def draw_kde_cauchy(sizes : list, left_border : float, right_border : float, koefs : list):
    sns.set_style('whitegrid')
    for size in range(len(sizes)):
        figures, axs = plt.subplots(ncols=len(koefs), figsize=(15,5))
        x = np.linspace(left_border, right_border, 10000)
        for kf in range(len(koefs)):
            y = cauchy.pdf(x)
            sample = cauchy.rvs(size = sizes[size])
            
            axs[kf].plot(x, y, color='blue', label='pdf')
            sns.kdeplot(data=sample, bw_method='silverman', bw_adjust = koefs[kf], ax=axs[kf], 
                        fill=True, common_norm=False, palette="crest", alpha=0, linewidth=2, label='kde')
            axs[kf].legend(loc='upper right')
            axs[kf].set(xlabel='x', ylabel='$f(x)$')
            axs[kf].set_xlim([left_border, right_border])
            axs[kf].set_title(' h = ' + str(koefs[kf]))

        figures.suptitle('Cauchy KDE n = ' + str(sizes[size]))
        figures.savefig(u.SAVE_PATH + "Cauchy KDE"+ str(sizes[size])+".jpg")
    return