from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

import utils as u

def plot_poisson(mu : float, sizes : list, x_label : str, y_label : str, clr : str):
    for size in sizes:
        histogram = poisson.rvs(mu, size = size)
        rv = poisson(mu)

        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, density=True, histtype='stepfilled', color=clr, alpha = 0.55)

        x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
        ax.plot(x, rv.pmf(x), 'k--', lw=1.5)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("SIZE: " + str(size))
        plt.grid()
        #plt.show()
        plt.savefig(u.SAVE_PATH + "poisson"+str(size)+".png")
    return

def print_table_poisson(mu : float, sizes : list, repeats : int):
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
            distr = poisson.rvs(mu, size = size)
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
        u.print_table_rows(E, D, "Poisson E(z) " + str(size), "Poisson D(z) " + str(size))
    return

def boxplot_Tukey_poisson(mu : float, sizes : list, repeats : int):
    tips, result, count = [], [], 0
    for size in sizes:
        for i in range(repeats):
            distr = poisson.rvs(mu, size = size)
            distr.sort()
            count += u.number_of_emissions(distr)
        result.append(count/(size * repeats))

        distr = poisson.rvs(mu, size = size)
        distr.sort()
        tips.append(distr)
    u.draw_boxplot_Tukey(tips, "Poisson Tukey")  
    u.print_emissions(sizes, result)
    return

def draw_emp_func_poisson(mu : float, sizes : list, left_border : float, right_border : float):
    sns.set_style('whitegrid')
    figures, axs = plt.subplots(ncols=3, figsize=(15,5))
    for size in range(len(sizes)):
        x = np.linspace(left_border, right_border, 10000)
        y = poisson.cdf(x, mu = mu)
        sample = poisson.rvs(mu, size = sizes[size])
        sample.sort()
        ecdf = ECDF(sample)
        axs[size].plot(x, y, color='blue', label='cdf')
        axs[size].plot(x, ecdf(x), color='black', label='ecdf')
        axs[size].legend(loc='lower right')
        axs[size].set(xlabel='x', ylabel='$F(x)$')
        axs[size].set_title("Poisson" + ' n = ' + str(sizes[size])) 
    figures.savefig(u.SAVE_PATH + "PoissonEmp.jpg")
    return

#x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))

def draw_kde_poisson(mu : float, sizes : list, left_border : float, right_border : float, koefs : list):
    sns.set_style('whitegrid')
    for size in range(len(sizes)):
        figures, axs = plt.subplots(ncols=len(koefs), figsize=(15,5))
        rv = poisson(mu)
        x = np.arange(left_border, right_border + 1, 1)
        for kf in range(len(koefs)):
            y = rv.pmf(x)
            sample = rv.rvs(size = sizes[size])
            
            axs[kf].plot(x, y, color='blue', label='pdf')
            sns.kdeplot(data=sample, bw_method='silverman', bw_adjust = koefs[kf], ax=axs[kf], 
                        fill=True, common_norm=False, palette="crest", alpha=0, linewidth=2, label='kde')
            axs[kf].legend(loc='upper right')
            axs[kf].set(xlabel='x', ylabel='$f(x)$')
            axs[kf].set_xlim([left_border, right_border])
            axs[kf].set_title(' h = ' + str(koefs[kf]))

        figures.suptitle('Poisson KDE n = ' + str(sizes[size]))
        figures.savefig(u.SAVE_PATH + "Poisson KDE"+ str(sizes[size])+".jpg")
    return