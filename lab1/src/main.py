from normal_dist import plot_normal, print_table_normal, boxplot_Tukey_normal, draw_emp_func_normal, draw_kde_normal
from uniform_dist import plot_uniform, print_table_uniform, boxplot_Tukey_uniform, draw_emp_func_uniform, draw_kde_uniform
from cauсhy_dist import plot_cauchy, print_table_cauchy, boxplot_Tukey_cauchy, draw_emp_func_cauchy, draw_kde_cauchy
from poisson_dist import plot_poisson, print_table_poisson, boxplot_Tukey_poisson, draw_emp_func_poisson, draw_kde_poisson

# LAB-1
# Task #1 - build distributions
SIZES = [10, 100, 1000]
M = 10.0
begin = -1.7320508
length = begin * 2
#plot_normal(SIZES, "normal nums", "denstiny", "skyblue")
#plot_uniform(begin, length, SIZES, "uniform nums", "denstiny", "royalblue")
#plot_cauchy(SIZES, "cauchy nums", "denstiny", "cyan")
#plot_poisson(M, SIZES, "poisson nums", "denstiny", "pink")

# Task #2 - build characteristiёcs values of different distributions
repeats = 1000
print_table_normal(SIZES, repeats)
print("\n")
print_table_uniform(SIZES, repeats)
print("\n")
print_table_cauchy(SIZES, repeats)
print("\n")
print_table_poisson(M, SIZES, repeats)
print("\n")

# Task #3 - build Tukey boxplotes for every distribution kind
SIZES_2 = [20, 100]
#boxplot_Tukey_normal(SIZES_2, repeats)
print("\n")
#boxplot_Tukey_uniform(SIZES_2, repeats)
print("\n")
#boxplot_Tukey_cauchy(SIZES_2, repeats)
print("\n")
#boxplot_Tukey_poisson(M, SIZES_2, repeats)
print("\n")

# Task #4 - build Tukey boxplotes for every distribution kind
# WARNING : SMTHNG CRAZY WITH MATPLOTLIB - IF U WANT TO BUILD TASK 4 - COMMENT ALL PREVIOUS TASKS
SIZES_3 = [20, 60, 100]
common_l, common_r = -4, 4
poisson_l, poisson_r = 6, 14
#draw_emp_func_normal(SIZES_3, common_l, common_r)
#draw_emp_func_uniform(SIZES_3, common_l, common_r)
#draw_emp_func_cauchy(SIZES_3, common_l, common_r)
#draw_emp_func_poisson(M, SIZES_3, poisson_l, poisson_r)

koefs = [0.5, 1, 2]
#draw_kde_normal(SIZES_3, common_l, common_r, koefs)
#draw_kde_uniform(SIZES_3, common_l, common_r, koefs)
#draw_kde_cauchy(SIZES_3, common_l, common_r, koefs)
#draw_kde_poisson(M, SIZES_3, poisson_l, poisson_r, koefs)