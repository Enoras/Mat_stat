import math
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform
from scipy.stats import multivariate_normal

norm_loc = 0
norm_scale = 1

cauchy_loc = 0
cauchy_scale = 1

laplace_loc = 0
laplace_scale = 1. / math.sqrt(2)

poisson_mu = 10
poisson_loc = 0

uniform_loc = -math.sqrt(3)
uniform_scale = 2 * math.sqrt(3)


def get_norm_sample(size: int, loc=norm_loc, scale=norm_scale):
    return norm.rvs(size=size, loc=loc, scale=scale)


def get_cauchy_sample(size: int, loc=cauchy_loc, scale=cauchy_scale):
    return cauchy.rvs(size=size, loc=loc, scale=scale)


def get_laplace_sample(size: int, loc=laplace_loc, scale=laplace_scale):
    return laplace.rvs(size=size, loc=loc, scale=scale)


def get_poisson_sample(size: int, mu=poisson_mu, loc=poisson_loc):
    return poisson.rvs(size=size, mu=mu, loc=loc)


def get_uniform_sample(size: int, loc=uniform_loc, scale=uniform_scale):
    return uniform.rvs(size=size, loc=loc, scale=scale)


def norm_func(x, loc=norm_loc, scale=norm_scale):
    return norm.cdf(x, loc=loc, scale=scale)


def cauchy_func(x, loc=cauchy_loc, scale=cauchy_scale):
    return cauchy.cdf(x, loc=loc, scale=scale)


def laplace_func(x, loc=laplace_loc, scale=laplace_scale):
    return laplace.cdf(x, loc=loc, scale=scale)


def poisson_func(x, mu=poisson_mu, loc=poisson_loc):
    return poisson.cdf(x, mu=mu, loc=loc)


def uniform_func(x, loc=uniform_loc, scale=uniform_scale):
    return uniform.cdf(x, loc=loc, scale=scale)


def norm_density_func(x, loc=norm_loc, scale=norm_scale):
    return norm.pdf(x, loc=loc, scale=scale)


def cauchy_density_func(x, loc=cauchy_loc, scale=cauchy_scale):
    return cauchy.pdf(x, loc=loc, scale=scale)


def laplace_density_func(x, loc=laplace_loc, scale=laplace_scale):
    return laplace.pdf(x, loc=loc, scale=scale)


def poisson_density_func(x, mu=poisson_mu, loc=poisson_loc):
    return poisson.pmf(x, mu=mu, loc=loc)


def uniform_density_func(x, loc=uniform_loc, scale=uniform_scale):
    return uniform.pdf(x, loc=loc, scale=scale)


dist_1d_list = {
    'norm':
        {
            'name': 'normal',
            'get_sample': get_norm_sample,
            'func': norm_func,
            'density_func': norm_density_func
        },
    'cauchy':
        {
            'name': 'cauchy',
            'get_sample': get_cauchy_sample,
            'func': cauchy_func,
            'density_func': cauchy_density_func
        },
    'laplace':
        {
            'name': 'laplace',
            'get_sample': get_laplace_sample,
            'func': laplace_func,
            'density_func': laplace_density_func
        },
    'poisson':
        {
            'name': 'poisson',
            'get_sample': get_poisson_sample,
            'func': poisson_func,
            'density_func': poisson_density_func
        },
    'uniform':
        {
            'name': 'uniform',
            'get_sample': get_uniform_sample,
            'func': uniform_func,
            'density_func': uniform_density_func
        }
}


def get_normal_2d_sample(size, r, loc=(0., 0.), scale=(1., 1.)):
    return multivariate_normal.rvs(loc, [[scale[0], r], [r, scale[1]]], size=size)


def get_mix_normal_2d_sample(size):
    return 0.9 * multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + \
           0.1 * multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)
