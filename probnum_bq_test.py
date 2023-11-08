"""
Sandbox testing for probabilistic numerics implementation of Bayesian quadrature
"""

import probnum as pn
import numpy as np

input_dim = 1
domain = (0, 1)


def fun(x):
    return x.reshape(-1, )


F, info = pn.quad.bayesquad(fun, input_dim, domain=domain, rng=np.random.default_rng(0))
print(f'Result: {F.mean}, Uncertainty: {F.std}')

