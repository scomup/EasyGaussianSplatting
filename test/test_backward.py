import numpy as np


def calc_tau(alpha_prime):
    tau = [1.]
    for a in alpha_prime:
        tau.append(tau[-1] * (1-a))
    return np.array(tau)

def calc_gamma(alpha_prime, color):
    tau = calc_tau(alpha_prime)
    gamma = 0
    for a, c, t in zip(alpha_prime, color, tau):
        gamma = gamma + a * c * t
    return gamma

def numericalDerivative(func, param, idx, plus=lambda a, b: a + b, minus=lambda a, b: a - b, delta=1e-5):
    r = func(*param)
    m = r.shape[0]
    n = param[idx].shape[0]
    J = np.zeros([m, n])
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = delta
        param_delta = param.copy()
        param_delta[idx] = plus(param[idx], dx)
        J[:, j] = minus(func(*param_delta), r)/delta
    return J

alpha_prime = np.array([0.124329,0.115078,0.325255,0.652917])
color = np.array([[-0.000009, 1.000009, -0.000009],[1.000009, -0.000009, 1.000009], [1.000009, -0.000009, -0.000009], [-0.000009, -0.000009, 1.000009]])
# tau = calc_tau(alpha_prime)
# gamma = alpha_prime[0] * color[0] + alpha_prime[1] * tau[1] * color[1] + alpha_prime[2] * tau[2] * color[2] + alpha_prime[3] * tau[3] * color[3]
gamma = calc_gamma(alpha_prime, color)

dgamma_dalphaprime = numericalDerivative(calc_gamma, [alpha_prime, color], 0)
print(dgamma_dalphaprime)