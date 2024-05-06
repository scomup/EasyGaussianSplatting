import numpy as np

alpha_prime = [0.124329,0.115078,0.325255,0.652917]
tau = [1, (1- alpha_prime[0]), (1- alpha_prime[0])*(1- alpha_prime[1]), (1- alpha_prime[0])*(1- alpha_prime[1])*(1- alpha_prime[2])]
color = np.array([[-0.000009, 1.000009, -0.000009],[1.000009, -0.000009, 1.000009], [1.000009, -0.000009, -0.000009], [-0.000009, -0.000009, 1.000009]])

gamma = alpha_prime[0] * color[0] + alpha_prime[1] * tau[1] * color[1] + alpha_prime[2] * tau[2] * color[2] + alpha_prime[3] * tau[3] * color[3]
