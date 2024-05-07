# The Backward Process of 3D Gaussian Splatting

This document describes the **backward process** of 3D Gaussian Splatting, which is the process of training 3D Gaussians by given 2D image.

This process requires training the 3D Gaussians by optimizing, so we need to compute the jacabian of all the functions mentioned in the **forward process**.


## The input of backward process

### The output of forward process

- $\gamma_j$: The final values (RGB colors) for all pixels in the 2D image.


### Parameters

3D Gaussian parameters to be trained:

- $q_i$: Rotation 3D Gaussian (quaternion)
- $s_i$: Scale of 3D Gaussian (3D vector)
- $h_i$: Spherical harmonics parameters of 3D Gaussian
- $\alpha_i$: Opacity of 3D Gaussian
- ${p_w}_i$: The location of 3D Gaussian in the world frame

where $i$ is the index of 3D Gaussian. For ease of reading, $i$ is omitted in the following text.

## The output of backward process

$$
\newcommand{\diff}[2]{\frac{\partial #1}{\partial #2}} %norm 
$$

### Parameters

3D Gaussian parameters to be trained:

- $\diff{\gamma_j}{q_i}$: derivative of $\gamma_j$ with respect to rotation.
- $\diff{\gamma_j}{s_i}$: derivative of $\gamma_j$ with respect to scale .
- $\diff{\gamma_j}{h_i}$: derivative of $\gamma_j$ with respect to spherical harmonics parameters.
- $\diff{\gamma_j}{\alpha_i}$: derivative of $\gamma_j$ with respect to opacity.
- $\diff{\gamma_j}{{p_w}_i}$: derivative of $\gamma_j$ with respect to The location.

$$
\diff{\gamma_j}{q_i} = \diff{\gamma_j}{\sigma^{\prime}_i} \diff{\sigma^{\prime}_i}{\sigma_i} \diff{\sigma_i}{q_i}
$$

$$
\diff{\gamma_j}{s_i} = \diff{\gamma_j}{\sigma^{\prime}_i} \diff{\sigma^{\prime}_i}{\sigma_i} \diff{\sigma_i}{s_i}
$$

$$
\diff{\gamma_j}{h_i} = \diff{\gamma_j}{c_i} \diff{c_i}{h_i}
$$

$$
\diff{\gamma_j}{\alpha_i} = \diff{\gamma_j}{\alpha^{\prime}_i} \diff{\alpha^{\prime}_i}{\alpha}
$$

$$
\diff{\gamma_j}{{p_w}_i} = \diff{\gamma_j}{\sigma^{\prime}_i} \diff{\sigma^{\prime}_i}{\sigma_i} \diff{\sigma_i}{s_i}
$$

First, the required input data is defined as parameters and settings.


We compute the color $\gamma$ for each pixel by blending overlapped 2D Gaussians.

$c_i$ and $\alpha_i$ denote the color and alpha of the i-th 2D Gaussian. The larger the value of i, the farther the distance from the camera.

$$
 \tau_{ij} = \prod^{i-1}_{k=1} (1 - \alpha_{kj}^{\prime})
 \tag{1}
$$

$$
\gamma_j = \sum_{i \in N} \alpha_{ij}^{\prime} c_i \tau_{ij}
\tag{2}
$$ 

The calculation of equation (2) is equivalent to equation (3).

$$
\begin{aligned}
&\gamma = \\
&\gamma_1 = \alpha_1^{\prime} c_1 + (1 - \alpha_1^{\prime}) \gamma_2  \\
&\gamma_2 = \alpha_2^{\prime} c_2 + (1 - \alpha_2^{\prime}) \gamma_3 \\
&... \\
&\gamma_{N} = \alpha_{N}^{\prime} c_{N} + (1 - \alpha_{N}^{\prime})\gamma_{N+1} \\
&\gamma_{N+1} = 0
\end{aligned}
\tag{3}
$$

Find the derivative for each alpha.

$$
\begin{aligned}
&\diff{\gamma}{\alpha_1^{\prime}} = c_1 - \gamma_2 \\
&\diff{\gamma}{\alpha_2^{\prime}} = \diff{\gamma_1}{\gamma_2}(c_2 - \gamma_3) = \tau_2 (c_2 - \gamma_3)\\
&... \\
&\diff{\gamma}{\alpha_N^{\prime}} = \diff{\gamma_1}{\gamma_2}\diff{\gamma_2}{\gamma_3} ... \diff{\gamma_{N-1}}{\gamma_N} (c_N - \gamma_{N+1}) = \tau_N (c_N - \gamma_{N+1})\\
\end{aligned}
$$

Find the derivative for each color.

$$
\begin{aligned}
&\diff{\gamma}{c_1} = \alpha_1^{\prime} \\
&\diff{\gamma}{c_2} = \diff{\gamma_1}{\gamma_2}\alpha_2^{\prime} = \tau_2 \alpha_2^{\prime}\\
&... \\
&\diff{\gamma}{c_N} = \diff{\gamma_1}{\gamma_2}\diff{\gamma_2}{\gamma_3} ... \diff{\gamma_{N-1}}{\gamma_N} (c_N - \gamma_{N+1}) = \tau_N \alpha_N^{\prime}\\
\end{aligned}
$$


Base on mathematical induction, we can get:

$$
\diff{\gamma}{\alpha_i^{\prime}} = \tau_{i}(c_i - \gamma_{i+1})
\tag{4}
$$
$$
\diff{\gamma}{c_i} = \tau_{i}\alpha_i
\tag{5}
$$

opacity of the Gaussian


$$
\alpha^{\prime} = G \alpha \\

\diff{\alpha^{\prime}}{\alpha} = G
$$


$$
G = \exp(- 0.5 \omega_0 d_0^2 - 0.5 \omega_2 d_1^2 - \omega_1 d_0d_1) \\
\diff{G}{\omega} =
\begin{bmatrix}
-0.5d_0^2\\
-d_0d_1\\
-0.5d_1^2\\
\end{bmatrix}G
$$


$$
\diff{G}{u} =
\begin{bmatrix}
-\omega_0 d_0 -\omega_1 d_1 \\
 -\omega_2 d_1 -\omega_1 d_0\\
\end{bmatrix}G
$$