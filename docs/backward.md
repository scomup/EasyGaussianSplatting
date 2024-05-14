# The Backward Process of 3D Gaussian Splatting

This document describes the **backward process** of 3D Gaussian Splatting, which involves training 3D Gaussians using a given 2D image. The training process can generally be treated as an optimization problem, aiming to find a set of parameters that minimize an overall loss function $\mathcal{L}$ (or objective function).
$$
\underset{x}{\textrm{argmin}} \quad \mathcal{L} = \mathcal{L}(\gamma, \gamma_{gt}) \\\\
\tag{1}
$$

where $\gamma$ is the output image from the forward process, and $\gamma_{gt}$ is the given ground truth image.

The loss function $\mathcal{L}$ for 3D Gaussian Splatting is defined as a combination of L1 loss ($\mathcal{L}1$) and D-SSIM loss ($\mathcal{L}{D-SSIM}$).

$$
\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{D-SSIM}
\tag{2}
$$

The value of $\lambda$ ranges between 0 and 1. When $\lambda$ is close to 0, the loss function $\mathcal{L}$ is more similar to L1 loss, whereas when $\lambda$ is close to 1, $\mathcal{L}$ is more similar to D-SSIM loss.

To solve this optimization problem, it is necessary to find the Jacobians of the loss function with respect to each input parameter. These Jacobians provide information about how the loss changes as each input parameter is varied.

In the following sections, we will describe how to calculate these Jacobians.

## Jacobians
The computation of $\gamma$ in (2) has already been described in `forward.md`, so the Jacobian for each parameter can be computed using the chain rule.

$$
\newcommand{\diff}[2]{\frac{\partial #1}{\partial #2}} %diff 
$$

### The Jacobian of Rotation

$$
\begin{aligned}
\diff{\mathcal{L}}{q_i} &= \sum_{j}{
    \diff{\mathcal{L}}{\gamma_j}
    \diff{\gamma_j}{\sigma_i^{\prime}}
    \diff{\sigma_i^{\prime}}{\sigma_i}}
    \diff{\sigma_i}{q_i}
\end{aligned}
\tag{3}
$$

### The Jacobian of Scale

$$
\begin{aligned}
\diff{\mathcal{L}}{s_i} &= \sum_{j}{
    \diff{\mathcal{L}}{\gamma_j}
    \diff{\gamma_j}{\sigma_i^{\prime}}
    \diff{\sigma_i^{\prime}}{\sigma_i}}
    \diff{\sigma_i}{s_i}
\end{aligned}
\tag{4}
$$

### The Jacobian of Spherical Harmonics Parameters

$$
\begin{aligned}
\diff{\mathcal{L}}{h_i} &= \sum_{j}{
    \diff{\mathcal{L}}{\gamma_j}
    \diff{\gamma_j}{c_i}
    \diff{c_i}{h_i}} 
\end{aligned}
\tag{5}
$$

### The Jacobian of Alpha

$$
\begin{aligned}
\diff{\mathcal{L}}{\alpha_i} &= \sum_{j}{
    \diff{\mathcal{L}}{\gamma_j}
    \diff{\gamma_j}{\alpha_{ij}^{\prime}}
    \diff{\alpha_{ij}^{\prime}}{\alpha_i}} 
\end{aligned}
\tag{6}
$$

### The Jacobian of Location

$$
\begin{aligned}
\diff{\mathcal{L}}{p_{w,i}} 
&= \sum_{j}{
    \diff{\mathcal{L}}{\gamma_j}
    \diff{\gamma_j}{\alpha_{ij}^{\prime}}
    \diff{\alpha_{ij}^{\prime}}{u_{i}}}
    \diff{u_{i}}{p_{c,i}}
    \diff{p_{c,i}}{p_{w,i}} \\
&+  \sum_{j}{
    \diff{\mathcal{L}}{\gamma_j}
    \diff{\gamma_j}{c_i}
    \diff{c_i}{p_{w,i}}} \\
&+  \sum_{j}{
    \diff{\mathcal{L}}{\gamma_j}
    \diff{\gamma_j}{\alpha_{ij}^{\prime}}
    \diff{\alpha_{ij}^{\prime}}{\sigma_i^{\prime}}
    \diff{\sigma_i^{\prime}}{p_{c,i}}
    \diff{p_{c,i}}{p_{w,i}}
    }
\end{aligned}
\tag{7}
$$

Next, let's discuss all the partial derivatives mentioned above. The computation of $\frac{\partial \mathcal{L}}{\partial \gamma_j}$ can be performed using PyTorch's automatic differentiation, so we will not discuss it here.


### The Partial Derivatives of Transform Function (F.1.1)

$$
\diff{p_{c,i}}{p_{w,i}} =
R_{cw}
\tag{B.1.1}
$$

### The Partial Derivatives of Projection Function (F.1.2)

$$
\diff{u_i}{p_{c,i}} =
\begin{bmatrix} 
f_x/z & 0 & -f_x x/z^2 \\\\ 
0 & f_y/z & -f_y y/z^2
\end{bmatrix}  \\\\
\tag{B.1.2}
$$
x, y, z are the elements of $p_c$.


### The Partial Derivatives of Computing 3D Covariance (F.2)


### The Partial Derivatives of Computing 2D Covariance (F.3)


### The Partial Derivatives of Computing Spherical Harmonics (F.4)

### The Partial Derivatives of Computing Color of Pixel (F.5)

according to (F.5) in forward.md, we compute the $\gamma_j$ by following equation.


$$
\gamma_j = \sum_{i \in N} \alpha_{ij}^{\prime} c_i \tau_{ij}
$$ 

$$
 \tau_{ij} = \prod^{i-1}_{k=1} (1 - \alpha_{kj}^{\prime})
$$

The above equation can be written in the following iterative way.

Where N represents the number of 3D Gaussians and $\gamma_{i,j}$ represents the current color by considering 3D Gaussians from i to N.

$$
\begin{aligned}
&\gamma_{N+1,j} = 0 \\
&\gamma_{N, j} = \alpha_{N,j}^{\prime} c_{N,j} + (1 - \alpha_{N,j}^{\prime})\gamma_{N+1,j} \\
&... \\
&\gamma_{2,j} = \alpha_{2,j}^{\prime} c_2 + (1 - \alpha_{2,j}^{\prime}) \gamma_{3,j} \\
&\gamma_{1,j} = \alpha_{1,j}^{\prime} c_1 + (1 - \alpha_{1,j}^{\prime}) \gamma_{2,j}  \\
&\gamma_{j} = \gamma_{1,j}
\end{aligned}
$$

Therefore, we can calculate the partial derivatives of $\gamma_j$ with respect to each $\gamma_{i,j}$ iteratively.

$$
\diff{\gamma_j}{\alpha_{i,j}^{\prime}} = \tau_{i,j}(c_i - \gamma_{i+1,j})
\tag{B.5.a}
$$


Similarly, The partial derivatives of $\gamma_j$ with respect to $c_{i}$ can be calculated as follows.

$$
\diff{\gamma_j}{c_i} = \tau_{i,j}\alpha_{i,j}^{\prime}
\tag{B.5.b}
$$

The $\alpha_{ij}^{\prime}$ is calculated using the following equation (see forward.md (F.5.1)):

$$
\alpha_{ij}^{\prime}(\alpha_i, \sigma^{\prime}_i, x_{j}) = 
\exp\left(-0.5 (u_{i}-x_{j}) \Sigma^{\prime-1}_i (u_{i}-x_{j})^T\right) \alpha_i
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