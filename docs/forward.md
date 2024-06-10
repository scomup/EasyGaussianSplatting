# The Forward Process of 3D Gaussian Splatting

This document describes the **forward process** of 3D Gaussian Splatting, which is the process of rendering trained 3D Gaussians onto a 2D image.

First, the required input data is defined as parameters and settings.

## Parameters

The parameters are the data that can be obtained from the training process:

- $q_i$: Rotation 3D Gaussian (quaternion)
- $s_i$: Scale of 3D Gaussian (3D vector)
- $h_i$: Spherical harmonics parameters of 3D Gaussian
- $\alpha_i$: Opacity of 3D Gaussian
- ${p_w}_i$: The location of 3D Gaussian in the world frame

where $i$ is the index of 3D Gaussian. For ease of reading, $i$ is omitted in the following text.

## Settings

The settings are values related to the configuration of a virtual camera, which do not need to be trained:

- $R_{cw}$: The rotation of the camera (local frame)
- $t_{cw}$: The translation of the camera (local frame)
- $K$: Intrinsic Parameters (i.e., $f_x, f_y, c_x, c_y$)

## Pipeline

The overall pipeline for rendering a 2D image from a set of 3D Gaussians is as follows:

1. Calculate the 2D coordinates when projecting the mean of the 3D Gaussian onto the 2D image.
2. Compute the 3D covariance of the 3D Gaussian.
3. Calculate the 2D covariance when projecting the 3D covariance onto the 2D image.
4. Calculate the color of the 3D Gaussian based on spherical harmonics.
5. Render the 2D image using the information calculated in steps 1-4.

## 1. Project the Mean of 3D Gaussian

We use the camera's rotation ($R_{cw}$) and translation ($t_{cw}$) to transform the mean ($p_w$) of the 3D Gaussian to a point in the camera coordinate system ($p_c$). This process is shown in equation (F.1.1).

$$
\begin{aligned}
p_c & = \mathrm{T}_{cw}(p_w) \\
& = R_{cw} p_w + t_{cw}
\end{aligned}
\tag{F.1.1}
$$

Then, we calculate the pixel ($u$) coordinates when projecting the point ($p_c$) onto the 2D image.

$$
\begin{aligned}
u(p_c) &= 
\begin{bmatrix} x f_x /z + c_x \\ y f_y /z + c_y 
\end{bmatrix} 
\end{aligned}
\tag{F.1.2}
$$

Here, each component of ($p_c$) is represented as $x$, $y$, and $z$, respectively.

## 2. Calculate the 3D Gaussian from Rotation and Color

The 3D covariance $\Sigma$ of the 3D Gaussian is expressed not directly as a matrix but as a composition of rotation $q$ and scaling $s$. The composition calculation is shown in equation (F.2).

$$
\begin{aligned}
\Sigma(q, s) &= RSS^TR^T \\
\sigma(q, s) & = \mathrm{upper\_triangular}(\Sigma)
\end{aligned}
\tag{F.2}
$$

Here, $R$ is the matrix representation of the quaternion $q$. $S$ is a diagonal matrix formed from the vector $s$.

$$
\begin{aligned}
R &=
\begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_zq_w) & 2(q_xq_z + q_yq_w) \\
2(q_xq_y + q_zq_w) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_xq_w) \\
2(q_xq_z - q_yq_w) & 2(q_yq_z + q_xq_w) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}
\end{aligned}
\tag{F.2.1}
$$

$$
\begin{aligned}
S &=
\begin{bmatrix}
s_0 & 0 & 0  \\
0 & s_1 &0   \\
0 & 0 & s_2 
\end{bmatrix}
\end{aligned}
\tag{F.2.2}
$$

## 3. Project the 3D Covariance to 2D Image as a 2D Covariance

When a 3D Gaussian is projected onto a 2D image, it can be represented as a 2D Gaussian distribution. Equation (F.3) shows the formula for calculating the covariance matrix of this 2D Gaussian distribution.

$$
\begin{aligned}
\Sigma^{\prime}(\sigma, p_c) &= J R_{cw}\Sigma R_{cw}^T J^T \\
\sigma^{\prime}(\sigma, p_c) &= \mathrm{upper\_triangular}(\Sigma^{\prime})
\end{aligned}
\tag{F.3}
$$

**Proof of Equation (3)**

Let $\mathbb{p}_w$ be a set of 3D points in world coordinates, and let $m_w$ be the mean of $\mathbb{p}_w$. According to the definition of the covariance matrix, the covariance matrix of $\mathbb{p}_w$ is calculated as follows:

$$
\Sigma = \mathrm{E}[(\mathbb{p}_w-m_w)(\mathbb{p}_w-m_w)^T]
\tag{F.3.1}
$$

The covariance matrix of points in camera coordinates can be calculated using equation (F.3.2):

$$
\begin{aligned}
\Sigma_c 
&= \mathrm{E}[(\mathbb{p}_c-m_c)(\mathbb{p}_c-m_c)^T] \\
&= \mathrm{E}[(\mathrm{T}_{cw}(\mathbb{p}_w)- \mathrm{T}_{cw}(m_c))(\mathrm{T}_{cw}(\mathbb{p}_w)-\mathrm{T}_{cw}(m_w))^T] \\
&= R_{cw} \mathrm{E}[(\mathbb{p}_w - m_w)(\mathbb{p}_w-m_w)^T] R_{cw}^T \\
&= R_{cw} \Sigma R_{cw}^T \\
\end{aligned}
\tag{F.3.2}
$$

The covariance matrix of points in image coordinates can be calculated using equation (F.3.3):

$$
\begin{aligned}
\Sigma^{\prime} 
&= \mathrm{E}[(\mathrm{u}(\mathbb{p}_c)-\mathrm{u}(m_c))(\mathrm{u}(\mathbb{p}_c)-\mathrm{u}(m_c))^T] 
\end{aligned}
\tag{F.3.3}
$$

Here, $m_c$ is the mean of $\mathbb{p}_c$, and $\mathbb{p}_c - m_c$ is a small value, which is defined as $\delta$. Although $\mathrm{u}$ (Equation F.1.2) is a nonlinear function, there exists an approximate calculation using the Jacobian matrix $J$ of $\mathrm{u}$ as follows:

$$
\begin{aligned}
&\mathrm{u}(m_c + \delta) = \mathrm{u}(m_c) +  J \delta　\\
&\mathrm{u}(\mathbb{p}_c) - \mathrm{u}(m_c) = J(\mathbb{p}_c - m_c)
\end{aligned}
\tag{F.3.4}
$$

Substituting equations (F.3.4) and (F.3.2) into equation (F.3.3), the 2D covariance can be calculated as follows:

$$
\begin{aligned}
\Sigma^{\prime} 
&= \mathrm{E}[J(\mathbb{p}_c-m_c)(\mathbb{p}_c-m_c)^TJ^T] \\
&= J\Sigma_c J^T \\
&= J R_{cw} \Sigma R_{cw}^T J^T \\
\end{aligned}
\tag{F.3.5}
$$

The Jacobian of $\mathrm{u}$ is given by:

$$
J = \begin{bmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x  x}{z^2} \\
0 & \frac{f_y}{z} & -\frac{f_y  y}{z^2}
\end{bmatrix}
\tag{F.3.6}
$$

## 4. Calculate the Color from Spherical Harmonics

In 3D Gaussian Splatting, spherical harmonics are used to approximate complex light absorption, refraction, or reflection, which exhibit color variations when viewed from different directions.

The color calculation using spherical harmonics is given by the following equation:

$$
\mathrm{c}(r, h) = \sum_{l=0}^{l_{\text{max}}}{\sum_{m=-l}^{l}{h_{lm}\mathrm{Y}_{l}^{m}(r)}}
\tag{F.4}
$$

Here, $r$ is the unit direction vector between the camera and the 3D Gaussian, and $Y_l^m$ represents the $l$-th dimension $m$-th base function.

## 5. Calculate the Color for Each Pixel

Using the information obtained from steps 1 to 4, calculate the final values (RGB colors) for all pixels in the 2D image.

$$
\gamma_{j} 
= \sum_{i \in N} \alpha_{ij}^{\prime} c_i \tau_{ij}
\tag{F.5}
$$

Where $\alpha_{ij}^{\prime}$ represents the opacity of the $i$-th 3D Gaussian at pixel $j$, and is calculated using the following equation:

$$
\alpha_{ij}^{\prime}(\alpha_i, \sigma^{\prime}_i, u_{i}) = 
\exp\left(-0.5 (u_{i}-x_{j}) \Sigma^{\prime-1}_i (u_{i}-x_{j})^T\right) \alpha_i
\tag{F.5.1}
$$

where:
$$
\Sigma^{\prime}_i = \mathrm{symmetric\_matrix}(\sigma^{\prime})
 $$
The opacity increases as the Mahalanobis distance between the mean of the 2D Gaussian ($u_{i}$) and the current pixel ($x_{j}$) gets closer.

As light from objects farther away from the camera travels through multiple objects and ends up in the camera, the light becomes weaker. As a result, the contribution to the final color decreases as the distance from the camera increases.

The $\tau_{ij}$ is a coefficient that accounts for the attenuation that occurs when rendering the $i$-th Gaussian at pixel $j$ as the light passes through multiple 3D Gaussians in front of it. In other words, it represents the maximum amount of color that can be reflected on this pixel. If the value is 1, there is no attenuation and the color is shown as is. Conversely, if the value is 0, there is complete attenuation and it is totally not visible.

$$
\begin{aligned}
 &\tau_{ij} = \prod_{k=1}^{i-1} (1 - \alpha_{kj}^{\prime}) \\
 &\tau_{1j}= 1
 \end{aligned}
\tag{F.5.2}
$$


Here, the inverse of the 2D covariance can be calculated as follows:

$$
\Sigma^{\prime -1} = 
\frac{1}{a c - b^2}
\begin{bmatrix}
    c & -b \\
    -b & a
\end{bmatrix}
\tag{F.5.3}
$$

where a, b, and c are the upper triangular elements of the 2D covariance.