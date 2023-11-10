---
title: CROM Continuous Reduced Order Models
subtitle: Discretization-free Approximation of PDEs Using Implicit Neural Representations

# Summary for listings and search engines
summary: CROM - builds a continous low-dimensional embedding for vector fields of PDEs instead of relying on a predefined discretization of the latter.

# Link this post with a project
projects: []

# Date published
date: '2023-10-26T00:00:00Z'

# Date updated
lastmod: '2023-10-26T00:00:00Z'

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: ''
  focal_point: ''
  placement: 2
  preview_only: false

authors:
  - admin

tags:
  - Reduced Order Models

categories:
  - Model Order Reduction
  - Scientific Machine Learning
---
In this blog post, I want to talk about _continuous reduced order modeling_ (CROM) [[ChenEtAl23](https://doi.org/10.48550/arXiv.2206.02607)], an innovative discretization-free reduced order modeling approach to approximate PDEs. The key novelty lies in utilizing an implicit neural representation (INR) to create a _continuous representation_ of the reduced manifold, in contrast to the discrete representation commonly found in classical methods relying on reduction methods like proper orthogonal decomposition.
Assuming prior knowledge of the PDE's form, CROM involves using a solver to generate data for the training of a neural network (NN). This NN allows the generation of continuous spatial representations at any given time.

In order to evolve the dynamics at each time step, the original PDE is solved for the subsequent time point {{< math >}}$t_{i+1}${{< /math >}} using a numerical scheme, but on a reduced set of spatial points. This selective sampling of spatial points is the reason for time savings compared to high-fidelity solvers that operate on the entire spatial space.

The latent variable at time {{< math >}}$t_{i+1}${{< /math >}} is then inferred from the solutions at the small set of integration points by solving an inverse problem, enabling the INR to generate the complete solution for time t+1. This iterative process captures the dynamic evolution of the PDE.

Sounds complicated? Let's go through it step by step but first things first.

### Problem Setup

Many dynamical processes in engineering and sciences can be described by (nonlinear) _partial differential equations_ (PDEs)
{{< math >}}
  $$
  \mathcal{F}(\mathbf{f}, \nabla\mathbf{f}, \nabla^2\mathbf{f}, \dots, \dot{\mathbf{f}}, \ddot{\mathbf{f}})=0, \quad \mathbf{f}(\mathbf{x},t): \Omega \times \mathcal{T} \to \mathbb{R}^d
  $$
{{< /math >}}
where $\mathbf{f}$ describes a spatiotemporal vector field and $\nabla\mathbf{f}$ respective $\dot{\mathbf{f}}$ represent its spatial and time gradients. This vector field could for example describe the temperature or displacement of a continuous system for a given point (spatial coordinate) $\mathbf{x}\in\Omega\subseteq \mathbb{R}^m$ and time (temporal coordinate) $t \in \mathcal{T}\subseteq \mathbb{R}$.

A common approach to solve PDEs for $\mathbf{f}$ is to discretize them in space, e.g. using the finite element method, resulting in a set of _ordinary differential equations_ (ODEs)
{{< math >}}
  $$
  \dot{\mathbf{f}}(t) = \mathbf{q}(\mathbf{f},t): \Omega \times \mathcal{T} \to \mathbb{R}^d.
$$
{{< /math >}}
These ODEs can then be evolved in time using a time-stepping scheme such as Runge-Kutta methods. However, the discretization methods used often require very high resolutions to accurately approximate the continuous vector field. Consequently, the resulting high-fidelity models, also called _full-order models_ (FOM), can be extremely high-dimensional (millions of degrees of freedom are not uncommon in the modeling of complex three-dimensional systems). Hence, their evaluation is both, time consuming and resource demanding making them unsuitable for real-time applications, large parameter studies, or weak hardware. 

To alleviate this bottleneck, _reduced order models_ (ROMs) are used to significantly accelerate the calculations. The goal of reduced order modeling is to find efficient surrogate models that retain the expressiveness of the original high-fidelity simulation model while being much more cost-effective to evaluate. There are two main challenges: 
1. find expressive and low-dimensional coordinates to describe the vector field
2. evolve the dynamics

While conventional approaches create a reduced order model for a fixed discretization of the PDE, the recently proposed CROM directly approximates the continuous vector field itself. 

{{% callout note %}}
Reduced-order modeling is mainly about finding suitable low-dimensional embeddings to describe a given system, and a suitable way to evolve the dynamics on that embedding.
{{% /callout %}}

## Conventional Reduced Order Models
{{< figure src="ae.png" caption="Conventional autoencoder to construct a low-dimensional embedding for a discretized vector field." numbered="true" id="ae">}}
Conventional data-driven reduced order modeling approaches rely on a fixed discretization of a system. Fortunately, the number of degrees of freedoms results from the discretization method used and not from the intrinsic dimension of the given problem. This means that the actual system often lives on a low-dimensional manifold. For parameterized PDEs, the intrinsic dimension, i.e. the actual minimal dimension of the problem, equals at most the number of parameters $n_\text{p}$ plus one for the time $r=n_\text{p}+1$. 

Popular methods to find a low-dimensional embedding on which a given system can be described on include linear methods like the _principal component analysis_ (PCA) (also known as _proper orthogonal decomposition_ (POD)) or its nonlinear counterpart _autoencoders_ (AE). Those methods can be used to find a low-dimensional representation of the discretized vector field
 
{{< math >}}
  $$
  \mathbf{z} = enc_{\mathbf{\theta}_\text{e}}(\mathbf{f})
  $$
{{< /math >}}

but also to reconstruct the given discretization from this latent vector
  
{{< math >}}
  $$
  \mathbf{f} \approx dec_{\mathbf{\theta}_\text{d}}(\mathbf{z}).
  $$
{{< /math >}}

In case of an autoencoder, these mappings are found by optimizing the reconstruction loss

{{< math >}}
  $$
  \mathbf{\theta}_\text{e}^*, \mathbf{\theta}_\text{d}^* = \underset{\mathbf{\theta}_\text{e}, \mathbf{\theta}_\text{d}}{\text{argmin}} \left| \mathbf{f} - dec_{\mathbf{\theta}_\text{d}}(enc_{\mathbf{\theta}_\text{e}}(\mathbf{f})) \right|^2
  $$
{{< /math >}}

for the networks' weights {{< math >}}$\mathbf{\theta}_\text{e}, \mathbf{\theta}_\text{d}${{< /math >}} given some training data. Whereas a truncated singular value decomposition of the training data can be used in case of PCA.

Just as there are different approaches for the reduction, there are also different methods to approximate the temporal dynamics of a system. 
In case the underlying PDE is known, quasi-Newton solvers for implicit time integration of the latent dynamics on a continuously-differentiable nonlinear manifold have been used in [[LeeCarlberg20](https://doi.org/10.1016/j.jcp.2019.108973)]. 
When there is no knowledge about the underling equations present, some of the purely data-driven approaches try to directly approximate the time-dependent latent variable $\mathbf{z}(t) \approx \mathbf{\psi}(t)$ [[KneiflEtAl23](https://doi.org/10.1007/s00419-023-02458-5)] (Caution self-promotion), while others try to approximate the right-hand side of an ODE $\mathbf{\dot{z}}(t) = \mathbf{q}(\mathbf{z}, t) \approx \mathbf{\psi}(\mathbf{z}, t)$ on the low-dimensional embedding [[ChampionEtAl19](https://doi.org/10.1073/pnas.1906995116)]. In both cases, $\mathbf{\psi}(t)$ could be parameterized by a neural network. 

What all those approaches share is that they rely on an already discretized vector field. Consequently, a change of resolution requires a new architecture and new training of the reduced order model. Furthermore, they don't allow adaptive spatial resolution what can be useful in scenarios of varying intensity (low resolution when nothing is happening, high resolution when things get exciting) or for applications under changes of available computational resources.
CROM claims to be able to overcome these disadvantages. So, let's have a look how CROM differs from the previously presented approaches.

## CROM
{{< figure src="featured.png" caption="CROM latent manifold construction." numbered="true" id="ae">}}
CROM follows unconventional approaches in both, finding the low-dimensional embedding as well as in evolving the latent dynamics. 
### Embedding
Similar to a conventional autoencoder approaches it constructs an encoder and a decoder. While there are no changes to the encoder compared to conventional methods, the decoder this time does not try to reconstruct a discretized vector field from the latent variable. Instead the decoder 

{{< math >}}
  $$
  dec(\mathbf{x}, \mathbf{z}(t)) \approx \mathbf{f}(\mathbf{x}, t) \quad \forall \mathbf{x}\in\Omega, \quad \forall t\in\mathcal{T}
  $$
{{< /math >}}

takes the latent variable $\mathbf{z}(t)$ along with the positional variable $\mathbf{x}$ as input to reconstruct the vector field at this certain point. 
Consequently, the decoder directly approximates the continuous vector field. That's what the title of the original paper is referring to when speaking of implicit neural representations: Neural representations are class of methods using neural networks to represent arbitrary vector fields. 

To reproduce the behavior of a classical autoencoder, the decoder must be evaluated at all points of the discretization with different positional input. 
However, it is also possible to reconstruct the vector field for any other point $\mathbf{x}\in\Omega$. 
In order to optimize the weights of the encoder and decoder the loss function (ref) changes to 
<!-- % \min_{\mathbf{\theta}_\text{e}, \mathbf{\theta}_\text{d}} \  -->
{{< math >}}
  $$
  \mathcal{L}_{\text{crom}}=
    {\color{teal}\sum_{i=1}^{P}} \left| \mathbf{f}_{\color{teal}i} - dec_{\mathbf{\theta}_\text{d}}({\color{teal}\mathbf{x}^i}, \mathbf{z}(t)) \right|=
    {\color{teal}\sum_{i=1}^{P}} \left| \mathbf{f}_{\color{teal}i} - dec_{\mathbf{\theta}_\text{d}}({\color{teal}\mathbf{x}^i}, enc_{\mathbf{\theta}_\text{e}}(\mathbf{f}(t))) \right|
  $$
{{< /math >}}

{{% callout note %}}
CROM uses a decoder that directly approximates the continuous vector field.
{{% /callout %}}

### Dynamics
{{< figure src="dynamics.png" caption="Evolution of the latent dynamics." numbered="true" id="ae">}}

In contrast to conventional approaches CROM, evaluates the actual PDE for a small number of domain points {{< math >}}$\mathcal{X}=\{\mathbf{x}^i\}_{i=1}^{m}${{< /math >}} to evolve in time. For a small set, i.e. $m \ll P$, this leads to dramatic time savings compared to the full-order model where the PDE must be evaluated at all $P$ points.
The approach to update the latent variable from the current state {{< math >}}$\mathbf{z}_{n}=\mathbf{z}(t_{n})${{< /math >}} to the subsequent one {{< math >}}$\mathbf{z}_{n+1}=\mathbf{z}(t_{n+1})${{< /math >}} at the next time step consists of three steps:
1. network inference 
2. PDE time-stepping 
3. network inversion 

#### Network Inference
During the first step of the computation of the latent space dynamics, all spatiotemporal information for the PDE time-stepping is gathered for the selected samples {{< math >}}$\mathcal{X}${{< /math >}}. The function value itself {{< math >}}$\mathbf{f}(\mathbf{x},t_n)=dec(\mathbf{x}, \mathbf{z}_n)${{< /math >}} is obtained from the decoder for all {{< math >}}$\mathbf{x}\in\mathcal{X}${{< /math >}}along with the spatial {{< math >}}$\nabla\mathbf{f}(\mathbf{x},t_n) = \nabla_\mathbf{x}dec(\mathbf{x}, \mathbf{z}_n)${{< /math >}} as well as temporal gradients {{< math >}}$\dot{\mathbf{f}}(\mathbf{x},t_n) = \frac{\partial dec(\mathbf{x}, \mathbf{z}_n)}{\partial \mathbf{z}}\dot{\mathbf{z}}_n${{< /math >}}. A similar procedure can be used for higher order terms ({{< math >}}$\nabla^2_\mathbf{x}dec(\mathbf{x}, \mathbf{z}_n), \ddot{\mathbf{f}}(\mathbf{x},t_n), \dots ${{< /math >}}).

#### PDE Time-stepping
Having all necessary information available, the PDE {{< math >}}$\mathcal{F}(\mathbf{f}_n, \nabla\mathbf{f}_n, \dots, \dot{\mathbf{f}}_{n+1}, \dots)=0${{< /math >}} is evolved from $t_n$ to $t_{n+s}$. For each domain point {{< math >}}$\mathbf{x}\in\mathcal{X}${{< /math >}}, the PDE is solved for the temporal derivative {{< math >}}$\dot{\mathbf{f}}_{n+1}${{< /math >}} and the configuration is then evolved in time using a, in this case, explicit time integration method {{< math >}}$\mathcal{I}_{\mathcal{F}}${{< /math >}} like Runge-Kutta methods. The update then follows 
{{< math >}}
  $$\mathbf{f}_{n+1} = \mathcal{I}_{\mathcal{F}}(\Delta t, \mathbf{f}_{n}, \dot{\mathbf{f}}_{n+1}) \quad \forall \mathbf{x}\in\mathcal{X} .
  $$
{{< /math >}} 

#### Network Inversion
In the last step the dynamics on the reduced manifold are updated in such a way that {{< math >}}$\mathbf{z}_{n+1}${{< /math >}}  matches best to the just calculated {{< math >}}$\mathbf{f}_{n+1}${{< /math >}}. To achieve this the optimization problem
{{< math >}}
  $$\mathbf{z}_{n+1}^*= \underset{\mathbf{z}_{n+1}}{\text{argmin}} \sum_{\mathbf{x}\in\mathcal{X}} \left| dec(\mathbf{x}, \mathbf{z}_{n+1}) - \mathbf{f}(\mathbf{x}, t_{n+1})\right|
  $$
{{< /math >}} 
is solved with a with Gauss-Newton algorithm or using linearization.

{{% callout note %}}
CROM evolves the actual PDE for some domain points and updates the dynamics on the reduced manifold based on these results.
{{% /callout %}}

## Examples
{{< figure src="diffusion.gif" caption="Example simulation of the heat equation. The background is colored with respect to the corresponding diffusion factor in that interval." numbered="true" id="ae">}}
Enough of all the theory, let's dedicate ourselves to an example.
Let's consider a one dimensional diffusion problem
{{< math >}}
  $$
    \frac{\partial u}{\partial t} = \alpha(x) \frac{\partial^2 u}{\partial^2 x}
  $$
{{< /math >}} 
with the temperature {{< math >}}$u${{< /math >}} and a spatially-varying diffusion speed {{< math >}}$\alpha${{< /math >}}. Following the example from the CROM paper, the spatial coordinates is departed into three intervals of equal size with different diffusion factors, i.e. 
{{< math >}}
  $$
    \alpha(x) = \left\{ 
        \begin{array}{ll}
            \alpha_1 & 0 \leq x < 1/3 \\
            \alpha_2 & \, 1/3 \leq x < 2/3 \\
            \alpha_3 & \, 2/3 \leq x \leq 1 \\
        \end{array}
    \right. 
  $$
{{< /math >}}

#### High-fidelity Model
The high-fidelity spatial vector field is discretized with $P=501$ points and solved using the forward time-centered space method, a finite difference method that is based on the forward Euler method. 
An update of the vector field thus results in 
{{< math >}}
  $$
    \frac{u_{n+1}^i - u_{n}^i}{\Delta t} = \alpha \frac{u_n^{i+1} - 2u_n^{i} + u_n^{i-1}}{\Delta x^2}\\
      \Leftrightarrow \\
    u_{n+1}^i = u_n + \frac{\alpha \Delta t}{\Delta x^2}(u_n^{i+1} - 2u_n^{i} + u_n^{i-1}).
  $$
{{< /math >}}
With this method training and test data is generated based on different diffusion factors {{< math >}}$\alpha_1, \alpha_2, \alpha_3 \in [0.0,1.0]^3${{< /math >}}. Thereby, the initial condition is the same for every simulation and just the diffusion factors change.

For a given initial condition and values of alpha and it can be solved using the following code snippet
```python
import numpy as np

def solve_diffusion(Nt, Nx, dx, dt, alpha, u_init):
    """
    Solve the diffusion equation u_t - alpha * u_xx = 0 with a first-order Euler scheme
    :param Nt: number of time steps
    :param Nx: number of spatial grid points
    :param dx: spatial step size
    :param dt: time step size
    :param alpha: diffusion coefficient
    :param u_init: initial condition
    :return: u
    """
    # create temperature vector of size (timesteps x spatial points)
    u = np.zeros((Nt, Nx))
    # set initial condition
    u[0] = u_init

    # Main time-stepping loop
    for t in range(1, Nt):
        # Compute the second derivative using finite differences
        u_xx = np.zeros(Nx)
        u_xx[1:Nx-1] = (u[t - 1, 2:] - 2 * u[t - 1, 1:-1] + u[t - 1, :-2]) / (dx ** 2)
        # Update the solution
        u_t = alpha * u_xx
        # update solution using a first-order Euler scheme
        u[t] = u[t - 1] + dt * u_t

    return u
```

#### Reduced-Order Model
The latent dimension of the reduced order model is set to {{< math >}}$r=16${{< /math >}} and the decoder {{< math >}}$dec(x, \mathbf{z})${{< /math >}} consists out of three fully-connected hidden layers with 128 neurons each. 

A very simple implementation of the autoencoder for this example in PyTorch could look like this:
```python
import torch
from torch import nn

class CROMAutoencoder(nn.Module):
    def __init__(self, r, Nx, Nd=1):
        """
        :param r: latent space dimension
        :param Nx: number of spatial points
        :param Nd: number of coordinates required to describe the position (1 for 1D, 2 for 2D, ...)
        """
        super(CROMAutoencoder, self).__init__()
        self.r = r
        self.Nx = Nx
        self.Nd = Nd

        # create fully-connected encoder from u to z
        self.encoder = nn.Sequential(
            nn.Linear(self.Nx, 128),
            nn.ELU(True),
            nn.Linear(128, 64),
            nn.ELU(True),
            nn.Linear(64, self.r),
            nn.ELU(True),
        )
        # create fully-connected decoder from [x, z] to u(x)
        self.decoder = nn.Sequential(
            nn.Linear(self.r + self.Nd, 128),
            nn.ELU(True),
            nn.Linear(128, 128),
            nn.ELU(True),
            nn.Linear(128, 128),
            nn.ELU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x, u):
        """
        :param x: (batchsize x n) positional coordinates
        :param u: (batchsize x n) vector field values
        """
        
        # save batch size for reshaping later
        batch_size_local = x.size(0)

        # encode input to latent space
        z = self.encoder(u)

        
        # repeat z for every point in x
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        # add new axis to x in the middle
        x = x.unsqueeze(2)
        # concatenate x and z
        decoder_input = torch.cat((x, z), dim=2)
        # reshape for decoder so that all the points are processed at once (batchsize = batch_size_local * Nx)
        decoder_input = decoder_input.view(-1, self.r + self.Nd)
        
        # decode latent space to output
        u_rec = self.decoder(decoder_input)
        # reshape x_rec
        u_rec = u_rec.view(batch_size_local, -1)

        return u_rec
```

This autoencoder can then be trained on the FOM simulation data to find the low-dimensional embedding. 
After 12000 epochs, the autoencoder is able to reconstruct the full vector field quite well as demonstrated in the following gif.
{{< figure src="diffusion_rec_test_2.gif" caption="Reconstruction of the full vector field for test data." numbered="true" id="ae">}}

Hereafter, we can evolve the latent dynamics in time using only a few (in this case {{< math >}}$m=22${{< /math >}}) integration points. For a guidance how to select those points, please refer to the original paper.
First, we need to define a function to call the decoder for a given position {{< math >}}$x${{< /math >}} and latent variable {{< math >}}$\mathbf{z}${{< /math >}}
```python
def decode(x, z):
    """
    :param x: set of positional coordinates
    :param z: latent space coordinates
    """

    n = x.size(0)
    # repeat z for every point in x
    z = z.unsqueeze(0).repeat(n, 1)
    # add new axis to x in the middle
    x_support = x.unsqueeze(1)
    # concatenate x and z
    decoder_input = torch.cat((x, z), dim=1)
    # cast to float
    decoder_input = decoder_input.float()

    # decode latent space to output
    u_n = model.decoder(decoder_input).squeeze()

    return u_n
```
which can then be used inside the PDE time-stepping scheme:
```
# encode initial condition
z_init = model.encoder(torch.from_numpy(u_init).float()).detach()
# create latent vector of size (timesteps x (spatial points + 1))
z = torch.zeros((Nt, r))
# set initial condition
z[0] = z_init

# Main time-stepping loop
for t in range(1, Nt):
    u_old = decode(x_support, z[t-1])
    # compute the second order spatial gradient using autograd
    hessian = torch.func.hessian(decode, argnums=0)(x_support, z[t-1])
    u_xx = torch.diagonal(torch.diagonal(hessian, dim1=0, dim2=1), dim1=0, dim2=1).detach()
    # get time derivative
    u_t = alpha_support * u_xx
    # update solution using a first-order Euler scheme
    u_new = u_old + dt * u_t

    # todo
    # find z_new that best matches to u_new
    z[t] = ?
```
From this time series, we can reconstruct the full vector field at arbitrary points {{< math >}}$x\in\Omega${{< /math >}}. An example result for a test scenario can be seen in the following gif.
{{< figure src="diffusion_approx_test_4.gif" caption="Approximation of the full vector field for test data." numbered="true" id="ae">}}
Although the result is not perfect yet, one can see the potential. The noticable errors results from a model that is not optimized to the full extent and from the linearization that is used to find the optimal {{< math >}}$z_{i+1}${{< /math >}}.
Please also note that the presented code does only serve academic and illustrative purposes and is neither optimized nor complete. For a general implementation and high quality results please refer to the official [github page](https://crom-pde.github.io). 

### Concluding Remarks
CROM is a very powerful discretization-free ROM scheme for PDEs. The authors showed that it can outperform conventional approaches regarding accuracy and resource demands while offering adaptive spatial resolutions. Nevertheless, it shares some limitations with classic data-driven ROM approaches like the dependency on the training data, i.e. it won't generalize well to unseen scenarios. 
In contrast to complete data-driven approaches it offers a nice blend of ML and PDE solutions at the cost of increased method complexity.

#### Should you use CROM?
Well, it depends. It is a very promising method which is likely to pave the way for many more great research projects. Especially, when you can take advantage of adaptively adjusting your discretization ,it is worth a try. If you, however, don't mind the discretization present in your problem, you can still go for the more conventional approaches and in that case even consider whether linear reduction (PCA) isn't suitable for your certain problem.  

### Did you find this page helpful? Consider sharing it 🙌
