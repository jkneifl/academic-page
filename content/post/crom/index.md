---
title: CROM Continoues Reduced Order Models
subtitle: Discretization-free Approximation of PDEs Using Implicit Neural Representations

# Summary for listings and search engines
summary: CROM - builds a continous low-dimensional embedding for continous vector fields of PDEs themselves instead relying on a predefined discretization of the latter.

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

Many dynamical processes in engineering and sciences can be described by (nonlinear) _partial differential equations_ (PDE)
{{< math >}}
  $$
  \mathcal{F}(\mathbf{f}, \nabla\mathbf{f}, \nabla^2\mathbf{f}, \dots, \dot{\mathbf{f}}, \ddot{\mathbf{f}})=0, \quad \mathbf{f}(\mathbf{x},t): \Omega \times \mathcal{T} \to \mathbb{R}^d
  $$
{{< /math >}}
where $\mathbf{f}$ describes a spatiotemporal vector field and $\nabla\mathbf{f}$ respective $\dot{\mathbf{f}}$ represent its spatial and time gradients. This vector field could for example describe the displacement of a continous system for a given point (spatial coordinate) $\mathbf{x}\in\Omega\subseteq \mathbb{R}^m$ and time (temporal coordinate) $t \in \mathcal{T}\subseteq \mathbb{R}$.

A common approach to solve PDEs for $\mathbf{f}$ is to descretize them in space, e.g. using the finite element method, resulting in a set of _ordinary differential equations_ (ODEs)
{{< math >}}
  $$
  \dot{\mathbf{f}}(t) = \mathbf{q}(\mathbf{f},t): \Omega \times \mathcal{T} \to \mathbb{R}^d.
$$
{{< /math >}}
These ODEs can then be evolved in time using a time-stepping scheme like Runge-Kutta methods. However, the discretization methods used often require very high resolutions to accurately approximate the continoues vector field. Consequently, the resulting equations can be extremely high-dimensional (millions of degrees of freedom are not uncommon in the modeling of complex three-dimensional systems). Hence, their evaluation is both, time consuming and resource demanding making them unsuitable for real-time applications, large parameter studies, or weak hardware. 

To alleviate this bottleneck, _reduced order models_ (ROMs) are used to significantly accelerate the calculations. The goal of reduced order modeling is to find efficient surrogate models that maintain the expressiveness of the original high-fidelity simulation model while being way cheaper to evaluate. There are two main challenges: 
1. find expressive and low-dimensional coordinates to describe the vector field
2. evolve the dynamics

While conventional approaches create a reduced order model for a fixed discretization of the PDE, the recently proposed _continous reduced order modeling_ CROM (ref) directly approximates the continous vector field itself. 

{{% callout note %}}
Reduced order modeling mainly is concerned with finding suitable low-dimensional embeddings to describe a given system and a suitable way to evolve the dynamics on this embedding.
{{% /callout %}}

## Conventional Reduced Order Models
{{< figure src="ae.png" caption="Conventional autoencoder to construct a low-dimensional embedding for a discretized vector field." numbered="true" id="ae">}}
Conventional data-driven reduced order modeling approaches rely on a fixed description of a system. Fortunately, the number of degrees of freedoms results from the discretization method used and not from the intrinsic dimension of the given problem. This means that the actual system often lives on a low-dimensional embedding. For parameterized PDEs, the intrinsic dimension, i.e. the actual minimal dimension of the problem, equals at most the number of parameters $n_\text{p}$ plus one for the time $r=n_\text{p}+1$. 

Popular methods to find a low-dimensional embedding on which a given system can be described on include linear methods like the _principal component analysis_ (PCA) (also known as _proper orthorgnal decomposition_ (POD)) or its nonlinear counterpart _autoencoders_ (AE). Those methods can be used to find a low-dimensional representation of the discretized vector field
 
{{< math >}}
  $$
  \mathbf{z} = enc_{\mathbf{\theta}_\text{e}}(\mathbf{f})
  $$
{{< /math >}}

but also to reconstruct the given discretization from this reduced quantity
  
{{< math >}}
  $$
  \mathbf{f} \approx dec_{\mathbf{\theta}_\text{d}}(\mathbf{z}).
  $$
{{< /math >}}

In case of an autoencoder, these mappings are found by optimizing the reconstruction loss

{{< math >}}
  $$
  \min_{\mathbf{\theta}_\text{e}, \mathbf{\theta}_\text{d}} \left| \mathbf{f} - dec_{\mathbf{\theta}_\text{d}}(enc_{\mathbf{\theta}_\text{e}}(\mathbf{f})) \right|
  $$
{{< /math >}}

for the networks' weights {{< math >}}$\mathbf{\theta}_\text{e}, \mathbf{\theta}_\text{d}${{< /math >}} while a truncated singular value decomposition can be used in case of the PCA.

Just as there are different approaches for the reduction, there are also different methods to approximate the temporal dynamics of a system. Purely data-driven approaches either try to directly approximate the time-dependent latent variable $\mathbf{z}(t) = \mathbf{\psi}(t)$, where $\mathbf{\psi}(t)$ could be parameterized by a neural network, or try to approximate the right-hand side of an ODE $\mathbf{\dot{z}}(t) = \mathbf{q}(\mathbf{z}, t) \approx \mathbf{\psi}(\mathbf{z}, t)$ (as done in PINNs) on the low-dimensional embedding. Examples can be found in **cite**.
Let's have a look how CROM differs from such approaches.
Other equation based approaches solve the PDE in the latent space cite(Hesthaven20?)

Drawbacks of discretization based approaches: Change of resolution -> new architecture / training etc., no adaptivity of spatial resolution

## CROM
{{< figure src="featured.png" caption="A caption" numbered="true" id="ae">}}
### Embedding
Continous reduced order modeling (cite) follows unusual approaches in finding the embedding as well as in the time evolution of the latent dynamics.
Similar to a conventional autoencoder it consists of an encoder and decoder. While the encoder does not change (as it is only used during training anyways), the decoder this time does not try to reconstruct the discretized vector field from the latent variable. Instead the decoder 

{{< math >}}
  $$
  dec(\mathbf{x}, \mathbf{z}(t)) \approx \mathbf{f}(\mathbf{x}, t) \quad \forall \mathbf{x}\in\Omega, \quad \forall t\in\mathcal{T}
  $$
{{< /math >}}

takes the latent variable $\mathbf{z}(t)$ along with the positional variable $\mathbf{x}$ as input to reconstruct the vector field for this certain point. 
Consequently, the decoder directly approximates the continous vector field. That's what the title of the original paper is refering to when speaking of implicit neural representations: they use neural networks to represent arbitrary vector fields. 
To reproduce the behavior of a classical autoencoder, the decoder must be evaluated at all points of the discretization with different positional input. 
However, it is also possible to reconstruct the vector field for any other point $\mathbf{x}\in\Omega$. 
In order to optimize the weights of the encoder and decoder the loss function (ref) changes to 
<!-- % \min_{\mathbf{\theta}_\text{e}, \mathbf{\theta}_\text{d}} \  -->
{{< math >}}
  $$
  \mathcal{L}_{\text{crom}}=
    \sum_{i=1}^{P} \left| \mathbf{f}_i - dec_{\mathbf{\theta}_\text{d}}(\mathbf{x}, \mathbf{z}(t)) \right|=
    \sum_{i=1}^{P} \left| \mathbf{f}_i - dec_{\mathbf{\theta}_\text{d}}(\mathbf{x}, enc_{\mathbf{\theta}_\text{e}}(\mathbf{f}(t))) \right|
  $$
{{< /math >}}

{{% callout note %}}
CROM uses a decoder that directly approximates the continous vector field.
{{% /callout %}}

### Dynamics
In contrast to conventional approaches CROM evaluates the actual PDE for a small number of domain points {{< math >}}$\mathcal{X}=\{\mathbf{x}_i\}_{i=1}^{n}${{< /math >}} to evolve in time. 
The approach to update the latent variable from {{< math >}}$\mathbf{z}_{n}=\mathbf{z}(t_{n})${{< /math >}} to {{< math >}}$\mathbf{z}_{n+1}=\mathbf{z}(t_{n+1})${{< /math >}} at the next time step. consists of three steps -->
1. network inference 
    {{< math >}}$\mathbf{f}(\mathbf{x},t_n)=dec(\mathbf{x}, \mathbf{z}_n) \quad \forall \mathbf{x}\in\mathcal{X}${{< /math >}} {{< math >}}$\to${{< /math >}} 
    {{< math >}}$\nabla\mathbf{f}(\mathbf{x},t_n) = \nabla_\mathbf{x}dec(\mathbf{x}, \mathbf{z}_n)${{< /math >}},
    {{< math >}}$\dot{\mathbf{f}}(\mathbf{x},t_n) = \frac{\partial dec(\mathbf{x}, \mathbf{z}_n)}{\partial \mathbf{z}}\dot{\mathbf{z}}_n${{< /math >}}
2. PDE time-stepping 
    PDE {{< math >}}$\mathcal{F}(\mathbf{f}_n, \nabla\mathbf{f}_n, \dots, \dot{\mathbf{f}}_{n+1}, \dots)=0${{< /math >}}
    Time-stepping {{< math >}}$\mathbf{f}_{n+1} = \mathcal{I}_{\mathcal{F}}(\Delta t, \mathbf{f}_{n}, \dot{\mathbf{f}}_{n+1}) \quad \forall \mathbf{x}\in\mathcal{X} ${{< /math >}} 
3. network inversion 
    find {{< math >}}$\mathbf{z}_{n+1}: \min_{\mathbf{z}_{n+1}} \sum_{\mathbf{x}\in\mathcal{X}} \left| dec(\mathbf{x}, \mathbf{z}_{n+1}) - \mathbf{f}(\mathbf{x}, t_{n+1})\right|${{< /math >}} with Gauss-Newton algorithm (or linearization)

{{% callout note %}}
CROM evolves the actual PDE for some domain points and updates the dynamics on the reduced manifold based on these results.
{{% /callout %}}

### 

```python
import pandas as pd
data = pd.read_csv("data.csv")
data.head()
```


### Charts

Wowchemy supports the popular [Plotly](https://plot.ly/) format for interactive charts.

Save your Plotly JSON in your page folder, for example `line-chart.json`, and then add the `{{</* chart data="line-chart" */>}}` shortcode where you would like the chart to appear.

Demo:

{{< chart data="line-chart" >}}

You might also find the [Plotly JSON Editor](http://plotly-json-editor.getforge.io/) useful.

### Diagrams

Wowchemy supports a Markdown extension for diagrams. You can enable this feature by toggling the `diagram` option in your `config/_default/params.toml` file or by adding `diagram: true` to your page front matter.

### Todo lists

- [x] Write math example
  - [x] Write diagram example
- [ ] Do something else

### Tables

Save your spreadsheet as a CSV file in your page's folder and then render it by adding the _Table_ shortcode to your page:

renders as

### Icons

{{< icon name="terminal" pack="fas" >}} Terminal  
{{< icon name="python" pack="fab" >}} Python  
{{< icon name="r-project" pack="fab" >}} R

### Did you find this page helpful? Consider sharing it 🙌
