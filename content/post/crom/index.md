---
title: CROM Continoues Reduced Order Modeling
subtitle: of PDEs Using Implicit Neural Representations

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
\mathcal{F}(\bm{f}, \nabla\bm{f}, \nabla^2\bm{f}, \dots, \dot{\bm{f}}, \ddot{\bm{f}})=0, \quad \bm{f}(\bm{x},t): \Omega \times \mathcal{T} \to \mathbb{R}^d
$$
{{< /math >}}
where $\bm{f}$ describes a spatiotemporal vector field and $\nabla\bm{f}$ respective $\dot{\bm{f}}$ represent its spatial and time gradients. This vector field could describe the displacement of a continous system for a given point (spatial coordinate) $\bm{x}\in\Omega\subseteq \mathbb{R}^m$ and time (temporal coordinate) $t \in \mathcal{T}\subseteq \mathbb{R}$.

A common approach to solve PDEs for $\bm{f}$ is to descretize them in space, e.g. using the finite element method, and then use a time-stepping scheme like Runge-Kutta methods to evolve their dynamics in time. However, the discretization methods used often require very high resolutions to accurately approximate the continoues vector field. Consequently, the resulting equations can be extremely high-dimensional (millions of degrees of freedom are not uncommon in the modeling of complex three-dimensional systems). Hence, their evaluation is both, time consuming and resource demanding making them unsuitable for real-time applications, large parameter studies, or weak hardware. 

To alleviate this bottleneck, _reduced order models_ (ROMs) are used to significantly accelerate the calculations. The goal of reduced order modeling is to find efficient surrogate models that maintain the expressiveness of the original high-fidelity simulation model while being way cheaper to evaluate. There are two main challenges: 
1. find expressive and low-dimensional coordinates to describe the vector field
2. evolve the dynamics

While conventional approaches create a reduced order model for a fixed discretization of the PDE, the recently proposed _continous reduced order modeling_ CROM (ref) directly approximates the continous vector field itself. 

## Conventional Reduced Order Models


## Examples

### Code

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

### Math

Wowchemy supports a Markdown extension for $\LaTeX$ math. You can enable this feature by toggling the `math` option in your `config/_default/params.yaml` file.

To render _inline_ or _block_ math, wrap your LaTeX math with `{{</* math */>}}$...${{</* /math */>}}` or `{{</* math */>}}$$...$${{</* /math */>}}`, respectively. (We wrap the LaTeX math in the Wowchemy _math_ shortcode to prevent Hugo rendering our math as Markdown. The _math_ shortcode is new in v5.5-dev.)

Example **math block**:

{{< math >}}
$$
f(k;p_{0}^{*}) = \begin{cases}p_{0}^{*} & \text{if }k=1, \\
1-p_{0}^{*} & \text{if }k=0.\end{cases}
$$
{{< /math >}}

### Diagrams

Wowchemy supports a Markdown extension for diagrams. You can enable this feature by toggling the `diagram` option in your `config/_default/params.toml` file or by adding `diagram: true` to your page front matter.

### Todo lists

- [x] Write math example
  - [x] Write diagram example
- [ ] Do something else

### Tables

Save your spreadsheet as a CSV file in your page's folder and then render it by adding the _Table_ shortcode to your page:

renders as

### Callouts

Academic supports a [shortcode for callouts](https://wowchemy.com/docs/content/writing-markdown-latex/#callouts), also referred to as _asides_, _hints_, or _alerts_. By wrapping a paragraph in `{{%/* callout note */%}} ... {{%/* /callout */%}}`, it will render as an aside.

{{% callout note %}}
A Markdown aside is useful for displaying notices, hints, or definitions to your readers.
{{% /callout %}}

### Icons

{{< icon name="terminal" pack="fas" >}} Terminal  
{{< icon name="python" pack="fab" >}} Python  
{{< icon name="r-project" pack="fab" >}} R

### Did you find this page helpful? Consider sharing it 🙌
