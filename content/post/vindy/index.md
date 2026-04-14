---
title: VENI, VINDy, VICI
subtitle: A Generative Approach to Reduced-Order Modeling with Uncertainty Quantification

# Summary for listings and search engines
summary: VENI, VINDy, VICI is an interpretable, data-driven framework for building generative reduced-order models with uncertainty quantification, combining variational autoencoders with a probabilistic extension of SINDy.

# Link this post with a project
projects: []

# Date published
date: '2024-11-07T00:00:00Z'

# Date updated
lastmod: '2024-11-07T00:00:00Z'

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
  - Uncertainty Quantification
  - Generative AI

categories:
  - Model Order Reduction
  - Scientific Machine Learning
---

Complex physical systems — think fluid flows, structural vibrations, or chemical reactions — are typically modeled by partial differential equations (PDEs). Solving these PDEs numerically is accurate but expensive: a single simulation can take hours or days. Reduced-order models (ROMs) or deep learning surrogates tackle this by finding a low-dimensional representation of the system that is cheap to evaluate while modern. The catch is that most ROMs are deterministic and assume clean, reliable data, neither of which holds in practice; while data-driven approaches often lack interpretability.

**VENI, VINDy, VICI** [[paper](https://doi.org/10.1016/j.neunet.2026.108543)] addresses those issues at once. It builds a generative ROM that (1) handles noisy input data, (2) identifies interpretable governing equations in a low-dimensional space, and (3) produces predictions with calibrated uncertainty estimates — all within a single probabilistic framework.

{{< figure src="featured.gif" caption="Overview of the VENI, VINDy, VICI framework: noisy snapshots are encoded into a probabilistic latent space (VENI), governing equations are identified as distributions over sparse coefficient vectors (VINDy), and predictions are decoded back into the full space with uncertainty intervals (VICI)." numbered="true" id="overview">}}

{{% callout note %}}
The code is available on GitHub: [github.com/jkneifl/VENI-VINDy-VICI](https://github.com/jkneifl/VENI-VINDy-VICI)
{{% /callout %}}

---

## The Big Picture

The framework consists of three tightly coupled components:

- **VENI** — *Variational Encoding of Noisy Inputs*: compress high-dimensional, noisy state snapshots into a low-dimensional probabilistic latent space using a variational autoencoder.
- **VINDy** — *Variational Identification of Nonlinear Dynamics*: discover sparse governing equations in the latent space, where the equation coefficients are themselves probability distributions.
- **VICI** — *Variational Inference with Certainty Intervals*: propagate both latent dynamics and the coefficient uncertainty forward in time to produce predictions with confidence bounds.

What sets this apart from standard ROMs is that uncertainty is embedded into every step, from the encoding of raw data all the way to the final prediction.

---

## VENI: Encoding with a Variational Autoencoder

A standard autoencoder maps each input snapshot {{< math >}}$\mathbf{x} \in \mathbb{R}^N${{< /math >}} to a single point {{< math >}}$\mathbf{z} \in \mathbb{R}^r${{< /math >}} in a low-dimensional latent space ({{< math >}}$r \ll N${{< /math >}}). This works well for clean data, but when measurements are noisy, the encoder has no principled way to separate signal from noise.

A **variational autoencoder** (VAE) takes a different approach: instead of mapping to a point, the encoder maps each input to a *distribution* over the latent space,

{{< math >}}
$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\!\left(\mathbf{z};\, \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}),\, \text{diag}(\boldsymbol{\sigma}^2_{\boldsymbol{\phi}}(\mathbf{x}))\right).
$$
{{< /math >}}

Concretely, the encoder network outputs two vectors — a mean {{< math >}}$\boldsymbol{\mu}${{< /math >}} and a standard deviation {{< math >}}$\boldsymbol{\sigma}${{< /math >}} — for every input snapshot. A latent state is then *sampled* from this Gaussian rather than being read off directly. The decoder takes this sample and reconstructs the full-dimensional state.

{{% callout note %}}
The encoder does not output a single latent vector. It outputs the **mean and variance of a Gaussian distribution**. The latent code used for decoding is a *sample* from that distribution.
{{% /callout %}}

Training maximises the **evidence lower bound** (ELBO), where a reconstruction term encourages the decoder to faithfully recover the input and the KL divergence pulls the learned posteriors towards a standard Gaussian prior (see paper for details). The balance between the two forces the latent space to be both informative and smooth — any noise in the input is naturally absorbed by the width of the posterior.

The practical effect is elegant: clean snapshots get narrow posteriors (the encoder is confident about where they live in the latent space), while noisy or ambiguous snapshots get wider posteriors (the encoder admits its uncertainty). This propagates naturally into downstream uncertainty estimates.

---

## VINDy: Identifying Dynamics as Distributions

### From SINDy to VINDy

Once we have a low-dimensional latent trajectory {{< math >}}$\mathbf{z}(t) \in \mathbb{R}^r${{< /math >}}, we want to find the governing equations of its dynamics. **SINDy** (Sparse Identification of Nonlinear Dynamics) does this by assuming the right-hand side of the latent ODE is a sparse linear combination of candidate functions:

{{< math >}}
$$
\dot{\mathbf{z}}(t) = \boldsymbol{\Theta}(\mathbf{z})\,\boldsymbol{\xi},
$$
{{< /math >}}

where {{< math >}}$\boldsymbol{\Theta}(\mathbf{z}) = [1,\, z_1,\, z_2,\, z_1^2,\, z_1 z_2,\, \dots]${{< /math >}} is a library of candidate functions and {{< math >}}$\boldsymbol{\xi}${{< /math >}} is a sparse coefficient vector — most entries are zero, meaning only a handful of terms actually drive the dynamics. This sparsity makes the identified model interpretable: we get an explicit equation rather than a black-box neural network.

Standard SINDy fits a single coefficient vector, which is fine for clean data but fragile in the presence of noise: small errors in {{< math >}}$\dot{\mathbf{z}}${{< /math >}} can corrupt the identified equations.

**VINDy** replaces the point-estimate coefficients with *distributions*:

{{< math >}}
$$
\boldsymbol{\xi}_i \sim \mathcal{N}(\mu_i, \sigma_i^2), \quad i = 1, \dots, n_\text{lib}.
$$
{{< /math >}}

Each coefficient is now a Gaussian or Laplacian distribution parametrized by a learnable location {{< math >}}$\mu_i${{< /math >}} and scaling factor {{< math >}}$\sigma_i${{< /math >}}. A coefficient with a large mean and small variance corresponds to a term that is confidently important. A coefficient near zero with large variance corresponds to a term that could be pruned — the model is uncertain whether it belongs in the equation at all.

{{< figure src="coeffs3.gif" caption="Evolution of the coefficient distributions during training. Relevant terms converge to tight Laplacian distributions away from zero; irrelevant terms collapse towards zero with small variance, achieving automatic sparsification." numbered="true" id="coeffs">}}

### Training

VINDy is trained jointly with the VAE by adding a dynamics term to the ELBO. Latent codes {{< math >}}$\mathbf{z}(t)${{< /math >}} are sampled from the VAE encoder, numerical time derivatives {{< math >}}$\dot{\mathbf{z}}${{< /math >}} are computed, and the coefficient distributions are optimised so that {{< math >}}$\boldsymbol{\Theta}(\mathbf{z})\,\boldsymbol{\xi}${{< /math >}} matches {{< math >}}$\dot{\mathbf{z}}${{< /math >}} in expectation. A sparsity-promoting prior (analogous to the KL term in the VAE) further encourages most coefficients to shrink to zero, recovering interpretable, parsimonious dynamics.

The animation above captures the key behaviour: as training progresses, most coefficient distributions collapse towards zero while a small subset converge to confident, non-zero values — the framework automatically discovers which terms matter.

---

## VICI: Predictions with Uncertainty Intervals

With a trained VAE and a distribution over governing equations in hand, making predictions is straightforward:

1. **Sample** multiple coefficient vectors {{< math >}}$\boldsymbol{\xi}^{(k)} \sim \mathcal{N}(\boldsymbol{\mu}_\xi, \text{diag}(\boldsymbol{\sigma}^2_\xi))${{< /math >}}.
2. **Integrate** the latent ODE {{< math >}}$\dot{\mathbf{z}} = \boldsymbol{\Theta}(\mathbf{z})\,\boldsymbol{\xi}^{(k)}${{< /math >}} forward from an initial condition, giving a bundle of latent trajectories {{< math >}}$\{\mathbf{z}^{(k)}(t)\}${{< /math >}}.
3. **Decode** each trajectory back to the full state space using the VAE decoder.
4. **Summarise** the resulting ensemble: the mean is the point prediction; the spread gives the uncertainty interval.

This yields not just a single trajectory but a *predictive distribution* over future states. The uncertainty grows naturally for longer horizons or when the initial condition lies away from the training distribution — precisely the situations where a user most needs to know how much to trust the model.

Two sources of uncertainty are represented separately and propagate independently:
- **Data noise** (captured by {{< math >}}$\boldsymbol{\sigma}^2_{\boldsymbol{\phi}}${{< /math >}} of the VAE encoder)
- **Model uncertainty** (captured by {{< math >}}$\boldsymbol{\sigma}^2_\xi${{< /math >}} of the VINDy coefficients)

---

## Example: Reaction–Diffusion System

We applied VENI, VINDy, VICI among others to a reaction–diffusion system that generates rotating spiral waves — a PDE with rich spatio-temporal dynamics that lives on a genuinely low-dimensional manifold.

The full state is a spatial grid with thousands of degrees of freedom. The framework compresses this to just **two latent variables**, finds an interpretable oscillatory equation governing their interaction via VINDy, and uses VICI to predict future states together with uncertainty bounds — closely matching the high-fidelity simulation while providing a measure of confidence in the prediction.

The latent dynamics take the form of a simple nonlinear oscillator, and the VINDy coefficients cleanly identify the relevant coupling terms. This interpretability is a direct consequence of the sparse probabilistic identification: rather than a black-box neural ODE, we obtain an equation we can inspect, simulate cheaply, and reason about physically.

---

## Why Does This Matter?

| Property | Standard ROM | VENI, VINDy, VICI |
|---|---|---|
| Handles noisy data | ✗ | ✓ |
| Interpretable dynamics | Partial | ✓ (sparse equations) |
| Uncertainty quantification | ✗ | ✓ (end-to-end) |
| Generative (can sample states) | ✗ | ✓ |
| Works without knowledge of PDE | ✓ | ✓ |

The combination of interpretability and probabilistic uncertainty quantification is what distinguishes this approach. An engineer using the model gets not just a fast surrogate, but also an explicit equation and a principled confidence estimate — both critical for any safety-relevant application.

---

### Did you find this page helpful? Consider sharing it 🙌
