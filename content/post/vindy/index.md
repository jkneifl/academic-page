---
title: VENI, VINDy, VICI
subtitle: A Generative Approach to Reduced-Order Modeling with Uncertainty Quantification

# Summary for listings and search engines
summary: VENI, VINDy, VICI is an interpretable data-driven framework for building generative reduced order models with embedded uncertainty quantification based on variational autoencoders and SINDy.

# Link this post with a project
projects: []

# Date published
date: '2024-11-12T00:00:00Z'

# Date updated
lastmod: '2024-11-12T00:00:00Z'

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
In scientific computing, we often work with complex systems governed by partial differential equations (PDEs) to model real-world phenomena in fields like fluid dynamics, structural mechanics, and reaction-diffusion systems. These high-dimensional models are incredibly accurate but also computationally demanding, especially when the goal is to run repeated simulations or operate in real-time environments. This is where reduced-order modeling (ROM) comes into play.

However, ROMs typically face several challenges:
	1.	They may lack interpretability, making it hard to understand the underlying physics.
	2.	They often assume access to high-fidelity models, which isn’t always the case.
	3.	They don’t always account for uncertainty, a key factor when dealing with noisy data or unknown system dynamics.

To address these issues, we developed the [[VENI, VINDy, VICI framework](https://arxiv.org/abs/2405.20905)]. This approach combines data-driven machine learning with traditional physics-based insights, enabling reduced-order modeling while quantifying uncertainty.

### Introducing the VENI, VINDy, VICI Framework
{{< figure src="featured.gif" caption=" Overview of the VENI, VINDy, VICI procedure." numbered="true" id="ae">}}

{{< math >}}
  $$
  \mathcal{F}(\mathbf{f}, \nabla\mathbf{f}, \nabla^2\mathbf{f}, \dots, \dot{\mathbf{f}}, \ddot{\mathbf{f}})=0, \quad \mathbf{f}(\mathbf{x},t): \Omega \times \mathcal{T} \to \mathbb{R}^d
  $$
{{< /math >}}

{{% callout note %}}
You can find the implementation of our code [[here](https://github.com/jkneifl/VENI-VINDy-VICI)] .
{{% /callout %}}

Our framework involves three main components:
	•	VENI – Encoding high-dimensional, noisy input data in a low-dimensional space.
	•	VINDy – Discovering nonlinear dynamics in this low-dimensional representation.
	•	VICI – Making predictions and providing uncertainty intervals around them.

By integrating these components, our approach builds models that not only approximate the high-dimensional system dynamics efficiently but also provide confidence levels for those approximations.

### Breaking Down the Framework: VENI, VINDy, and VICI

#### VENI: Variational Encoding of Noisy Inputs

In the first stage, VENI (Variational Encoding of Noisy Inputs) reduces high-dimensional data into a lower-dimensional latent space using a variational autoencoder (VAE). Here, we use noisy training data, as real-world measurements are rarely pristine. By encoding the data through the VAE, we find a compressed but meaningful representation of the system states, essentially capturing the main dynamics while filtering out the noise.

Let’s say we’re working with a reaction-diffusion system. This type of system is commonly used to model processes like chemical reactions or heat diffusion, where we have complex spatio-temporal dynamics. The VENI step would take these complex PDE solutions – perhaps thousands of spatial and temporal data points – and map them onto a much smaller set of latent variables, which still retain the essence of the system’s behavior.

The VAE accomplishes this by maximizing the likelihood of the observed data, given the latent variables, while introducing a probabilistic model to account for the inherent uncertainty in measurements. For example, for each spatial snapshot of the reaction-diffusion system, we obtain a compressed latent representation that reflects the system’s state. These latent variables now serve as the foundation for modeling the dynamics.

#### VINDy: Variational Identification of Nonlinear Dynamics

With our latent representation from VENI, the next step, VINDy (Variational Identification of Nonlinear Dynamics), comes into play. VINDy identifies the governing dynamics in this low-dimensional latent space, allowing us to predict how the latent variables change over time.

This stage is inspired by Sparse Identification of Nonlinear Dynamics (SINDy), but our version goes further. Standard SINDy approximates dynamics using a small subset of possible functions, making it efficient and interpretable. However, it doesn’t account for uncertainty in the data. Our adaptation, VINDy, introduces a probabilistic element, where we treat the dynamics as a distribution over possible governing equations. This allows us to capture both the dynamics and the confidence we have in them.

Returning to the reaction-diffusion system, suppose our reduced-order model has only two latent variables, representing two main modes of the system’s oscillatory behavior. VINDy works by finding a sparse set of functions that describe how these two variables interact over time. We might find, for example, that the interaction between these two variables is governed by simple oscillatory dynamics, coupled with terms that account for nonlinear effects. Since the process is probabilistic, we get a clear sense of which terms are essential (e.g., oscillatory terms) and which are less certain, enabling us to quantify our confidence in each part of the model.

#### VICI: Variational Inference with Certainty Intervals

The final step is VICI (Variational Inference with Certainty Intervals). Once VINDy has identified the system’s dynamics, VICI uses those learned dynamics to make predictions and provide uncertainty bounds. This is essential when extrapolating beyond the data seen in training, especially when we introduce new parameters or initial conditions.

To illustrate, let’s revisit our reaction-diffusion example. Imagine we want to predict the system’s behavior over a longer period or under slightly altered conditions, such as a different rate of chemical reaction. The VICI stage allows us to make these predictions by generating multiple plausible trajectories, each representing a possible system behavior. We then calculate a mean trajectory and its uncertainty interval, capturing the range of possible outcomes.

VICI effectively quantifies our uncertainty across the entire predictive process. If the data were noisy, VICI accounts for that, showing how prediction uncertainty grows over time or how it fluctuates when parameters are adjusted.

#### Application: Reaction-Diffusion System

To test the VENI, VINDy, VICI framework, we applied it to a benchmark reaction-diffusion system. This type of system is governed by PDEs that describe how quantities, like chemical concentrations, diffuse and react over space and time. Specifically, we looked at a reaction-diffusion model that generates spiral waves, common in pattern formation studies.
	1.	Data Preparation: First, we generated simulation data by solving the reaction-diffusion PDEs over a grid. This gave us thousands of data points representing the system’s spatial and temporal evolution. However, to simulate real-world conditions, we added a degree of noise to this data.
	2.	VENI Encoding: Next, VENI took this noisy data and reduced it to a latent space with just two variables. These two variables captured the main oscillatory behavior of the spiral waves, acting as a simplified representation of the complex reaction-diffusion system.
	3.	VINDy Dynamics: With these latent variables, we applied VINDy to uncover the dynamics governing their interaction. The model identified an oscillatory relationship between the two variables, which mirrors the periodic nature of spiral wave formation in the reaction-diffusion system. Importantly, VINDy’s probabilistic nature provided insights into the confidence of each term in our model.
	4.	VICI Prediction: Finally, with VICI, we tested the model’s predictive capabilities. Given initial conditions and parameters, VICI generated possible future states of the system, along with certainty intervals. This is valuable because it allowed us to observe how uncertainty increased over time – a common issue when making predictions in nonlinear systems. The results closely matched the original high-fidelity simulations, affirming the accuracy and reliability of the VENI, VINDy, VICI framework.

#### Why VENI, VINDy, VICI?

The power of VENI, VINDy, VICI lies in its ability to generate interpretable and uncertainty-aware reduced-order models that operate even when data is noisy or sparse. Traditional ROM approaches are typically deterministic, meaning they cannot provide uncertainty estimates. This makes them less reliable when applied to real-world scenarios with imperfect measurements or incomplete information.

By embedding UQ directly into the training and prediction processes, VENI, VINDy, VICI achieves three key advantages:
	1.	Interpretability: The sparse models identified by VINDy are straightforward and aligned with known physics, enabling insights into system behavior.
	2.	Efficiency: The reduced-order nature of the models allows for rapid predictions, suitable for real-time applications or scenarios requiring repeated computations.
	3.	Reliability: VICI’s certainty intervals give confidence bounds around predictions, which is crucial for applications where knowing the range of possible outcomes is as important as knowing the most likely outcome.

#### Conclusion: Transforming Reduced-Order Modeling

The VENI, VINDy, VICI framework represents a significant advancement in reduced-order modeling, combining data-driven insights with a strong foundation in uncertainty quantification. This framework is adaptable, with applications spanning structural mechanics, fluid dynamics, chemical reactions, and beyond.

Our results showcase that VENI, VINDy, VICI can effectively model complex, high-dimensional systems with interpretable, low-dimensional representations and uncertainty-aware predictions, bringing new reliability and trustworthiness to ROM. By addressing the limitations of existing ROM approaches, VENI, VINDy, VICI has the potential to transform how we simulate, analyze, and understand complex systems in science and engineering.

For those interested in exploring VENI, VINDy, VICI, we’ve made the code available on GitHub to encourage further research and applications across diverse fields.

### Did you find this page helpful? Consider sharing it 🙌
