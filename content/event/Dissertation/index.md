---
title: 'Data-driven Surrogate Modeling of Structural Dynamical Systems via Latent Space Representations'

event: Doctoral Exam
# event_url: https://example.org

location: University of Stuttgart, Stuttgart
address:
  street: Pfaffenwaldring 9
  city: Stuttgart
  region: Baden-Wuerttemberg
  postcode: '70569'
  country: Germany

summary: At this event I am presenting the research conducted during my time as a doctoral candidate at the University of Stuttgart 
abstract: |
  Numerical simulations provide powerful predictive capabilities to analyze the dynamic behavior of complex systems and offer insights into physical phenomena that would otherwise remain inaccessible. Consequently, simulation-based analysis of structural dynamical systems plays a crucial role in engineering design, safety verification, and informed decision-making. However, the significant computational cost of conventional high-fidelity (HF) models limits their applicability—especially in multi-query scenarios, real-time contexts, or resource-constrained environments.

  This motivates the development of surrogate models—efficient approximations that emulate the essential behavior of HF simulations while drastically reducing computational overhead. This thesis focuses on non-intrusive, data-driven surrogate modeling approaches that integrate classical numerical methods with modern Machine Learning (ML) techniques, thereby harnessing the synergy between scientific principles and data-driven technologies.

  In detail, the proposed methods are embedded within a unified surrogate modeling framework that combines concepts from Model Order Reduction and ML. At the heart of this framework lies the construction of compact, low-dimensional latent representations in which the dynamics of complex, high-dimensional systems can be efficiently learned and accurately predicted. The thesis explores various strategies for identifying suitable coordinate representations and investigates multiple techniques for modeling latent dynamics, including both black-box approximations and system identification approaches that yield analytic, interpretable equations.

  The application examples cover a broad spectrum of structural dynamics problems: they range from simple academic cases—such as nonlinear pendulums, mass-spring-damper systems, or low-dimensional chaotic systems—to complex, high-dimensional multi-body systems and finite element models. The latter include coupled simulations as well as highly nonlinear and irregular systems involving contact, as found in simplified crash scenarios.

  Key contributions include a systematic benchmarking of dimensionality reduction and regression techniques, a multi-resolution surrogate modeling strategy capable of capturing multi-scale phenomena under varying hardware constraints, as well as structure-preserving model discovery methods. Furthermore, a variational generative modeling framework is developed to enable interpretable surrogate modeling under uncertainty.

  Collectively, these approaches enable the development of surrogate models that are not only computationally efficient and accurate, but also interpretable and deployable in real-time, multi-query, and resource-constrained settings—thereby broadening the applicability and accessibility of numerical simulation in science and engineering.

# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2025-08-29T10:00:00Z'
date_end: '2025-08-29T10:30:00Z'
all_day: false

# Schedule page publish date (NOT talk date).
publishDate: '2025-08-29T00:00:00Z'

authors: []
tags: []

# Is this a featured talk? (true/false)
featured: true

image:
  # caption: ''
  focal_point: Right
  caption: 'Picture by Pierre Kneifl'
  preview_only: false

# links:
#   # - icon: twitters
#     # icon_pack: fab
#     # name: Follow
#     # url: https://twitter.com/georgecushen
#   url_code: ''
#   url_pdf: ''
#   url_slides: ''
#   url_video: ''
# 
# Markdown Slides (optional).
#   Associate this talk with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: 
  - []
---