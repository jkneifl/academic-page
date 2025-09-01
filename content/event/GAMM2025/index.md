---
title: 'VENI, VINDy, VICI a generative reduced-order modeling framework with uncertainty quantification'

event: GAMM Annual Meeting 2025
# event_url: https://example.org

location: Poznan University of Technology, Poznan
address:
  street:
  city: Poznan
  region: 
  postcode: 
  country: Poland

summary: An interpretable data-driven framework for building generative reduced order models with embedded uncertainty quantification.
abstract: |
  Generative models are transforming science and engineering by enabling efficient synthetization and exploration of new scenarios for complex physical phenomena with minimal cost. 
  Although they provide uncertainty-aware predictions to support decision making, they typically lack physical consistency, which is the backbone of computational science.
  Hence, we propose VENI, VINDy, VICI -- a novel physical generative framework that integrates data-driven system identification into a probabilistic modeling approach to construct physically consistent and efficient Reduced-Order Models (ROMs) with Uncertainty Quantification (UQ). 

  First, VENI (Variational Encoding of Noisy Inputs) employs variational autoencoders to identify reduced coordinates from high-dimensional, noisy measurements. 
  Simultaneously, VINDy (Variational Identification of Nonlinear Dynamics) extends sparse system identification methods by embedding probabilistic modeling into the discovery process.
  Last, VICI (Variational Inference with Credibility Intervals) enables efficient generation of full-time solutions and provides UQ for unseen parameters and initial conditions.
  We demonstrate the performance of the framework across chaotic and high-dimensional nonlinear systems.

# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
date: '2025-04-07T09:00:00Z'
date_end: '2025-04-11T15:00:00Z'
all_day: false

# Schedule page publish date (NOT talk date).
publishDate: '2025-04-01T00:00:00Z'

authors: []
tags: []

# Is this a featured talk? (true/false)
featured: false

image:
  # caption: ''
  focal_point: Right
  caption: ''
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