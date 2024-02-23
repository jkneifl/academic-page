---
title: Multi-Hierarchical Surrogate Learning for Structural Dynamical Crash Simulations
  Using Graph Convolutional Neural Networks

# Authors
# A YAML list of author names
# If you created a profile for a user (e.g. the default `admin` user at `content/authors/admin/`), 
# write the username (folder name) here, and it will be replaced with their full name and linked to their profile.
authors:
- Jonas Kneifl
- Jörg Fehr
- Steven L. Brunton
- J. Nathan Kutz
author_notes: []

date: '2024-01-01'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2024-02-23T13:47:48.327789Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- 'preprint'
publication: '*arXiv*'
publication_short: ''

doi: 10.48550/ARXIV.2402.09234

abstract: Crash simulations play an essential role in improving vehicle safety, design
  optimization, and injury risk estimation. Unfortunately, numerical solutions of
  such problems using state-of-the-art high-fidelity models require significant computational
  effort. Conventional data-driven surrogate modeling approaches create low-dimensional
  embeddings for evolving the dynamics in order to circumvent this computational effort.
  Most approaches directly operate on high-resolution data obtained from numerical
  discretization, which is both costly and complicated for mapping the flow of information
  over large spatial distances. Furthermore, working with a fixed resolution prevents
  the adaptation of surrogate models to environments with variable computing capacities,
  different visualization resolutions, and different accuracy requirements. We thus
  propose a multi-hierarchical framework for structurally creating a series of surrogate
  models for a kart frame, which is a good proxy for industrial-relevant crash simulations,
  at different levels of resolution. For multiscale phenomena, macroscale features
  are captured on a coarse surrogate, whereas microscale effects are resolved by finer
  ones. The learned behavior of the individual surrogates is passed from coarse to
  finer levels through transfer learning. In detail, we perform a mesh simplification
  on the kart model to obtain multi-resolution representations of it. We then train
  a graph-convolutional neural network-based surrogate that learns parameter-dependent
  low-dimensional latent dynamics on the coarsest representation. Subsequently, another,
  similarly structured surrogate is trained on the residual of the first surrogate
  using a finer resolution. This step can be repeated multiple times. By doing so,
  we construct multiple surrogates for the same system with varying hardware requirements
  and increasing accuracy.

# Summary. An optional shortened abstract.
summary: ''

tags:
- Machine Learning (cs.LG)
- Dynamical Systems (math.DS)
- 'FOS: Computer and information sciences'
- 'FOS: Computer and information sciences'
- 'FOS: Mathematics'
- 'FOS: Mathematics'
featured: false

# Links
url_pdf: ''
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

# Publication image
# Add an image named `featured.jpg/png` to your page's folder then add a caption below.
image:
  caption: 'Multi-hierarchic surrogate modeling scheme'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects: ['internal-project']` links to `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []
links:
- name: URL
  url: https://doi.org/10.48550/ARXIV.2402.09234
---

