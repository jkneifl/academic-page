---
title: Physics-informed Neural Networks-based Model Predictive Control for Multi-link
  Manipulators

# Authors
# A YAML list of author names
# If you created a profile for a user (e.g. the default `admin` user at `content/authors/admin/`), 
# write the username (folder name) here, and it will be replaced with their full name and linked to their profile.
authors:
- Jonas Nicodemus
- Jonas Kneifl
- Jörg Fehr
- Benjamin Unger

# Author notes (such as 'Equal Contribution')
# A YAML list of notes for each author in the above `authors` list
author_notes: []

date: '2021-01-01'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2023-10-26T15:15:44.318661Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- article-journal

# Publication name and optional abbreviated publication name.
publication: '*ArXiv e-print 2109.10793*'
publication_short: ''

doi: ''

abstract: We discuss nonlinear model predictive control (NMPC) for multi-body dynamics
  via physics-informed machine learning methods. Physics-informed neural networks
  (PINNs) are a promising tool to approximate (partial) differential equations. PINNs
  are not suited for control tasks in their original form since they are not designed
  to handle variable control actions or variable initial values. We thus present the
  idea of enhancing PINNs by adding control actions and initial conditions as additional
  network inputs. The high-dimensional input space is subsequently reduced via a sampling
  strategy and a zero-hold assumption. This strategy enables the controller design
  based on a PINN as an approximation of the underlying system dynamics. The additional
  benefit is that the sensitivities are easily computed via automatic differentiation,
  thus leading to efficient gradient-based algorithms. Finally, we present our results
  using our PINN-based MPC to solve a tracking problem for a complex mechanical system,
  a multi-link manipulator.

# Summary. An optional shortened abstract.
summary: ''

tags:
- exc2075 myown pn4 preprint

# Display this page in a list of Featured pages?
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
  caption: ''
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
  url: https://arxiv.org/abs/2109.10793
---

Add the **full text** or **supplementary notes** for the publication here using Markdown formatting.
