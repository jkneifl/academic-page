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

date: '2022-01-01'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2023-10-26T15:27:19.812390Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- article-journal

# Publication name and optional abbreviated publication name.
publication: '*IFAC-PapersOnLine*'
publication_short: ''

doi: https://doi.org/10.1016/j.ifacol.2022.09.117

abstract: We discuss nonlinear model predictive control (MPC) for multi-body dynamics
  via physics-informed machine learning methods. In more detail, we use a physics-informed
  neural networks (PINNs)-based MPC to solve a tracking problem for a complex mechanical
  system, a multi-link manipulator. PINNs are a promising tool to approximate (partial)
  differential equations but are not suited for control tasks in their original form
  since they are not designed to handle variable control actions or variable initial
  values. We thus follow the strategy of Antonelo et al. (arXiv:2104.02556, 2021)
  by enhancing PINNs with adding control actions and initial conditions as additional
  network inputs. Subsequently, the high-dimensional input space is reduced via a
  sampling strategy and a zero-hold assumption. This strategy enables the controller
  design based on a PINN as an approximation of the underlying system dynamics. The
  additional benefit is that the sensitivities are easily computed via automatic differentiation,
  thus leading to efficient gradient-based algorithms for the underlying optimal control
  problem.

# Summary. An optional shortened abstract.
summary: ''

tags:
- Physics-informed Machine Learning
- Model Predictive Control
- Surrogate Model
- Mechanical System
- Real-time Control

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
  url: https://www.sciencedirect.com/science/article/pii/S2405896322013118
---

Add the **full text** or **supplementary notes** for the publication here using Markdown formatting.
