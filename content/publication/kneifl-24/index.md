---
title: Multi-hierarchical surrogate learning for explicit structural dynamical systems using graph convolutional neural networks

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

date: '2024-5-10'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2024-05-10T13:32:09.789877Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- '0'
publication: '*Computational Mechanics*'
publication_short: ''

doi: 10.1007/s00466-024-02553-6

abstract: Highly nonlinear dynamic finite element simulations using explicit time integration are particularly valuable tools for structural analysis in fields like automotive, aerospace, and civil engineering, or in the study of injury biomechanics. However, such state-of-the-art simulation models demand significant computational resources. Conventional data-driven surrogate modeling approaches address this by evolving the dynamics on low-dimensional embeddings, yet the majority of them operate directly on high-resolution data obtained from numerical discretizations, making them costly and unsuitable for adaptive resolutions or for handling information flow over large spatial distances. We therefore propose a multi-hierarchical framework for the structured creation of a series of surrogate models at different resolutions. Macroscale features are captured on coarse surrogates, while microscale effects are resolved on finer ones, while leveraging transfer learning to pass information between scales. The objective of this study is to develop efficient surrogates for a kart frame model in a frontal impact scenario. To achieve this, its mesh is simplified to obtain multi-resolution representations of the kart. Subsequently, a graph-convolutional neural network-based surrogate learns parameter-dependent low-dimensional latent dynamics on the coarsest representation. Following surrogates are trained on residuals using finer resolutions, allowing for multiple surrogates with varying hardware requirements and increasing accuracy.

# Summary. An optional shortened abstract.
summary: ''

tags:
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
  caption: Multi-hierarchic surrogate modeling scheme
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
  url: https://doi.org/10.1007/s00466-024-02553-6
---

