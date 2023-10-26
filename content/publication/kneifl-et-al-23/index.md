---
title: Low-dimensional data-based surrogate model of a continuum-mechanical musculoskeletal
  system based on non-intrusive model order reduction

# Authors
# A YAML list of author names
# If you created a profile for a user (e.g. the default `admin` user at `content/authors/admin/`), 
# write the username (folder name) here, and it will be replaced with their full name and linked to their profile.
authors:
- Jonas Kneifl
- David Rosin
- Okan Avci
- Oliver Röhrle
- Jörg Fehr

# Author notes (such as 'Equal Contribution')
# A YAML list of notes for each author in the above `authors` list
author_notes: []

date: '2023-06-01'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2023-10-26T15:27:19.792850Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- article-journal

# Publication name and optional abbreviated publication name.
publication: '*Archive of Applied Mechanics*'
publication_short: ''

doi: 10.1007/s00419-023-02458-5

abstract: Over the last decades, computer modeling has evolved from a supporting tool
  for engineering prototype design to an ubiquitous instrument in non-traditional
  fields such as medical rehabilitation. This area comes with unique challenges, e.g.
  the complex modeling of soft tissue or the analysis of musculoskeletal systems.
  Conventional modeling approaches like the finite element (FE) method are computationally
  costly when dealing with such models, limiting their usability for real-time simulation
  or deployment on low-end hardware, if the model at hand cannot be simplified without
  losing its expressiveness. Non-traditional approaches such as surrogate modeling
  using data-driven model order reduction are used to make complex high-fidelity models
  more widely available regardless. They often involve a dimensionality reduction
  step, in which the high-dimensional system state is transformed onto a low-dimensional
  subspace or manifold, and a regression approach to capture the reduced system behavior.
  While most publications focus on one dimensionality reduction, such as principal
  component analysis (PCA) (linear) or autoencoder (nonlinear), we consider and compare
  PCA, kernel PCA, autoencoders, as well as variational autoencoders for the approximation
  of a continuum-mechanical system. In detail, we demonstrate the benefits of the
  surrogate modeling approach on a complex musculoskeletal system of a human upper-arm
  with severe nonlinearities and physiological geometry. We consider both, the model's
  deformation and the internal stress as the two main quantities of interest in a
  FE context. By doing so we are able to create computationally low-cost surrogate
  models which capture the system behavior with high approximation quality and fast
  evaluations.

# Summary. An optional shortened abstract.
summary: ''

tags: []

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
  url: https://doi.org/10.1007/s00419-023-02458-5
---

Add the **full text** or **supplementary notes** for the publication here using Markdown formatting.
