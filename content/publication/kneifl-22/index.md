---
title: Real-time Human Response Prediction Using a Non-intrusive Data-driven Model
  Reduction Scheme

# Authors
# A YAML list of author names
# If you created a profile for a user (e.g. the default `admin` user at `content/authors/admin/`), 
# write the username (folder name) here, and it will be replaced with their full name and linked to their profile.
authors:
- J. Kneifl
- J. Hay
- J. Fehr
author_notes: []

date: '2022-01-01'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2024-08-16T13:32:09.711415Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- '2'
publication: '*IFAC-PapersOnLine*'
publication_short: ''

doi: https://doi.org/10.1016/j.ifacol.2022.09.109

abstract: Recent research in non-intrusive data-driven model order reduction (MOR)
  enabled accurate and efficient approximation of parameterized ordinary differential
  equations (ODEs). However, previous studies have focused on constant parameters,
  whereas time-dependent parameters have been neglected. The purpose of this paper
  is to introduce a novel two-step MOR scheme to tackle this issue. In a first step,
  classic MOR approaches are applied to calculate a low-dimensional representation
  of high-dimensional ODE solutions, i.e., to extract the most important features
  of simulation data. Based on this representation, a long short-term memory (LSTM)
  is trained to predict the reduced dynamics iteratively in a second step considering
  the parameters at the respective time step. The potential of this approach is demonstrated
  on an occupant model within a car driving scenario. The reduced model's response
  to time-varying accelerations matches the reference data with high accuracy for
  a limited amount of time. Furthermore, real-time capability is achieved. Accordingly,
  it is concluded that the presented method is well suited to approximate parameterized
  ODEs and can handle time-dependent parameters in contrast to common methods.

# Summary. An optional shortened abstract.
summary: ''

tags:
- Model Reduction
- Machine Learning
- Occupant Safety
- Human Body Modeling
- Parameterized Ordinary Differential Equations
- Long Short-Term Memory
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
  url: https://www.sciencedirect.com/science/article/pii/S2405896322013015
---
