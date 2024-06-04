---
title: VENI, VINDy, VICI – a variational reduced-order modeling framework with uncertainty quantification

# Authors
# A YAML list of author names
# If you created a profile for a user (e.g. the default `admin` user at `content/authors/admin/`), 
# write the username (folder name) here, and it will be replaced with their full name and linked to their profile.
authors:
- Paolo Conti
- Jonas Kneifl
- Andrea Manzoni
- Attilio Frangi
- Jörg Fehr
- Steven L. Brunton
- J. Nathan Kutz
author_notes: []

date: '2024-06-03'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2024-06-03T13:47:48.327789Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- 'preprint'
publication: '*arXiv*'
publication_short: ''

doi: 10.48550/ARXIV.2405.20905

abstract: The simulation of many complex phenomena in engineering and science requires solving expensive, high-dimensional systems of partial differential equations (PDEs). To circumvent this, reduced-order models (ROMs) have been developed to speed up computations. However, when governing equations are unknown or partially known, typically ROMs lack interpretability and reliability of the predicted solutions. In this work we present a data-driven, non-intrusive framework for building ROMs where the latent variables and dynamics are identified in an interpretable manner and uncertainty is quantified. Starting from a limited amount of high-dimensional, noisy data the proposed framework constructs an efficient ROM by leveraging variational autoencoders for dimensionality reduction along with a newly introduced, variational version of sparse identification of nonlinear dynamics (SINDy), which we refer to as Variational Identification of Nonlinear Dynamics (VINDy). In detail, the method consists of Variational Encoding of Noisy Inputs (VENI) to identify the distribution of reduced coordinates. Simultaneously, we learn the distribution of the coefficients of a pre-determined set of candidate functions by VINDy. Once trained offline, the identified model can be queried for new parameter instances and new initial conditions to compute the corresponding full-time solutions. The probabilistic setup enables uncertainty quantification as the online testing consists of Variational Inference naturally providing Certainty Intervals (VICI). In this work we showcase the effectiveness of the newly proposed VINDy method in identifying interpretable and accurate dynamical system for the Roessler system with different noise intensities and sources. Then the performance of the overall method - named VENI, VINDy, VICI - is tested on PDE benchmarks including structural mechanics and fluid dynamics.

# Summary. An optional shortened abstract.
summary: 'The proposed framework discovers probabilistic governing equations from high-dimensional data in a low-dimensional latent space'

tags:
- Machine Learning (cs.LG)
- Dynamical Systems (math.DS)
- 'FOS: Computer and information sciences'
- 'FOS: Mathematics'
featured: true

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
  caption: 'VENI, VINDy, VICI'
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
  url: https://doi.org/10.48550/ARXIV.2405.20905
---

