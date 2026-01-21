---
title: Data-driven identification of latent port-Hamiltonian systems

# Authors
# A YAML list of author names
# If you created a profile for a user (e.g. the default `admin` user at `content/authors/admin/`), 
# write the username (folder name) here, and it will be replaced with their full name and linked to their profile.
authors:
- Johannes Rettberg
- Jonas Kneifl
- Julius Herb
- Patrick Buchfink
- Jörg Fehr
- Bernard Haasdonk

# Author notes (such as 'Equal Contribution')
# A YAML list of notes for each author in the above `authors` list
author_notes: []

date: '2025-11-01'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2026-01-08T14:10:20.871120Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- article-journal

# Publication name and optional abbreviated publication name.
publication: '*Comput. Sci. Eng.*'
publication_short: ''

doi: ''

abstract: 'Conventional physics-based modeling techniques involve high effort, e.g.,
  time and expert knowledge, while data-driven methods often lack interpretability,
  structure, and sometimes reliability. To mitigate this, we present a data-driven
  system identification framework that derives models in the port-Hamiltonian (pH)
  formulation. This formulation is suitable for multi-physical systems while guaranteeing
  the useful system theoretical properties of passivity and stability. Our framework
  combines linear and nonlinear reduction with structured, physics-motivated system
  identification. In this process, high-dimensional state data obtained from possibly
  nonlinear systems serves as input for an autoencoder, which then performs two tasks:
  (i) nonlinearly transforming and (ii) reducing this data onto a low-dimensional
  latent space. In this space, a linear pH system that satisfies the pH properties
  per construction is parameterized by the weights of a neural network. The mathematical
  requirements are met by defining the pH matrices through Cholesky factorizations.
  The neural networks that define the coordinate transformation and the pH system
  are identified in a joint optimization process to match the dynamics observed in
  the data while defining a linear pH system in the latent space. The learned, low-dimensional
  pH system can describe even nonlinear systems and is rapidly computable due to its
  small size. The method is exemplified by a parametric mass-spring-damper and a nonlinear
  pendulum example, as well as the high-dimensional model of a disc brake with linear
  thermoelastic behavior.'

# Summary. An optional shortened abstract.
summary: ''

tags: []

# Display this page in a list of Featured pages?
featured: false

# Links
url_pdf: ''
url_code: 'https://github.com/Institute-Eng-and-Comp-Mechanics-UStgt/ApHIN'
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
---

Add the **full text** or **supplementary notes** for the publication here using Markdown formatting.
