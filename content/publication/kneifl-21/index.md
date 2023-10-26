---
title: A nonintrusive nonlinear model reduction method for structural dynamical problems
  based on machine learning

# Authors
# A YAML list of author names
# If you created a profile for a user (e.g. the default `admin` user at `content/authors/admin/`), 
# write the username (folder name) here, and it will be replaced with their full name and linked to their profile.
authors:
- Jonas Kneifl
- Dennis Grunert
- Joerg Fehr

# Author notes (such as 'Equal Contribution')
# A YAML list of notes for each author in the above `authors` list
author_notes: []

date: '2021-01-01'

# Date to publish webpage (NOT necessarily Bibtex publication's date).
publishDate: '2023-10-26T15:27:19.799546Z'

# Publication type.
# A single CSL publication type but formatted as a YAML list (for Hugo requirements).
publication_types:
- article-journal

# Publication name and optional abbreviated publication name.
publication: '*International Journal for Numerical Methods in Engineering*'
publication_short: ''

doi: https://doi.org/10.1002/nme.6712

abstract: Abstract Model order reduction (MOR) has become one of the most widely used
  tools to create efficient surrogate models for time-critical applications. For nonlinear
  models, however, linear MOR approaches are only practicable to a limited extent.
  Nonlinear approaches, on the contrary, often require intrusive manipulations of
  the used simulation code. Hence, nonintrusive MOR approaches using classic model
  order reduction along with machine learning (ML) algorithms can provide remedy.
  Such approaches have drawn a lot of attention in the recent years. They rely on
  the idea to learn the dynamics not in a high dimensional but in a reduced space,
  that is, they predict the discrete sequence of reduced basis' coefficients. Open
  questions are the suitability of such methods in the field of structural dynamics
  and the best choice of the used ML algorithm. Both are addressed in this article
  in addition to the integration of the methodology into a modular and flexible framework
  that can effortless be adapted to various requirements. By applying the methodology
  to a dynamic mechanical system, accurate surrogate models are received, which can
  speed up the simulation time significantly, while still providing high-quality state
  approximations.

# Summary. An optional shortened abstract.
summary: ''

tags:
- black box model
- data-based model reduction
- machine learning
- nonlinear behavior
- structural dynamics
- surrogate model

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
  url: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.6712
---

Add the **full text** or **supplementary notes** for the publication here using Markdown formatting.
