---
title: 'Does Physics Live in the Activations? Localizing Physical Quantities in Video Diffusion Models'
authors:
- admin
- Jakub Skalski
- Bartłomiej Twardowski
- Kamil Deja
date: '2026-10-02'
publishDate: '2026-10-15T00:00:00Z'
publication_types:
- article
publication: '*arXiv preprint* arXiv:2610.03154'
abstract: 'Video generation models produce strikingly realistic sequences and are increasingly proposed as world models, yet recent benchmarks reveal pronounced deficits in their physical reasoning. This raises the question of whether these models internalize physical principles or merely reproduce familiar motion patterns. We address this by probing internal representations of video Diffusion Transformers (DiTs) for simulator-derived ground-truth physical quantities spanning kinematic motion and rigid-body dynamics under gravity and contact. We find that these quantities are linearly decodable with high accuracy early in the denoising process, substantially outperforming a baseline decoded directly from the model''s own noised latents, indicating that the relevant physical information is actively constructed during denoising rather than already present in the input. Additionally, we show that activations at on-object tokens carry the relevant physical information and that quantities defined over multiple frames are readable from single latent frames. Hence, information is sharply localized within the token sequence and is computed globally but stored locally. The probes further show partial extrapolation, transferring to scene variations and object configurations outside their training regime, so what they read is not simply a correlate of the scenes they were fit on. When fitted directly in the full-resolution activation space, the probing directions can serve as steering vectors to change the model''s output.'
tags:
- Mechanistic Interpretability
- Video Diffusion Models
- Generative AI
featured: true
url_pdf: 'https://arxiv.org/pdf/2610.03154'
links:
- name: arXiv
  url: https://arxiv.org/abs/2610.03154
image:
  caption: ''
  focal_point: ''
  preview_only: false
projects: [physics-interpretability]
---
