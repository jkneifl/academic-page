---
title: Data-driven identification of latent port-Hamiltonian systems
authors:
- Johannes Rettberg
- Jonas Kneifl
- Julius Herb
- Patrick Buchfink
- Jörg Fehr
- Bernard Haasdonk
date: '2025-11-01'
publishDate: '2026-01-21T13:37:33.235808Z'
publication_types:
- article-journal
publication: '*Comput. Sci. Eng.*'
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
---
