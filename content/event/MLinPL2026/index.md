---
title: 'FPS: Frames, Physics, and Steering – Probing and Controlling Physical Observables in Video Diffusion Models'

event: ML in PL Conference 2026
event_url: https://conference.mlinpl.org/2026/program#contributed-talk-22

location: Copernicus Science Centre, Hall B
address:
  street: Wybrzeże Kościuszkowskie 20
  city: Warsaw
  postcode: '00-390'
  country: Poland

summary: Do video diffusion models know physics? Probing their activations for velocity, momentum, and gravity — and steering them.
abstract: |
  Video-generation models produce strikingly realistic sequences and are increasingly proposed as world models for domains like automotive and robotics, where users need consistent long rollouts and generalization to unseen configurations. That promise rests on physics: a model that has internalized physical laws should be able to roll out coherently and support meaningful downstream tasks. Yet we still lack a general understanding of whether video models actually learn physics—and how to extract it if they do.

  Existing work only judges whether a model can produce plausible-looking video—but plausibility is not understanding. Instead, we inspect the model's internal representations for ground-truth physical quantities, shifting the question from "does it look right?" to "does it know?". We do so by probing its activations with linear decoders for physical observables such as velocity, momentum, and gravity.

  However, probing against ground truth is not straightforward: diffusion models synthesize video from noise, so a from-scratch generation is uncontrolled and has no known physical parameters to compare against. We therefore work backwards: using a physics simulator, we generate ground-truth videos of various scenes such as elastic collisions or projectile motion while keeping full control over their parameters. Running each video through the forward diffusion pass adds noise to create a starting point for the reverse process. As the model denoises the input, we capture the intermediate activations that form its internal representation of the scene. With ground-truth physics in hand, linear probes then reveal whether the model has genuinely internalized these physical observables.

  The learned representations turn out to be highly structured: the observables are linearly decodable, and the probes extrapolate across objects and scenes—evidence that the model interpolates physics, not pixels. This information is also sharply localized in space and time: physical properties are read best from object-centric, frame-aligned tokens rather than the full latent space. Exploiting this, our structured probe avoids the spatio-temporal pooling of prior work, making it more memory- and compute-efficient while yielding better results. Crucially, the representation is not only readable but manipulable: adding differences of activations lets us flip the direction of objects in generated videos, indicating that the encoded physics is causally used rather than incidental—with steering along the learned probe directions as a natural next step.

  Alongside these successes, we surface clear limits in current physical understanding, and report results across several state-of-the-art open-weight video models, focusing on their largest variants.

date: '2026-10-10T09:30:00+02:00'
date_end: '2026-10-10T10:00:00+02:00'
all_day: false

publishDate: '2026-10-01T00:00:00Z'

authors:
  - admin
tags:
  - Mechanistic Interpretability
  - Generative AI

featured: true

projects:
  - physics-interpretability
---
