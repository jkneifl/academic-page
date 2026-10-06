---
# Leave the homepage title empty to use the site title
title:
date: 2023-10-08
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Biography
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
  - block: features
    id: research
    content:
      title: Research
      items:
        - name: '[Physics in Generative Models](/project/physics-interpretability/)'
          description: Probing and steering what video generation models internally represent about physics.
          icon: microscope
          icon_pack: fas
        - name: '[Interpretable System Identification](/project/system-identification/)'
          description: Learning sparse, physically structured, and uncertainty-aware dynamics in latent spaces.
          icon: square-root-alt
          icon_pack: fas
        - name: '[Data-Driven Surrogate Modeling](/project/surrogate-modeling/)'
          description: Fast, non-intrusive reduced-order models that replace expensive simulations.
          icon: tachometer-alt
          icon_pack: fas
  - block: experience
    id: experience
    content:
      title: Experience
      # Date format for experience
      #   Refer to https://wowchemy.com/docs/customization/#date-format
      date_format: Jan 2006
      # Experiences.
      #   Add/remove as many `experience` items below as you like.
      #   Required fields are `title`, `company`, and `date_start`.
      #   Leave `date_end` empty if it's your current employer.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - title: Postdoctoral Researcher
          company: IDEAS Research Institute
          company_url: https://www.ideas.edu.pl/en/
          company_logo: Ideas-Research-Institute-White-1
          location: Warsaw, Poland
          date_start: '2026-05-01'
          date_end: ''
          description: Fundamental AI research in the fields of Computer Vision and Generative AI.
        - title: Research Associate
          company: Institute of Engineering and Computational Mechanics 
          company_url: https://www.itm.uni-stuttgart.de/en/
          company_logo: Uni_stuttgart_logo
          location: University of Stuttgart, Stuttgart, Germany
          date_start: '2020-06-01'
          date_end: '2025-10-30'
          description: Researcher in the field of Scientific Machine Learning within the cluster of excellence "Data-Integrated Simulation Science (SimTech)" and lecture assistant.
        - title: Visiting Researcher
          company: Department of Civil and Environmental Engineering
          company_url: https://www.dica.polimi.it/?lang=en
          company_logo: polimi
          location: Polytechnic University of Milan, Milan, Italy
          date_start: '2023-09-01'
          date_end: '2023-09-30'
          description: Development of a reduced-order modeling with uncertainty quantification framework using generative AI algorithms.
        - title: Research Intern
          company: Artificial Intelligence Institute in Dynamic Systems
          company_url: 'https://dynamicsai.org/'
          company_logo: UW
          location: University of Washington, Seattle (US)
          date_start: '2022-08-01'
          date_end: '2022-11-16'
          description: Development of a multi-hierarchic surrogate modeling approach using graph convolutional neural networks and mesh simplification.
      columns: '2'
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Filter on criteria
      filters:
        folders:
          - post
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: compact
      columns: '2'
  # - block: collection
  #   id: featured
  #   content:
  #     title: Featured Publications
  #     filters:
  #       folders:
  #         - publication
  #       featured_only: true
  #   design:
  #     columns: '2'
  #     view: card
  - block: collection
    id: talks
    content:
      title: Selected Talks
      filters:
        folders:
          - event
        featured_only: true
      archive:
        enable: true
        text: All talks
    design:
      columns: '2'
      view: compact
  - block: collection
    id: publications
    content:
      title: Featured Publications
      # text: |-
      #   {{% callout note %}}
      #   Quickly discover relevant content by [filtering publications](./publication/).
      #   {{% /callout %}}
      filters:
        folders:
          - publication
        featured_only: true
      archive:
        enable: true
        text: All publications
    design:
      columns: '2'
      view: compact
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle:
      # text: |-
      #   Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam mi diam, venenatis ut magna et, vehicula efficitur enim.
      # Contact (add or remove contact options as necessary)
      email: jonas.kneifl@ideas.edu.pl
      address:
        city: Warsaw
        region: Masovian Voivodeship
        country: Poland
        country_code: PL
      contact_links:
        - icon: linkedin
          icon_pack: fab
          name: Connect on
          link: 'https://www.linkedin.com/in/jonas-kneifl/'
      # Automatically link email and phone or display as text?
      autolink: true
      # Email form provider
      # form:
      #   provider: netlify
      #   formspree:
      #     id:
      #   netlify:
      #     # Enable CAPTCHA challenge to reduce spam?
      #     captcha: false
    design:
      columns: '2'
---
