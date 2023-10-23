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
    content:
      title: Skills
      items:
        - name: ML 
          description: 90%
          icon: r-project
          icon_pack: fab
        - name: Model Reduction
          description: 100%
          icon: chart-line
          icon_pack: fas
        - name: Photography
          description: 10%
          icon: camera-retro
          icon_pack: fas
  - block: experience
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
        # - title: Visiting Researcher
        #   company: Department of Civil and Environmental Engineering
        #   company_url: https://www.dica.polimi.it/?lang=en
        #   company_logo: polimi
        #   location: Polytechnic University of Milan, Milan, Italy
        #   date_start: '2023-09-01'
        #   date_end: '2023-09-01'
        #   description: Development of a reduced-order modeling with uncertainty quantification framework using generative AI algorithms.
        # - title: Research Intern
        #   company: Artificial Intelligence Institute in Dynamic Systems 
        #   company_url: 'https://dynamicsai.org'
        #   company_logo: UW
        #   location: Seattle (US)
        #   date_start: '2022-08-01'
        #   date_end: '2022-11-16'
        #   description: Development of a multi-hierarchic surrogate modeling approach using graph convolutional neural networks and mesh simplification.
    design:
      columns: '2'
  - block: collection
    id: featured
    content:
      title: Featured Publications
      filters:
        folders:
          - publication
        featured_only: true
    design:
      columns: '2'
      view: card
  - block: collection
    content:
      title: Publications
      text: |-
        {{% callout note %}}
        Quickly discover relevant content by [filtering publications](./publication/).
        {{% /callout %}}
      filters:
        folders:
          - publication
        exclude_featured: true
    design:
      columns: '2'
      view: citation
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - event
    design:
      columns: '2'
      view: compact
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle:
      text: |-
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam mi diam, venenatis ut magna et, vehicula efficitur enim.
      # Contact (add or remove contact options as necessary)
      email: jonas.kneifl@itm.uni-stuttgart.de
      address:
        street: Pfaffenwaldring 9
        city: Stuttgart
        region: Baden-Württemberg
        postcode: '70569'
        country: Germany
        country_code: DE
      directions: ITM Room 3.101
      contact_links:
        - icon: linkedin
          icon_pack: fas
          name: Zoom Me
          link: 'https://linkedin.com/in/jonas-kneifl-918ba4194'
      # Automatically link email and phone or display as text?
      autolink: true
      # Email form provider
      form:
        provider: netlify
        formspree:
          id:
        netlify:
          # Enable CAPTCHA challenge to reduce spam?
          captcha: false
    design:
      columns: '2'
---
