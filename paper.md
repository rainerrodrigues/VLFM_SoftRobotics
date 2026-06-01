---
title: 'VLFM_SoftRobotics: A High-Performance Differentiable Framework for Real-Time Soft Actuator Control'
tags:
  - Julia
  - Soft Robotics
  - Gaussian Processes
  - Model Predictive Control
authors:
  - name: Rainer Rodrigues
    orcid: 0009-0000-4246-8339
    affiliation: Independent Researcher
date: 1 June 2026
bibliography: paper.bib
---

# Summary
Soft and deformable robots have infinite-degree-of-freedom dynamics that defy traditional analytical modeling. `VLFM_SoftRobotics` is a Julia package designed to resolve this bottleneck using a continuous-time Variational Latent Force Model (VLFM)[1]. 

# Statement of Need
While exact Gaussian Process (GP) regression scales as $O(N^3)$, rendering real-time closed-loop control impossible on embedded platforms[2], this software integrates sparse recursive state-space methods directly inside continuous-time differential operators[3]. Leveraging `Turing.jl` for online NUTS sampling and `Zygote.jl` for source-to-source automatic differentiation[4], our framework achieves ~480 Hz inference with a trajectory-tracking MSE of 0.0107[5].

# Mathematical Architecture
The underlying physics is modeled as a non-linear mass-spring-damper system[1]:
$$m\ddot{x}(t) + c\dot{x}(t) + kx(t) + 0.1kx^3(t) = u(t) + f_{\text{latent}}(t)$$

The unmodeled dynamics are captured via a zero-mean Gaussian Process driven by a Squared Exponential kernel[1]:
$$\kappa(t, t') = \sigma_{\text{gp}}^2 \exp\left(-\frac{(t - t')^2}{2\ell^2}\right)$$

# References