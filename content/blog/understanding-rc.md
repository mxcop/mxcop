+++
title = "Understanding Radiance Cascades"
description = "Recently I started to look into Radiance Cascades, here's what I've learned so far."
date = 2024-08-17

[[extra.tags]]
name = "graphics"
color = "emerald"

[[extra.tags]]
name = "radiance cascades"
color = "lime"
+++

At it's core Radiance Cascades is a method for efficiently computing & storing a radiance field.

## Equations

Final pass:

$ f_r(x) = \int_\Omega{L(x,\omega) * \cos(\theta_x) * d\omega} $

$ f_r(x) = \sum_i{L(x,\vec{\omega_i}) * (\vec{n} \cdot \vec{\omega_i})} $

Cook-Torrance Microfacet BRDF:

$$ f_r(v,l) = \frac{\rho_d}{\pi} + \frac{F(v, h) * D(h) * G(l, v)}{4 * (n \cdot l) * (n \cdot v)} $$
