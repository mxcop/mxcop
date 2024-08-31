+++
title = "Understanding Radiance Cascades"
description = "Recently I started to look into Radiance Cascades, here's what I've learned so far."
authors = [ "Max &lt;mxcop&gt;" ]
date = 2024-08-17

[[extra.tags]]
name = "graphics"
color = "emerald"

[[extra.tags]]
name = "radiance cascades"
color = "lime"
+++

## Introduction

At it's core Radiance Cascades *(abbreviated as RC)* is a method for efficiently representing a <span class="highlight">radiance field</span>.  
In the article we will make an interesting set of observations about the nature of penumbrae which make RC possible.

> For the sake of simplicity I will keep the explenations 2D, however RC can be expanded into the third dimension aswell.

So, what can RC in 2D *(also referred to as Flatland)* achieve?  
My implementation is able to compute <span class="highlight">diffuse global illumination</span> in real-time:

{{ image(
    src="/img/articles/understanding-rc/showcase.png", alt="Diffuse global illumination in flatland.",
    width="640px"
) }}

An awesome property of this method is that this is done <span class="highlight">fully-deterministically</span> and without temporal re-use!  
Furthermore there are already plenty of clever ways to get it's performance to acceptable levels for modern hardware.

*So without further ado, let's dive in!*

---

## Observations

Let us begin by making **two** key observations:

### Angular Observation

{{ video_loop(file = "/anim/articles/understanding-rc/angular-anim.mp4", alt = "Figure A: Circular light source with a radiance probe.", width = "540px") }}

*Figure A*, depicts a <span class="highlight">circular light source</span> on the left, with a radiance probe to the right of it.  
The radiance probe has an angular resolution which can be defined as the number of evenly spaced rays it evaluates. *(Shown in blue)*  
As the radiance probe moves <span class="highlight">further away</span> from the light source, we can see that it's <span class="highlight">angular resolution</span> becomes insufficient.

What we can observe here is that the minimum angular resolution we can get away with for a probe, depends on **two** factors:
1. $ D $ The <span class="highlight">distance</span> to the furthest light source.
2. $ w $ The <span class="highlight">size</span> of the smallest light source.

In the [paper](https://github.com/Raikiri/RadianceCascadesPaper) this restriction is formalized with this equation: $ \Delta_\omega < w/D $  
Which states our angular resolution $ \Delta_\omega $ should be smaller than the <span class="highlight">minimum angular resolution</span> $ w/D $.

### Spatial Observation

{{ video_loop(file = "/anim/articles/understanding-rc/spatial-anim.mp4", alt = "Figure B: Penumbra created by line light and line occluder.", width = "540px") }}

*Figure B*, shows that we can resolve a penumbra by <span class="highlight">interpolating</span> between 2 probes. *(Shown as blue dots)*  
The spacing of these probes increases the further away from the light source we get.

We can observe that the probe spacing is dependent on **two** factors:
1. $ D $ The <span class="highlight">distance</span> to the closest light source / occluder.
2. $ w $ The <span class="highlight">size</span> of the largest light source / occluder.

> Does that not sound familiar?

It's the **inverse** of the angular observation!  

### Penumbra Condition

While the angular resolution $ \Delta_\omega $ increases the spatial resolution $ \Delta_p $ decreases and vice versa.  
They are <span class="highlight">inversely proportional</span>.

In the [paper](https://github.com/Raikiri/RadianceCascadesPaper) this relationship is formalized as the <span class="highlight">penumbra condition</span> with this equation:

$
\begin{cases}
    \Delta_p <\sim D, \\\\
    \Delta_\omega <\sim 1/D
\end{cases}
$
> Because $ w $ would be the same for all probes in the scene, we omit it here.

---

## Data Structure

> Explain how we split probes into rings to exploit the angular and spatial observations.
> Explain the idea of cascades and the cascade hierarchy.

> Show how we can store the cascades as textures.

---

## Merging

> Explain the idea of merging for low-frequency diffuse lighting.

> Show code for the merge process.

---

## Results

> Show results.

---

## More Resources

There's quite a few resources already out there related to RC, I will list a few down below:
- Alexander Sannikov's [paper](https://github.com/Raikiri/RadianceCascadesPaper) on Radiance Cascades.
- Yaazarai's articles, [part 1](https://mini.gmshaders.com/p/radiance-cascades) & [part 2](https://mini.gmshaders.com/p/radiance-cascades2).
- SimonDev's video [https://youtu.be/3so7xdZHKxw](https://youtu.be/3so7xdZHKxw).
- Christopher M. J. Osborne's [paper](https://arxiv.org/abs/2408.14425) diving deeper into the bilinear fix.
- Jason's blog post [https://jason.today/rc](https://jason.today/rc).

## Equations

Final pass:

$ f_r(x) = \int_\Omega{L(x,\omega) * \cos(\theta_x) * d\omega} $

$ f_r(x) = \sum_i{L(x,\vec{\omega_i}) * (\vec{n} \cdot \vec{\omega_i})} $

Cook-Torrance Microfacet BRDF:

$$ f_r(v,l) = \frac{\rho_d}{\pi} + \frac{F(v, h) * D(h) * G(l, v)}{4 * (n \cdot l) * (n \cdot v)} $$
