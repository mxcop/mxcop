+++
title = "Fundamentals of Radiance Cascades"
description = "For a few months now I've worked with Radiance Cascades, here's my understanding of the fundamentals provided by the paper."
authors = [ "Max &lt;mxcop&gt;" ]
date = 2024-08-31
draft = true

[[extra.tags]]
name = "graphics"
color = "emerald"

[[extra.tags]]
name = "radiance cascades"
color = "lime"
+++

## Introduction

In this article I'm going to share my understanding of the fudamentals of Radiance Cascades. *(abbreviated as RC)*  
At it's core Radiance Cascades is a method for efficiently representing a <span class="highlight">radiance field</span>.

> For the sake of simplicity I will keep the explenations 2D, however RC can be expanded into the third dimension aswell.  
> I will also assume the reader has a rudimentary understanding of ray tracing & the concept of radiance probes.

So, what can RC in 2D *(also referred to as Flatland)* achieve?  
My implementation is able to compute <span class="highlight">diffuse global illumination</span> in real-time:

<!-- {{ image(
    src="/img/articles/fundamental-rc/showcase.png", alt="Diffuse global illumination in flatland.",
    width="640px"
) }} -->

{{ video_loop(file = "/anim/articles/fundamental-rc/showcase.mp4", alt = "Diffuse global illumination in flatland.", width = "640px") }}

An awesome property of this method is that this is done <span class="highlight">fully-deterministically</span> and without temporal re-use!  
Furthermore there are already plenty of clever ways to get it's performance to acceptable levels for modern hardware.

*So without further ado, let's dive in!*

---

## Observations

Radiance Cascades is build on **two** key observations, which we will exploit.  
So first, let's observe these together, and have a <span class="highlight">quick recap</span> afterwards.

### Angular Observation

{{ video_loop(file = "/anim/articles/fundamental-rc/angular-anim.mp4", alt = "Figure A: Circular object with a radiance probe.", width = "540px") }}

*Figure A*, depicts a <span class="highlight">circular object</span> on the left, with a radiance probe to the right of it.  
The radiance probe has an angular resolution which can be defined as the angle between its evenly spaced rays. *(Shown in blue)*  
As the radiance probe moves <span class="highlight">further away</span> from the light source, we can see that it's <span class="highlight">angular resolution</span> becomes insufficient.

What we can observe here is that the angle between rays we can get away with for a probe, depends on **two** factors:
1. $ D $ The <span class="highlight">distance</span> to the furthest object.
2. $ w $ The <span class="highlight">size</span> of the smallest object.

In the [paper](https://github.com/Raikiri/RadianceCascadesPaper) this restriction is formalized with this equation: $ \Delta_\omega < w/D $  
Which states that the angle between our evenly spaced rays $ \Delta_\omega $ should be smaller than $ w/D $.

### Spatial Observation

{{ video_loop(file = "/anim/articles/fundamental-rc/spatial-anim.mp4", alt = "Figure B: Penumbra created by line light and line occluder.", width = "540px") }}

*Figure B*, shows that we can resolve a penumbra by <span class="highlight">interpolating</span> between only 2 probes. *(Shown as blue dots)*  
The spacing of these probes can be increased the further away we get from all objects in the scene.

{{ video_loop(file = "/anim/articles/fundamental-rc/penumbra-anim.mp4", alt = "Figure C: Moving the line occluder around.", width = "540px") }}

*Figure C*, shows that regardless of the distance between the light and the occluder, the penumbra still grows with distance.  
We can observe that the probe spacing is dependent on **two** factors:
1. $ D $ The <span class="highlight">distance</span> to the closest object.
2. $ w $ The <span class="highlight">size</span> of the smallest object.

> Does that not sound familiar?

The <span class="highlight">distance</span> is the **inverse** of the angular observation!  

### Penumbra Condition

While the maximum angle between rays ($ \Delta_\omega $) decreases, the maximum distance between probes ($ \Delta_p $) increases and vice versa.  
They are <span class="highlight">inversely proportional</span>.

In the [paper](https://github.com/Raikiri/RadianceCascadesPaper) this relationship is formalized as the <span class="highlight">penumbra condition</span> with this equation:

$
\begin{cases}
    \Delta_p <\sim D, \\\\
    \Delta_\omega <\sim 1/D
\end{cases}
$
> $ A <\sim B $ means that; $ A $ is less than the output of some function, which scales linearly with $ B $.

$ w $ is not included in the <span class="highlight">penumbra condition</span> because it is the same at all points in the scene.  
We can also observe that the required angular & spatial resolution both increase when $ w $ decreases.  
Because we need higher resolution for both in order to resolve the smallest object in the scene.

### Recap

Ok, these <span class="highlight">observations</span> took me some time to *wrap my head around* but they're key to understanding RC.  
*So let's have a quick recap.*  

What we've <span class="highlight">observed</span> is that the **further** we are from all objects in the scene:
1. The **less** spatial resolution we need. *(e.g. the larger spacing can be between probes)*
2. The **more** angular resolution we need. *(e.g. the more evenly spaced rays we need per probe)*

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
