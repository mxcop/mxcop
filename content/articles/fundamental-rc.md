+++
title = "Fundamentals of Radiance Cascades"
description = "For a few months now I've worked with Radiance Cascades, here's my understanding of the fundamentals provided by the paper."
authors = [ "Max &lt;mxcop&gt;" ]
date = 2024-08-31

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
Essentially allowing us to represent the <span class="highlight">incoming light</span> from the scene at any point in the scene.

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

Radiance Cascades is build on **two** key observations.  
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

We can observe that the probe spacing is dependent on **two** factors:
1. $ D $ The <span class="highlight">distance</span> to the closest object.
2. $ w $ The <span class="highlight">size</span> of the smallest object.

> Does that not sound familiar?

The <span class="highlight">distance</span> is the **inverse** of the angular observation!  

{{ video_loop(file = "/anim/articles/fundamental-rc/penumbra-anim.mp4", alt = "Figure C: Moving the line occluder around.", width = "540px") }}

*Figure C*, shows that regardless of the distance between the light and the occluder, the penumbra still <span class="highlight">grows with distance</span>.  
However, the sharpness of the penumbra changes, RC is notoriously bad at representing *very* sharp shadows. 

I want *Figure C* to <span class="highlight">clearify</span> that we're interested in the nearest or furthest object, **not light source**.  
*At the end of the day, a wall is just a light source that emits no light, and a light source is just a wall that emits light.*

### Penumbra Condition / Theorem

While the required angle between rays ($ \Delta_\omega $) decreases, the required distance between probes ($ \Delta_p $) increases and vice versa.  
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
*So let's quickly reiterate our observations.*  

What we've <span class="highlight">observed</span> is that the **further** we are from the closest object in the scene:
1. The **less** spatial resolution we need. *(e.g. the <span class="highlight">larger spacing</span> can be between probes)*
2. The **more** angular resolution we need. *(e.g. the <span class="highlight">more rays</span> we need per probe)*

---

## Exploiting Observations

Now that we've made the observations and defined the penumbra theorem, let's look at how we can <span class="highlight">exploit</span> these observations.

### Angular

We've got a **problem**: normal probes we're all used to, can hit objects at <span class="highlight">virtually any distance</span>.  
In order to exploit the <span class="highlight">penumbra theorem</span> we need some way to *narrow* this possible <span class="highlight">distance window</span>.

{{ video_loop(file = "/anim/articles/fundamental-rc/splitting-anim.mp4", alt = "Figure D: Probe being split into &ldquo;rings&rdquo;.", width = "360px") }}

*Figure D*, shows one way of narrowing this window, we can split our probes into rings.  
By doing this we **not only** know that each ring will hit within a narrow distance window.  
We can also <span class="highlight">vary</span> the <span class="highlight">angular resolution</span> between rings!

> These new rays with a limited range, are referred to as **intervals**.

This is exactly what we're looking for to <span class="highlight">exploit</span> the <span class="highlight">angular part</span> of the penumbra theorem.  
We can increase the interval count *(aka, decrease the angle between rays)* with each consecutive ring that hits objects further away.

{{ image(
    src="/img/articles/understanding-rc/inc-angular-split.png", alt="Figure E: Increasing angular resolution for more distant &ldquo;rings&rdquo;.",
    width="360px"
) }}

In order to still <span class="highlight">capture</span> our entire scene, we will have many of these *rings*.  

In the example in *Figure E*, we increase the interval count by **2x** with every consecutive ring.  
Which let's us increase the <span class="highlight">distance window</span> of each ring *(the length of its intervals)* by that same factor.  
This ensures the <span class="highlight">gap</span> between intervals remains *approximately* equal between rings.

### Spatial

So far, with the angular observation we haven't really achieved any <span class="highlight">reduction</span> in ray count.  
We're still casting a <span class="highlight">very large number</span> of rays for each probe using this method, *good thing that's about to change.*

This is when we <span class="highlight">drop the idea</span> that these rings together make up a **single** probe.  
Instead let's view each consecutive ring as its own probe, which *can be moved*.

> From now on when we refer to **probes**, we are referring to **rings**.

{{ image(
    src="/img/articles/understanding-rc/cascade-crown.png", alt="Figure F: 4 blue probes for 1 green probe.",
    width="360px"
) }}

*Figure F*, shows one way we can use this new <span class="highlight">perspective</span> on the probes.  
We saw during the spatial observation that when objects are <span class="highlight">further away</span>, we can have <span class="highlight">larger spacing</span> between probes.

So, when our <span class="highlight">distance window</span> gets further and further away, we may increase the <span class="highlight">spacing</span> between those probes.

---

## Cascades

Now that we understand how we can exploit the **two** key observations.  
Let's put the **two** together and finally define what exactly a <span class="highlight">cascade</span> is!

{{ image_2x1(
    src1="/img/articles/understanding-rc/cascade0.png", alt1="Figure G1: Cascade 0, with 4x4 probes.",
    src2="/img/articles/understanding-rc/cascade1.png", alt2="Figure G2: Cascade 1, with 2x2 probes.",
    width1="360px", width2="360px"
) }}

A cascade is basically a <span class="highlight">grid of probes</span>, in which all probes have **equal** properties.  
*(e.g. interval count, interval length, probe spacing)*

The reason we call them cascades is because they <span class="highlight">cascade outwards</span> with increasing interval count and length.  
*Or at least, that's how I like to think about it.*

### Cascade Hierarchy

A cascade <span class="highlight">on its own</span> isn't super useful, only capturing a small part of the scene.  
Many cascades together is what we're really after, we want to **combine** them into a hierarchy.  
For example in *Figure G1 & G2* we can see two cascades that could make up a <span class="highlight">cascade hierarchy</span>.

Most of the time, for <span class="highlight">simplicity</span> sake we will decrease probe count between cascades by **2x** along each axis.  
Like we've seen also in *Figure G1 & G2*, we will find out why this is convenient *later* on in this article. 

If we're following the <span class="highlight">penumbra condition</span>, the spatial and angular resolution should be **inversely proportional**.  
So if we increase probe spacing by **2x** we need to decrease the angle between intervals by **2x** as well. 

> However, there's also many implementation which decrease the angle between intervals by **4x** instead.  
> It is more costly, but it may produce higher quality results in some cases.

### Cascade Memory

{{ video_loop(file = "/anim/articles/fundamental-rc/probe-memory-anim.mp4", alt = "Figure H: 4x4 probe in texture memory.", width = "360px") }}

The most common way we <span class="highlight">store probes</span> in memory is using a **2D texture**.  
In *Figure H*, we can see one such probe, it has *16* intervals making it *4x4* texels in memory.  
Each <span class="highlight">texel</span> representing a single <span class="highlight">direction</span>, indicated by the white arrow in the center.

```glsl
const int dir_count = 16; /* 4x4 */
const int dir_index = /* ... */;

/* Compute interval direction from direction index */
float angle = TAU * ((float(dir_index) + 0.5) / float(dir_count));
vec2 dir    = vec2(cos(angle), sin(angle));
```

> The *code snippet* above shows how we can derive an interval direction from its index within its probe.

{{ image(
    src="/img/articles/understanding-rc/cascade-memory.png", alt="Figure I: Cascade in texture memory.",
    width="360px"
) }}

Now, of course we're not going to store <span class="highlight">each probe</span> in its own texture.  
Instead, let's store <span class="highlight">each cascade</span> in a texture, packing the probes together as shown in *Figure I*.

This is where we see why decreasing the probe count by **2x** on <span class="highlight">each axis</span> is nice.  
It works out *really well* when using this kind of packing.

If we decrease the angle between intervals by **2x** each cascade, each subsequent cascade will have <span class="highlight">half the intervals</span> of the previous.  
Because the probe count is decreasing by **2x** along 2 axes, making it decrease by **4x**, while the interval count only increases by **2x**.  
> Meaning our total interval count will aproach **2x** the interval count of the first cascade as we add more cascades.

If instead we decrease the angle between intervals by **4x** each cascade, each cascade will have <span class="highlight">equal the intervals</span>. 
> Meaning our total interval count will grow linearly with cascade count.

I <span class="highlight">recommend</span> using the **4x** branching method where interval count remains equal, it is <span class="highlight">simpler</span> to work with in practice.

### Cascade Gathering

To gather the <span class="highlight">radiance</span> for each cascade we simply loop over each texel in its memory texture.  
For each of those texels we *calculate the direction* and cast our interval into the scene.

First, let's find out what our coordinate is within the probe we're apart of:
```glsl
/* Get the local texel coordinate in the local probe */
const ivec2 dir_coord = texel_coord % probe_size;
```

Second, we can convert this coordinate to a <span class="highlight">direction index</span>:
```glsl
/* Convert our local texel coordinate to a direction index */
const int dir_index = dir_coord.x + dir_coord.y * probe_size.x;
```

Third, using that direction index we can obtain the direction vector: *(like I showed earlier)*
```glsl
const int dir_count = probe_size.x * probe_size.y;

/* Compute interval direction from direction index */
float angle = TAU * ((float(dir_index) + 0.5) / float(dir_count));
vec2 dir    = vec2(cos(angle), sin(angle));
```

Now we have to <span class="highlight">cast the interval</span>, let's not forget intervals have a start and end time:
```glsl
vec2 interval_start = probe_pos + dir * start_time;
vec2 interval_end   = probe_pos + dir * end_time;
vec3 radiance       = cast_interval(interval_start, interval_end);
```

It's important to note, the `cast_interval` function can use whatever <span class="highlight">ray casting method</span> you want.  
As long as it returns the radiance information from the scene from the start to the end position.

The <span class="highlight">start & end time</span> of our intervals depends on which cascade we're evaluating, and what branching is used.  
For **4x** branching (the branching I recommend) we can use this code to find the start & end times:
```glsl
/* Get the scale factor for an interval in a given cascade */
float interval_scale(int cascade_index) {
    if (cascade_index <= 0) return 0.0;

    /* Scale interval by 4x each cascade */
    return float(1 << (2 * cascade_index));
}

/* Get the start & end time of an interval for a given cascade */
vec2 interval_range(int cascade_index, float base_length) {
    return base_length * vec2(interval_scale(i), interval_scale(i + 1));
}
```

> The `base_length` above is the length you want intervals in cascade0 to have.

---

## Merging

We now should have our cascades stored in textures filled with radiance information. *Awesome!*  
The next step is to actually <span class="highlight">extract the data</span> we want from this data structure *(the cascade hierarchy)*.

We're going to extract specifically the <span class="highlight">diffuse irradiance</span> of the scene.  
This basically means *adding together* the radiance coming in at a specific point from <span class="highlight">all directions</span>.  

To do this efficiently, we traverse the cascades in a downwards fashion, starting at the highest cascade *(with the least probes)*.

> Explain the idea of merging for low-frequency diffuse lighting.

### Interval Merging

To resolve the <span class="highlight">diffuse radiance</span> in the scene, we can recursively merge intervals to capture the radiance from a <span class="highlight">cone</span>.  
The question is: *"How do we merge intervals, and how do we merge them between cascades?"*

> [Figure G: graphic showing intervals making up a cone]

*Figure G, shows a <span class="highlight">cone</span> made out of intervals from different <span class="highlight">cascades</span>.  
In this case, the number of <span class="highlight">intervals</span> grows by **x4** with each subsequent cascade.  
Which also means we will <span class="highlight">merge 4</span> intervals from cascade N+1 <span class="highlight">into 1</span> interval of cascade N.

The way we do this is by first <span class="highlight">averaging</span> the 4, N+1 intervals together.  
Then we can use the following function to merge the averaged interval with the N interval.

> Show code for the merge process.

---

## Results

> Show results.

---

## Amazing Resources

There's quite a few resources already out there related to RC. *(which helped me)*  
I will list a few of them here, so you can get explanations from <span class="highlight">different perspectives</span>:
- Alexander Sannikov's [paper](https://github.com/Raikiri/RadianceCascadesPaper) on Radiance Cascades.
- XorDev & Yaazarai's articles, [part 1](https://mini.gmshaders.com/p/radiance-cascades) & [part 2](https://mini.gmshaders.com/p/radiance-cascades2).
- SimonDev's video [https://youtu.be/3so7xdZHKxw](https://youtu.be/3so7xdZHKxw).
- Christopher M. J. Osborne's [paper](https://arxiv.org/abs/2408.14425) diving deeper into the bilinear fix.
- Jason's blog post [https://jason.today/rc](https://jason.today/rc).

## Equations

Final pass:

$ f_r(x) = \int_\Omega{L(x,\omega) * \cos(\theta_x) * d\omega} $

$ f_r(x) = \sum_i{L(x,\vec{\omega_i}) * (\vec{n} \cdot \vec{\omega_i})} $

Cook-Torrance Microfacet BRDF:

$$ f_r(v,l) = \frac{\rho_d}{\pi} + \frac{F(v, h) * D(h) * G(l, v)}{4 * (n \cdot l) * (n \cdot v)} $$
