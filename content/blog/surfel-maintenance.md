+++
title = "Surfel Maintenance for Global Illumination"
description = "A comprehensive explanation of mass Surfel probe maintenance."
authors = [ "Max &lt;mxcop&gt;" ]
date = 2025-01-16

[[extra.tags]]
name = "graphics"
color = "emerald"

[[extra.tags]]
name = "surfels"
color = "amber"

[extra]
hidden = false
+++

<p></p>

{{ image(
    src="/img/blog/surfel-maintenance/surfels.png", width="640px"
) }}

## Introduction

Since you've found this blog post, it's likely you already know what Surfels are.  
Regardless, I will start with a brief explanation of what they are, and what we can use them for.

{{ image(
    src="/img/blog/surfel-maintenance/surfel-parameters.png", alt="Figure A: The parameters that make up a Surfel.", width="640px"
) }}

The name Surfel comes from combining the words <span class="highlight">Surface & Element</span>.  
Surfels are described using 3 parameters:
1. <b style="color: #4dabf7">Position</b> *(Position on a surface)*
2. <b style="color: #40c057">Radius</b> *(How much area the Surfel represents on the surface)*
3. <b style="color: #ffc034">Normal</b> *(Normal of the surface)*

In *Figure A*, we can see a visual of these 3 parameters.

You may be wondering now *what are these Surfels useful for?*  
At it's core Surfels are a dynamic <span class="highlight">probe placement strategy</span>, a way to distribute probes on scene geometry.  
So, it is not limited to one use case only, however I personally used it for capturing <span class="highlight">Global Illumination</span>. 

{{ image_2x1(
    src1="/img/blog/surfel-maintenance/surfel-radiance-cascades.png", alt1="Figure B: Surfel Radiance Cascades. (by me)", width1="550px",
    src2="/img/blog/surfel-maintenance/ea-gibs.png", alt2="Figure C: EA SEED's GIBS.", width2="550px"
) }}

I also highly recommend you to check out EA SEED's [GIBS](https://www.ea.com/seed/news/siggraph21-global-illumination-surfels) *(Global Illumination based on Surfels)*  
Their talk at <span class="highlight">SIGGRAPH 21</span> has been my primary source for information on Surfels.

Now that we know what a Surfel is made out of, and what we can use it for.  
Let's dive into how I <span class="highlight">dynamically managed</span> Surfels for my Global Illumination solution specifically.

> There will be many small/big differences between my Surfel maintenance and that of GIBS.

---

## The Pipeline

{{ image(
    src="/img/blog/surfel-maintenance/surfel-pipeline.png", alt="Figure D: Overview of the Surfel Pipeline.", width="800px"
) }}

Let's start with a <span class="highlight">high level overview</span> of the maintenance pipeline for a single frame.  
As we can see in *Figure D*, the Surfel maintenance consists of 4 stages:
1. <b style="color: #40c057">Spawn</b> — *Uses the GBuffers to decide where on screen to spawn new Surfels.*
2. <b style="color: #ffc034">Transform</b> — *Updates the Surfel world-space positions based on the object they're attached to.*
3. <b style="color: #e599f7">Accelerate</b> — *Inserts Surfels into a spatial acceleration structure to accelerate lookups.*
4. <b style="color: #4dabf7">Recycle</b> — *Decides whether Surfels are still relevant.*

I marked the <b style="color: #ffc034">Transform</b> pass with a <b style="color: #ffc034">*</b> because it is technically optional.  
Due to time constraints, I didn't get to implementing a <b style="color: #ffc034">Transform</b> pass myself.  
However, I will still try to explain how one would implement it.

{{ image(
    src="/img/blog/surfel-maintenance/surfel-buffers.png", alt="Figure E: Overview of the Surfel Buffers.", width="640px"
) }}

The pipeline also needs a place to store all the <span class="highlight">Surfel data</span> of course.  
We allocate these buffers upfront, with a <span class="highlight">fixed maximum</span> number of live Surfels in the scene at once.  
I will go into more detail about the buffers in the sections below.

Now that we have an idea of how the pipeline fits together, let's dive into <span class="highlight">the details</span> of each pass.

---

### Surfel Acceleration

> Explain what & why we use a spatial acceleration structure.

> Talk about different kinds of spatial acceleration structures. (Uniform Grid, Trapezoidal Grid, Hash Grid)

---

### Surfel Spawning

> Explain the process of the spawning pass and how it uses the spatial acceleration structure.

> This is also where I can detail the Surfel Stack & the actual Surfel data buffers.

> Talk about optimization for this pass (1:4 pixels each frame)

---

### Surfel Recycling

> Talk about the idea of recycling Surfels that are no longer relevant.

> Explain different info which we can use for deciding when to recycle a Surfel.

> Explain how we push Surfels back onto the Surfel Stack to recycle them.

---

### Surfel Transformation

> Explain the issue with moving objects (Surfels ending up inside geometry or in the air)

> Explain the idea of attaching Surfels to object transforms using a global transform buffer.

> Unfortunately I didn't get to implementing this yet, should note that!

---

## Performance

> Give some performance metrics on my AMD Radeon 890M iGPU.

> Maybe note some ideas for increasing stability & performance?

---

## Conclusion

> ...
