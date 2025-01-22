+++
title = "Surfel Maintenance for Global Illumination"
description = "A comprehensive explanation of my implementation of Surfel probe maintenance."
authors = [ "Max &lt;mxcop&gt;" ]
date = 2025-01-16

[[extra.tags]]
name = "graphics"
color = "emerald"

[[extra.tags]]
name = "surfels"
color = "amber"

[extra]
hidden = true
splash = "img/blog/surfel-maintenance/surfel-splash.png"
+++

## Introduction

Since you've found this blog post, it's likely you already know what <span class="highlight">Surfels</span> are.  
Regardless, I will start with a brief explanation of what they are, and what we can use them for.

Surfels can be used to <span class="highlight">divide up</span> the surface of <span class="highlight">geometry</span> into discrete patches.  
This is very useful for <span class="highlight">caching lighting</span> information for example in the case of Global Illumination.

We can see this division of the surface of geometry <span class="highlight">below</span> here, where each Surfel patch is given a random color.

{{ image_2x1(
    src1="/img/blog/surfel-maintenance/surfels.png", alt1="Surfels discretizing scene geometry into patches.", width1="610px",
    src2="/img/blog/surfel-maintenance/surfel-parameters-2.png", alt2="Figure A: The parameters that make up a Surfel.", width2="550px"
) }}

The name Surfel comes from combining the words <span class="highlight">Surface & Element</span>.  
Surfels are commonly described using 3 parameters:
1. <b style="color: #4dabf7">Position</b> *(Position on a surface)*
2. <b style="color: #40c057">Radius</b> *(How much area the Surfel represents on the surface)*
3. <b style="color: #ffc034">Normal</b> *(Normal of the surface)*

In *Figure A*, we can see a visual of these 3 parameters.

You may be wondering now *what are these Surfels useful for?*  
At it's core Surfels are a method for <span class="highlight">dividing up</span> scene geometry into discrete patches, as I mentioned above.  
So, it is not limited to one use case only, however it is commonly used for caching lighting information for <span class="highlight">Global Illumination</span>. 

{{ image_2x1(
    src1="/img/blog/surfel-maintenance/surfel-radiance-cascades.png", alt1="Figure B: Surfel Radiance Cascades. (by me)", width1="550px",
    src2="/img/blog/surfel-maintenance/ea-gibs.png", alt2="Figure C: EA SEED's GIBS.", width2="550px"
) }}

I highly recommend you to check out EA SEED's [GIBS](https://www.ea.com/seed/news/siggraph21-global-illumination-surfels) *(Global Illumination based on Surfels)*  
Their talk at <span class="highlight">SIGGRAPH 21</span> has been my primary source for information on Surfels.

A big advantage of Surfels as a light information cache, is that Surfels persist between frames.  
So, we can simply <span class="highlight">accumulate</span> information within them between frames, without a need for reprojection.

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

In many cases we want to <span class="highlight">find Surfels</span> around a given position.  
For example, when spawning Surfels we want to know if there are already Surfels nearby.  
Or when recycling we want to know if there are too many Surfels nearby.

#### Spatial Acceleration Structures

In order to do this quickly, we need a spatial acceleration structure.  
I'm going to cover a few options which I personally investigated:
1. Uniform Grid *(The most basic structure)*
2. Trapezoidal Grid *(Used by GIBS)*
3. Multi-level Hash Grid *(What I ended up using)*

I initially began by using a <span class="highlight">3D uniform grid</span>, for me that was a good starting point.  
However, it scales very poorly when you want larger scenes, and when your Surfels are relatively small.

{{ image(
    src="/img/blog/surfel-maintenance/trapezoidal-grid.png", alt="Figure F: Trapezoidal grid structure used by GIBS.", width="480px"
) }}

<span class="highlight">GIBS</span> combines a small uniform grid centered on the camera with <span class="highlight">trapezoidal grids</span> extending outwards.  
This gets you very fast lookups, as the trapezoidal grids are simply uniform grids with a non-linear transform.  
However, I instead decided on using a <span class="highlight">multi-level hash grid</span>, so I never implemented this scheme.

{{ image(
    src="/img/blog/surfel-maintenance/multi-level-hash-grid.png", alt="Figure G: Heatmap view of the multi-level hash grid.", width="640px"
) }}

The multi-level hash grid is relatively simple, it is a normal hash grid storing a `uint` for each cell.  
Cell hashes are created by combining a <span class="highlight">grid position</span> & <span class="highlight">grid level</span>, I used 9 bits for each axis and 5 bits for the level.

```glsl
/* Hash a Surfel grid location. (XYZ9L5) */
uint surfel_hash_function(const uvec3 loc, const uint level) {
    return xxhash32(((loc.x % 512u) << (9u * 0u))
                  | ((loc.y % 512u) << (9u * 1u))
                  | ((loc.z % 512u) << (9u * 2u))
                  | ((level &  31u) << (9u * 3u)));
}
```

> The grid level is determined using a logarithm of the distance squared to the camera.

> For each live Surfel the <span class="highlight">acceleration pass</span> will insert the Surfel into all cells it overlaps.  
> It's also very important here to be aware of the grid <span class="highlight">level transitions</span>, on which you need to insert the Surfel into both levels.

#### Filling the Structure

I mentioned how each grid cell only stores a `uint`, this is key to making the structure memory efficient.  
Surfels are represented by a <span class="highlight">unique ID</span>, however we do not want to store those directly in the grid cells.  
Instead we want each grid cell to store an index into a <span class="highlight">list of Surfel IDs</span>.

> This unique ID we use to refer a Surfel is also the index inside the buffers where it's data is located.

To achieve this, we will need <span class="highlight">multiple passes</span> for filling the acceleration structure.  
This <span class="highlight">applies to all</span> the acceleration structure variants I described above.

{{ image(
    src="/img/blog/surfel-maintenance/surfel-grid-list.png", alt="Figure H: Visualization of the grid & list structure.", width="640px"
) }}

As we can see in *Figure H*, the idea is that after the passes, the grid buffer points to a <span class="highlight">range of elements</span> in the Surfel list buffer.  
Because a Surfel can be in multiple cells, the Surfel list can contain <span class="highlight">duplicate IDs</span>.

To achieve this, I used the following 3 passes:
1. <span class="highlight">Surfel counting</span> *(for each Surfel, increment the `uint` inside each cell it overlaps)*
2. <span class="highlight">Prefix sum</span> *(perform a prefix sum over the entire grid buffer)*
3. <span class="highlight">Surfel insertion</span> *(for each Surfel, decrement the `uint` inside each cell it overlaps and write the Surfel ID into the Surfel list)*

> When looping over the Surfels, we always loop over the entire Surfel buffer.  
> If the *radius* of a Surfel is 0 we know the Surfel is not live, so we can return early.

```glsl
/* <===> Pass 1 <===> */
for (surfels) for (overlapping cells) {
    /* Increment the atomic counter in the hash grid cell */
    atomicAdd(surfel_grid[surfel_hash_function(...) % grid_capacity], 1);
}

/* <===> Pass 2 <===> */
/* Perform a inclusive prefix sum on the grid buffer */

/* <===> Pass 3 <===> */
for (surfels) for (overlapping cells) {
    /* Decrement the atomic counter in the hash grid cell */
    const uint offset = atomicAdd(surfel_grid[surfel_hash_function(...) % grid_capacity], -1) - 1u;
    surfel_list[offset] = surfel_ptr; /* <- Insert the Surfel ID into the Surfel list */
}
```

> It's very important here to be aware of the grid <span class="highlight">level transitions</span> while counting & inserting.  
> On the transitions you have to insert Surfels into both grid levels.

After this you will now be able to <span class="highlight">quickly</span> find nearby Surfels with no limit of the bounds of your scene.  
However, it is important to note, hash grids experience <span class="highlight">hash collisions</span>.  
So we have to <span class="highlight">be aware</span> that you might sometimes get Surfels which are actually very far away.

---

### Surfel Spawning

When spawning Surfels we spawn them from the <span class="highlight">GBuffer</span>, depth & normals.  
The <span class="highlight">tricky part</span> is avoiding spawning Surfels too close to each other.

The way we combat that is by splitting the screen into <span class="highlight">16x16</span> pixel tiles.  
Each tile will find the pixel within itself which has the least Surfel coverage.  
If that coverage is below a certain <span class="highlight">threshold</span>, we will spawn a new Surfel on that pixel.

> You may want to choose a different tile size, depending on the size of your Surfels.

#### Coverage Testing

We can check the coverage for a pixel by checking the <span class="highlight">acceleration structure</span> of the *previous frame* at the position of the pixel:
```glsl
/* Location in world-space visible through this pixel */
const vec3 pixel_pos = camera_pos + pixel_dir * pixel_depth;

/* Grab the start & end indices for this pixel's hash cell */
const uint hashkey = surfel_pos_hash(pixel_pos, camera_pos) % grid_capacity;
const uint start = surfel_grid[hashkey], end = surfel_grid[hashkey + 1u];

float coverage = 1e30;
for (uint i = start; i < end; ++i) {
    /* Grab the Surfel ID from the List */
    const uint surfel_ptr = surfel_list[i];

    /* Use the Surfel ID to fetch it's position & radius */
    const vec3  p = surfel_pos[surfel_ptr];
    const float r = surfel_radius[surfel_ptr];

    /* Find the highest coverage */
    coverage = max(coverage, point_coverage(pixel_pos, p, r));
}
```

> `surfel_pos_hash(...)` finds the grid position, level, and feeds those into `surfel_hash_function(...)`.

Now, in order to efficiently <span class="highlight">communicate</span> across the 16x16 tile of pixels on the GPU we can use <span class="highlight">group shared memory</span>.  
The idea is simple, we have a single `uint` as group shared memory, groups are 16x16 lanes in size.  
```glsl
coherent uint gs_candidate;
```
Each lane corresponds to a pixel in a 16x16 tile, and will combine its coverage & local index into a `uint`:
```glsl
/* Compound the pixel coverage & pixel index within its 16x16 tile into a uint */
const uint score = min(65535u, (uint)(coverage * 1000.0));
const uint candidate = ((score << 16u) & 0xffff0000) | (local_idx & 0x0000ffff);
```
The reason we add the local index is make each pixel have a unique `uint`.  
Because, now we're going to perform an <span class="highlight">atomic minimum</span> on that group shared `uint`.
```glsl
gs_candidate = 0xFFFFFFFF; /* Reset the groupshared candidate */
barrier();

/* Find the pixel with the lowest coverage in this group */
atomicMin(gs_candidate, candidate);
barrier();
```
Now, after this <span class="highlight">memory barrier</span> we can check if our pixel was the one with the least coverage:
```glsl
if (gs_candidate == candidate) {
    /* Spawn Surfel if the coverage is below the threshold... */
}
```

#### Spawning Surfels

We now know which pixels want to spawn a Surfel without causing much overlap.  
So, let's have a look at how exactly we spawn Surfels.

This is where the <span class="highlight">Surfel Stack</span> buffer comes into play.

{{ image(
    src="/img/blog/surfel-maintenance/surfel-spawning.png", alt="Figure I: Visualization of spawning a Surfel using the Stack.", width="960px"
) }}

First of all, as we can see in *Figure I*, the Surfel Stack has to start out filled with all unique Surfel IDs.  
The order in which they are placed doesn't matter, as long as they're <span class="highlight">all unique</span>.

We can see the spawning sequence in *Figure I*, we simply increment the Surfel Stack pointer.  
And use the unique ID in the stack as the <span class="highlight">Surfel ID</span> which we can write data into:
```glsl
/* Pop the Surfel Stack (atomic) */
const uint stack_ptr = atomicAdd(surfel_stack_pointer, 1);
const uint surfel_ptr = surfel_stack[stack_ptr];

/* Write our new Surfel data into the Surfel buffers */
surfel_pos[surfel_ptr] = ...;
surfel_radius[surfel_ptr] = ...;
```

> For simplicity I left out the bounds check here, but you should make sure the stack isn't full!

#### Spawning Optimization

Because I am managing multiple sets of Surfels, spawning can get pretty expensive.  
An optimization I came up with is to only <span class="highlight">check 1/4</span> of the pixels for coverage each frame.  
Cycling through a 2x2 tile of pixels every 4 frames.

On my <span class="highlight">AMD Radeon 890M iGPU</span> this resulted in the total time spend spawning going from:  
`~6ms` -> `~1ms`, which is a rather huge improvement, without losing any significant quality.

---

### Surfel Recycling

If all we do is spawn new Surfels, we will quickly run out of our <span class="highlight">Surfel budget</span>.  
Lucky for us, most Surfels can be recycled after a while, when they are no longer relevant.

#### Recycling Heuristic

The way we decide whether to recycle a Surfel is based on a <span class="highlight">heuristic</span>.  
This heuristic is usually a combination of a few different parameters:
1. Time since used *(Last time when the Surfel was sampled)*
2. Surfel coverage *(How many Surfels are nearby)*
3. Live Surfel count *(How many Surfels are currently in use)*

To find the Surfel <span class="highlight">coverage</span> we can use the same method we used during spawning to find the coverage.  
The time since used can be stored on the Surfel, to always be incremented during the recycling pass.  
We can <span class="highlight">reset that time</span> every time we sample the Surfel's lighting information.

Once we have the heuristic, we can use it as a <span class="highlight">random chance</span> for recycling.  
Or we can use it as a threshold for deterministic recycling.

#### Despawning Surfels

If we decide to recycle a Surfel we'll have to <span class="highlight">push</span> it back onto the <span class="highlight">Surfel Stack</span>.  

{{ image(
    src="/img/blog/surfel-maintenance/surfel-recycling.png", alt="Figure J: Visualization of recycling a Surfel using the Stack.", width="640px"
) }}

All Surfel IDs to the right of the Surfel Stack pointer will always be <span class="highlight">unique IDs</span> to unused Surfel Data.  
To maintain that while recycling, we can decrement the Surfel Stack pointer, and write out Surfel ID into the Stack slot that just opened up.  
We can see this <span class="highlight">sequence</span> in *Figure J* where we recycle Surfel ID `8` by pushing it back onto the Surfel Stack.
```glsl
/* First set the recycled Surfel's radius to 0.0, marking it as unused */
surfel_radius[surfel_ptr] = 0.0;

/* Then decrement the stack pointer & write the Surfel ID into the open slot */
const uint slot = atomicAdd(surfel_stack_pointer, -1) - 1u;
surfel_stack[slot] = surfel_ptr;
```

As mentioned ealier, the <span class="highlight">radius</span> is also used when looping over all Surfels to check if the Surfel is <span class="highlight">live</span>.  
This also allows us to early out during the recycling pass for example:
```glsl
/* Compute kernel executed for each Surfel in Surfel Data */
void main(uint thread_id) {
    const uint surfel_ptr = thread_id;

    const float radius = surfel_radius[surfel_ptr];
    if (radius == 0.0) return; /* Early out if this Surfel is not live */

    /* ... */
}
```

---

### Surfel Transformation

During the spawning sequence, when we place Surfels we're currently doing so in <span class="highlight">world-space</span>.  
Which means that Surfels won't move with the geometry that they're supposed to be attached to.

#### Transform Buffer

Ideally we want to instead attach Surfels to the model matrix *(transform)* of objects in the scene paired with a <span class="highlight">local</span> position.  
This is what <span class="highlight">GIBS</span> does, they have a buffer filled with the <span class="highlight">transforms</span> of all objects in the scene.  
When spawning a new Surfel, they assign that Surfel the transform ID of the object they are spawning on.  
Which points to the model matrix of that object in the global <span class="highlight">transform buffer</span>.

{{ image(
    src="/img/blog/surfel-maintenance/gibs-transform-gbuffer.png", alt="Figure K: Transform ID Gbuffer from GIBS.", width="640px"
) }}

To know what transform a Surfel should be attached to when spawning, GIBS has a transform ID <span class="highlight">Gbuffer</span>.  
We can see a <span class="highlight">debug view</span> of this in *Figure K*, each color represents a different transform ID.

> Unfortunately I cannot go into more detail on this part of the pipeline, because I personally skipped it due to time constraints.

---

## Performance

We're almost at the end of the blog post now, so let's look at some <span class="highlight">performance measurements</span>.  
Keep in mind that these timings are for <span class="highlight">6</span> individual Cascades of Surfels, so you can expect timings normally to be better.

> Each Surfel Cascade has `1/4` the Surfel count of the previous one, with the first having `262.144` Surfels at most.

{{ image(
    src="/img/blog/surfel-maintenance/surfel-performance-890m.png", alt="Figure L: Captured on AMD 890M iGPU. (6 Surfel Cascades)", width="960px"
) }}

First, here's the performance on my <span class="highlight">AMD Radeon 890M</span> integrated GPU, we can see that the hash insertion *(Surfel Insertion)* is the most expensive part.  
This is because it requires us to wait for an <span class="highlight">atomic operation</span> to complete, because we need to know which Surfel List index to insert the Surfel into.  
Besides that, I'd argue the performance is quite good, especially because this is maintaining 6 sets of Surfels.

{{ image(
    src="/img/blog/surfel-maintenance/surfel-performance-4070.png", alt="Figure M: Captured on RTX 4070 Mobile. (6 Surfel Cascades)", width="960px"
) }}

Now let's look at the performance on my <span class="highlight">NVIDIA RTX 4070</span> Mobile GPU.  
We can see here that the hash insertion has gone down quite a bit, and now the <span class="highlight">recycling pass</span> is actually the most expensive pass.  
However, again I'd argue the <span class="highlight">overhead</span> of maintenance is <span class="highlight">relatively small</span> here.

---

## Conclusion

To round this off, we've looked at how we can maintain a large number of Surfels in an efficient manor.  
We took a <span class="highlight">high level</span> overview of the entire pipeline, and then went into the <span class="highlight">details</span> of each maintenance pass.  
And in the end we briefly looked at performance on 2 modern GPUs.

It took me quite some research and <span class="highlight">trail & error</span> to figure out some of the details.  
So, I hope this blog post sheds some more light on the details of how to maintain Surfels.  

### Resources

Here's a few resources which helped me on my <span class="highlight">Surfel journey</span> :)
- Hybrid Rendering for Real-Time Ray Tracing [Ray Tracing Gems 2019](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/2019-ray-tracing-gems-chapter-25-barre-brisebois-et-al.pdf).
- SIGGRAPH 2021, GIBS [https://youtu.be/h1ocYFrtsM4](https://youtu.be/h1ocYFrtsM4).
- Surfel GI implementation in `kajiya` by Tomasz [https://github.com/h3r2tic/kajiya](https://github.com/h3r2tic/kajiya)
