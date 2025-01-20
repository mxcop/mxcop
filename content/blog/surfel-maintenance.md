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

As we can see in *Figure H*, the idea is that after the passes the grid buffer points to a <span class="highlight">range of elements</span> in the Surfel list buffer.  
Because a Surfel can be in multiple cells, the Surfel list can contain <span class="highlight">duplicate IDs</span>.

To achieve this, I used the following 3 passes:
1. <span class="highlight">Surfel counting</span> *(for each Surfel increment the `uint` inside each cell it overlaps)*
2. <span class="highlight">Prefix sum</span> *(perform a prefix sum over the entire grid buffer)*
3. <span class="highlight">Surfel insertion</span> *(for each Surfel decrement the `uint` inside each cell it overlaps and write the Surfel ID into the Surfel list)*

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

We can check the coverage for a pixel by checking the <span class="highlight">acceleration structure</span> at the position visible through that pixel:
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
    const float3 p = surfel_pos[surfel_ptr];
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
Because, now we're going to perform a <span class="highlight">atomic minimum</span> on that group shared `uint`.
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
