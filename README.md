Head-Related Transfer Function (HRTF) audio signal processor.

# Overview

HRTF stands for [Head-Related Transfer Function](https://en.wikipedia.org/wiki/Head-related_transfer_function)
and can work only with spatial sounds. For each of such sound source after it was processed by HRTF you can
definitely tell from which location sound came from. In other words HRTF improves perception of sound to
the level of real life.

# HRIR Spheres

This crate uses Head-Related Impulse Response (HRIR) spheres to create HRTF spheres. An HRIR sphere is a set of
points in 3D space which are connected into a mesh forming triangulated sphere. Each point contains the impulse
responses for the left and right ears. HRIR spheres can be found [here](https://github.com/mrDIMAS/hrir_sphere_builder/tree/master/hrtf_base/IRCAM).
HRIR spheres from the base are recorded in 44100 Hz sample rate, this crate performs **automatic** resampling to your
sample rate.

# Algorithm

Convolution is done directly in the time domain, and the sound's direction is interpolated at the audio rate
(once per sample) instead of once per block. Each input sample `x[m]` is filtered with the impulse response of
the direction the source had when that sample was emitted:

```text
y[n] = Σ_m x[m] · h_dir(m)[n - m]
```

`h_dir` is the barycentric blend of the three HRIRs at the corners of the sphere triangle that the direction
points into. Barycentric interpolation over a mesh is continuous across the triangles' shared edges, so the
effective filter changes smoothly as a source moves. There are no block-boundary filter switches, which is what
makes fast-moving sounds "buzz" in block-based renderers.

Each input sample is *scattered* forward into a short accumulator holding the next `len` output samples (where
`len` is the HRIR length). All memory accesses are contiguous, the triangle can change on any sample with no
extra bookkeeping, and there is no FFT, no block latency and no allocation while processing. The cost is
constant: three multiply-adds per tap per ear for every non-silent input sample. For the bundled-style IRCAM
sphere (512 taps at 44.1 kHz) that is under 1% of one CPU core per source.

# Usage

```rust,no_run
use hrtf::{HrirSphere, HrtfProcessor, HrtfSphere, Vec3};

let hrir_sphere = HrirSphere::from_file("your_file", 44100).unwrap();

// The sphere can be shared between any number of sources; each source has its own processor.
let sphere = HrtfSphere::new(&hrir_sphere);
let mut processor = HrtfProcessor::new(sphere);

let source = vec![0.0; 128]; // Fill with something useful.
let mut output = vec![(0.0, 0.0); 128];

// The direction (from the listener to the source, in the listener's space) and the gain
// are interpolated per sample from their values at the end of the previous call.
processor.process(&source, &mut output, Vec3::new(0.0, 0.0, 1.0), 1.0);
```

Use `HrtfProcessor::process_per_sample` to supply a direction and gain for every sample yourself.
