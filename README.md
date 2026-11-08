Head-Related Transfer Function (HRTF) audio signal processor.

# Overview

HRTF stands for [Head-Related Transfer Function](https://en.wikipedia.org/wiki/Head-related_transfer_function)
and can work only with spatial sounds. For each of such sound source after it was processed by HRTF you can
definitely tell from which location sound came from. In other words HRTF improves perception of sound to
the level of real life.

# HRIR Spheres

This crate uses Head-Related Impulse Response (HRIR) spheres to create HRTF spheres. HRTF sphere is a set of
points in 3D space which are connected into a mesh forming triangulated sphere. Each point contains spectrum
for left and right ears which will be used to modify samples from each spatial sound source to create binaural
sound. HRIR spheres can be found [here](https://github.com/mrDIMAS/hrir_sphere_builder/tree/master/hrtf_base/IRCAM).
HRIR spheres from the base are recorded in 44100 Hz sample rate, this crate performs **automatic** resampling to your
sample rate.

# Performance

HRTF is **heavy**, this is essential because HRTF requires some heavy math (fast Fourier transform, convolution,
etc.) and lots of memory copying.

# Algorithm

This crate uses overlap-save convolution to perform operations in frequency domain. Check
[this link](https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method) for more info. Convolution
is performed twice per an input signal block with different gain and sampling vectors and then
cross-fade is applied to remove distortions (clicks, pops, buzzing, etc.).  