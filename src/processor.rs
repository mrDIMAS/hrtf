//! Time-domain HRTF convolution with per-sample direction interpolation.
//!
//! See the crate docs for a description of the algorithm.

use crate::{lerpf, HrirSphere, Vec3};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Input samples are processed in chunks of at most this size, which bounds the accumulator's size.
const CHUNK: usize = 128;

/// Marker for a triangle edge with no neighboring triangle.
const NO_NEIGHBOR: u32 = u32::MAX;

/// Barycentric coordinates may be this much below zero (relative to their sum) before
/// a direction is considered to be outside of a triangle.
const INSIDE_EPSILON: f32 = 1e-6;

/// Samples quieter than this (about -400 dB) are skipped, which saves work during silence and
/// keeps denormals and NaNs out of the accumulator.
const SILENCE: f32 = 1e-20;

/// The maximum number of neighboring triangles to step through before searching the whole sphere.
const MAX_WALK_STEPS: usize = 16;

/// Fused multiply-add where the hardware supports it. Falls back to separate operations
/// otherwise because `f32::mul_add` is extremely slow without FMA instructions.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_feature = "fma"))]
    {
        a * b + c
    }
}

/// A triangle of the HRIR sphere.
#[derive(Clone, Debug, PartialEq)]
struct Triangle {
    /// Indices of the three corner points.
    vertices: [u32; 3],
    /// Row-major inverse of the matrix whose columns are the corner positions. Multiplying a
    /// direction by this yields its (unnormalized) barycentric coordinates in the triangle.
    /// All zeros for degenerate triangles.
    inverse: [f32; 9],
    /// The triangle across the edge that is opposite to each corner.
    neighbors: [u32; 3],
}

impl Triangle {
    /// Gets the unnormalized barycentric coordinates of `dir`. They are all
    /// non-negative, and sum to a positive number, exactly when a ray along
    /// `dir` passes through the triangle.
    #[inline(always)]
    fn barycentric(&self, dir: Vec3) -> [f32; 3] {
        let m = &self.inverse;
        [
            m[0] * dir.x + m[1] * dir.y + m[2] * dir.z,
            m[3] * dir.x + m[4] * dir.y + m[5] * dir.z,
            m[6] * dir.x + m[7] * dir.y + m[8] * dir.z,
        ]
    }
}

/// Tests whether the barycentric coordinates `lambda` lie inside of a triangle, and if so
/// returns them as normalized weights.
#[inline(always)]
fn weights_if_inside(lambda: [f32; 3]) -> Option<[f32; 3]> {
    let sum = lambda[0] + lambda[1] + lambda[2];
    let min = lambda[0].min(lambda[1]).min(lambda[2]);
    if sum > 0.0 && min >= -INSIDE_EPSILON * sum {
        // Clamp the tiny negative values allowed by the epsilon.
        let w = [lambda[0].max(0.0), lambda[1].max(0.0), lambda[2].max(0.0)];
        let inv = 1.0 / (w[0] + w[1] + w[2]);
        Some([w[0] * inv, w[1] * inv, w[2] * inv])
    } else {
        None
    }
}

#[derive(Debug, PartialEq)]
struct HrtfSphereInner {
    /// Length of every HRIR.
    length: usize,
    /// All impulse responses. For point `p` and tap `k`, the left sample is at index
    /// `2 * (p * length + k)` and the right sample is right after it.
    hrirs: Vec<f32>,
    triangles: Vec<Triangle>,
    source: PathBuf,
}

/// A triangulated sphere of impulse responses, prepared for time-domain convolution.
/// This is cheap to clone, and can be shared between any number of [`HrtfProcessor`]s.
#[derive(Clone, Debug, PartialEq)]
pub struct HrtfSphere(Arc<HrtfSphereInner>);

impl HrtfSphere {
    /// Prepares `hrir_sphere` for use with an [`HrtfProcessor`].
    ///
    /// # Panics
    ///
    /// If the sphere has no triangles, or if a triangle refers to a point that does not exist.
    pub fn new(hrir_sphere: &HrirSphere) -> Self {
        assert!(
            !hrir_sphere.faces.is_empty(),
            "HRIR sphere must have at least one triangle"
        );

        let length = hrir_sphere.length;
        let mut hrirs = Vec::with_capacity(hrir_sphere.points.len() * length * 2);
        for point in &hrir_sphere.points {
            for (&left, &right) in point.left_hrir.iter().zip(&point.right_hrir) {
                hrirs.push(left);
                hrirs.push(right);
            }
        }

        let mut triangles = hrir_sphere
            .faces
            .iter()
            .map(|face| {
                let vertices = [face.a, face.b, face.c].map(|v| {
                    assert!(v < hrir_sphere.points.len(), "triangle uses missing point");
                    v as u32
                });
                let [a, b, c] = vertices.map(|v| hrir_sphere.points[v as usize].pos);

                // The rows of the inverse of [a b c] are the dual basis vectors.
                let det = a.dot(b.cross(c));
                let inverse = if det.abs() > 1e-12 {
                    let inv = 1.0 / det;
                    let (ra, rb, rc) = (
                        b.cross(c).scale(inv),
                        c.cross(a).scale(inv),
                        a.cross(b).scale(inv),
                    );
                    [ra.x, ra.y, ra.z, rb.x, rb.y, rb.z, rc.x, rc.y, rc.z]
                } else {
                    [0.0; 9]
                };

                Triangle {
                    vertices,
                    inverse,
                    neighbors: [NO_NEIGHBOR; 3],
                }
            })
            .collect::<Vec<_>>();

        // Link triangles that share an edge.
        let mut edges = HashMap::<(u32, u32), Vec<(u32, usize)>>::new();
        for (index, triangle) in triangles.iter().enumerate() {
            for slot in 0..3 {
                let p = triangle.vertices[(slot + 1) % 3];
                let q = triangle.vertices[(slot + 2) % 3];
                edges
                    .entry((p.min(q), p.max(q)))
                    .or_default()
                    .push((index as u32, slot));
            }
        }
        for sharing in edges.values() {
            if let [(a, a_slot), (b, b_slot)] = sharing[..] {
                triangles[a as usize].neighbors[a_slot] = b;
                triangles[b as usize].neighbors[b_slot] = a;
            }
        }

        Self(Arc::new(HrtfSphereInner {
            length,
            hrirs,
            triangles,
            source: hrir_sphere.source.clone(),
        }))
    }

    /// The length of each impulse response, in samples.
    pub fn len(&self) -> usize {
        self.0.length
    }

    /// Returns a path to resource from which HrtfSphere was created.
    pub fn source(&self) -> &Path {
        &self.0.source
    }
}

impl HrtfSphereInner {
    /// Finds the triangle that `dir` points into, and the barycentric weights of `dir`
    /// within it. The search starts at triangle `hint`, which should be a recent result: moving
    /// sources cross at most a few triangle edges between samples.
    fn locate(&self, dir: Vec3, hint: usize) -> (usize, [f32; 3]) {
        let mut index = hint;
        for _ in 0..MAX_WALK_STEPS {
            let triangle = &self.triangles[index];
            let lambda = triangle.barycentric(dir);
            if let Some(weights) = weights_if_inside(lambda) {
                return (index, weights);
            }

            // Cross the edge that the direction is furthest beyond.
            let sum = lambda[0] + lambda[1] + lambda[2];
            let worst = if lambda[0] <= lambda[1] && lambda[0] <= lambda[2] {
                0
            } else if lambda[1] <= lambda[2] {
                1
            } else {
                2
            };
            let next = triangle.neighbors[worst];
            if sum <= 0.0 || next == NO_NEIGHBOR {
                break;
            }
            index = next as usize;
        }

        self.locate_exhaustive(dir)
    }

    /// Checks every triangle. This is the fallback for big jumps in direction.
    fn locate_exhaustive(&self, dir: Vec3) -> (usize, [f32; 3]) {
        let mut best = (0, f32::NEG_INFINITY);
        for (index, triangle) in self.triangles.iter().enumerate() {
            let lambda = triangle.barycentric(dir);
            if let Some(weights) = weights_if_inside(lambda) {
                return (index, weights);
            }

            // Track the closest miss in case rounding makes every triangle reject `dir`.
            let sum = lambda[0] + lambda[1] + lambda[2];
            if sum > 0.0 {
                let score = lambda[0].min(lambda[1]).min(lambda[2]) / sum;
                if score > best.1 {
                    best = (index, score);
                }
            }
        }

        let lambda = self.triangles[best.0].barycentric(dir);
        let w = [lambda[0].max(0.0), lambda[1].max(0.0), lambda[2].max(0.0)];
        let sum = w[0] + w[1] + w[2];
        if sum > 0.0 {
            (best.0, [w[0] / sum, w[1] / sum, w[2] / sum])
        } else {
            (best.0, [1.0, 0.0, 0.0])
        }
    }
}

/// Adds `w0 * h0 + w1 * h1 + w2 * h2` to `acc`.
#[inline(always)]
fn scatter(acc: &mut [f32], h0: &[f32], h1: &[f32], h2: &[f32], w0: f32, w1: f32, w2: f32) {
    for (((a, &x0), &x1), &x2) in acc.iter_mut().zip(h0).zip(h1).zip(h2) {
        *a = fma(w0, x0, fma(w1, x1, fma(w2, x2, *a)));
    }
}

/// Whether `dir` can be used to look up an impulse response.
#[inline(always)]
fn is_usable_direction(dir: Vec3) -> bool {
    let length_sq = dir.dot(dir);
    length_sq > 1e-20 && length_sq.is_finite()
}

/// Filters a mono sound source with the HRIRs for the direction that the source is
/// in, producing stereo output. Each processor holds the state for a single source, and
/// should be fed that source's samples in order.
///
/// See the crate docs for details.
#[derive(Clone, Debug, PartialEq)]
pub struct HrtfProcessor {
    sphere: HrtfSphere,
    /// Stereo output that has been partially computed by the samples seen so far.
    /// The sample at index `i` is the output that will be produced `i` samples from now.
    accumulator: Vec<f32>,
    /// The triangle that the previous sample pointed into.
    triangle: usize,
    /// The direction after the most recently processed sample.
    direction: Vec3,
    /// The gain after the most recently processed sample.
    gain: f32,
    /// Whether any samples have been processed yet.
    started: bool,
}

impl HrtfProcessor {
    /// Creates a new processor for a single source.
    pub fn new(sphere: HrtfSphere) -> Self {
        let accumulator = vec![0.0; 2 * (sphere.len() + CHUNK - 1)];
        Self {
            sphere,
            accumulator,
            triangle: 0,
            direction: Vec3::new(0.0, 0.0, 1.0),
            gain: 1.0,
            started: false,
        }
    }

    /// The sphere that this processor uses.
    pub fn sphere(&self) -> &HrtfSphere {
        &self.sphere
    }

    /// The number of samples that the output continues for after the input becomes silent.
    /// A source can be flushed by processing this many zeros.
    pub fn tail_len(&self) -> usize {
        self.sphere.len() - 1
    }

    /// Forgets all earlier samples and the previous direction and gain.
    pub fn reset(&mut self) {
        self.accumulator.fill(0.0);
        self.started = false;
    }

    /// Processes `source` and adds the result to `output`, which must be at least as long.
    ///
    /// `direction` is the vector from the listener to the sound source, in the sphere's coordinate
    /// system (so already transformed into the listener's space). It does not need to be
    /// normalized. The direction and `gain` are linearly interpolated from the values at the end of
    /// the previous call to these ones, at the end of `source`, once per sample. The first call
    /// after creation or a [`reset`](Self::reset) starts at these values immediately.
    ///
    /// Unusable directions (zero or non-finite) are replaced with the previous direction.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hrtf::{HrirSphere, HrtfProcessor, HrtfSphere, Vec3};
    ///
    /// let hrir_sphere = HrirSphere::from_file("your_file", 44100).unwrap();
    /// let mut processor = HrtfProcessor::new(HrtfSphere::new(&hrir_sphere));
    ///
    /// let source = vec![0.0; 128]; // Fill with something useful.
    /// let mut output = vec![(0.0, 0.0); 128];
    /// processor.process(&source, &mut output, Vec3::new(0.0, 0.0, 1.0), 1.0);
    /// ```
    pub fn process(
        &mut self,
        source: &[f32],
        output: &mut [(f32, f32)],
        direction: Vec3,
        gain: f32,
    ) {
        if source.is_empty() {
            return;
        }

        let direction = if is_usable_direction(direction) {
            direction
        } else {
            self.direction
        };
        let (start_direction, start_gain) = if self.started {
            (self.direction, self.gain)
        } else {
            (direction, gain)
        };

        let step = 1.0 / source.len() as f32;
        self.process_with(source, output, |i| {
            let t = (i + 1) as f32 * step;
            (
                start_direction.lerp(direction, t),
                lerpf(start_gain, gain, t),
            )
        });
    }

    /// Processes `source` and adds the result to `output`, using a different direction and gain
    /// for every sample. `directions` and `gains` must be the same length as `source`, and
    /// `output` must be at least as long. See [`process`](Self::process) for how directions are
    /// interpreted.
    pub fn process_per_sample(
        &mut self,
        source: &[f32],
        directions: &[Vec3],
        gains: &[f32],
        output: &mut [(f32, f32)],
    ) {
        assert_eq!(source.len(), directions.len());
        assert_eq!(source.len(), gains.len());
        self.process_with(source, output, |i| (directions[i], gains[i]));
    }

    /// Runs the convolution. `at(i)` gives the direction and gain for input sample `i`.
    fn process_with(
        &mut self,
        source: &[f32],
        output: &mut [(f32, f32)],
        mut at: impl FnMut(usize) -> (Vec3, f32),
    ) {
        assert!(output.len() >= source.len());

        let sphere = &*self.sphere.0;
        let stride = 2 * sphere.length;
        let accumulator = &mut self.accumulator[..];
        let mut direction = self.direction;
        let mut gain = self.gain;
        let mut triangle = self.triangle;

        let chunks = source.chunks(CHUNK).zip(output.chunks_mut(CHUNK));
        for (chunk_index, (input, output)) in chunks.enumerate() {
            for (j, &sample) in input.iter().enumerate() {
                let (sample_direction, sample_gain) = at(chunk_index * CHUNK + j);
                if is_usable_direction(sample_direction) {
                    direction = sample_direction;
                }
                gain = sample_gain;

                let scaled = sample * sample_gain;
                if scaled.abs() > SILENCE {
                    let (found, w) = sphere.locate(direction, triangle);
                    triangle = found;

                    let [v0, v1, v2] = sphere.triangles[found].vertices;
                    let hrir = |v: u32| &sphere.hrirs[v as usize * stride..][..stride];
                    scatter(
                        &mut accumulator[2 * j..][..stride],
                        hrir(v0),
                        hrir(v1),
                        hrir(v2),
                        scaled * w[0],
                        scaled * w[1],
                        scaled * w[2],
                    );
                }
            }

            // The first `input.len()` accumulated samples are now complete.
            for (j, out) in output.iter_mut().enumerate() {
                out.0 += accumulator[2 * j];
                out.1 += accumulator[2 * j + 1];
            }

            // Advance the accumulator by that many samples.
            let n = 2 * input.len();
            accumulator.copy_within(n.., 0);
            let end = accumulator.len();
            accumulator[end - n..].fill(0.0);
        }

        self.direction = direction;
        self.gain = gain;
        self.triangle = triangle;
        self.started = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Face, HrirPoint};

    /// Deterministic pseudorandom numbers in `[-1, 1)`.
    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 8) as f32 / (1 << 23) as f32 - 1.0
        }
    }

    /// An octahedron subdivided once: 18 points and 32 triangles, with random decaying HRIRs.
    fn test_sphere(length: usize) -> HrirSphere {
        let mut positions = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let octahedron = [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];

        let mut midpoints = HashMap::new();
        let mut faces = Vec::new();
        for [a, b, c] in octahedron {
            let mut mid = |p: usize, q: usize| {
                *midpoints.entry((p.min(q), p.max(q))).or_insert_with(|| {
                    let m = positions[p] + positions[q];
                    positions.push(m.scale(1.0 / m.dot(m).sqrt()));
                    positions.len() - 1
                })
            };
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            for [x, y, z] in [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]] {
                faces.push(Face { a: x, b: y, c: z });
            }
        }

        let mut rng = Lcg(7);
        let points = positions
            .into_iter()
            .map(|pos| {
                let mut hrir = || {
                    (0..length)
                        .map(|k| rng.next() * (-(k as f32) / (length as f32 / 3.0)).exp())
                        .collect()
                };
                HrirPoint {
                    pos,
                    left_hrir: hrir(),
                    right_hrir: hrir(),
                }
            })
            .collect();

        HrirSphere {
            length,
            points,
            faces,
            source: PathBuf::new(),
        }
    }

    fn random_direction(rng: &mut Lcg) -> Vec3 {
        loop {
            let v = Vec3::new(rng.next(), rng.next(), rng.next());
            let length_sq = v.dot(v);
            if length_sq > 0.05 && length_sq <= 1.0 {
                return v.scale(1.0 / length_sq.sqrt());
            }
        }
    }

    /// A direction that sweeps over many triangles of the sphere.
    fn spiral(i: usize, n: usize) -> Vec3 {
        let a = i as f32 * 0.05;
        let b = 0.2 + 2.7 * i as f32 / n as f32;
        Vec3::new(a.cos() * b.sin(), a.sin() * b.sin(), b.cos())
    }

    /// An independent (and slow) way to find the HRIR weights: intersect a ray with each triangle.
    fn reference_weights(sphere: &HrirSphere, dir: Vec3) -> Vec<(usize, f32)> {
        for face in &sphere.faces {
            let [a, b, c] = [face.a, face.b, face.c].map(|i| sphere.points[i].pos);
            let n = (b - a).cross(c - a);
            let t = a.dot(n) / dir.dot(n);
            if t <= 0.0 {
                continue;
            }
            let p = dir.scale(t);
            let nn = n.dot(n);
            let wa = (b - p).cross(c - p).dot(n) / nn;
            let wb = (c - p).cross(a - p).dot(n) / nn;
            let wc = (a - p).cross(b - p).dot(n) / nn;
            if wa >= -1e-5 && wb >= -1e-5 && wc >= -1e-5 {
                return vec![(face.a, wa), (face.b, wb), (face.c, wc)];
            }
        }
        panic!("no triangle for {:?}", dir);
    }

    /// Directly evaluates `y[n] = sum_m x[m] * gain[m] * h_dir(m)[n - m]`.
    fn reference_convolution(
        sphere: &HrirSphere,
        source: &[f32],
        directions: &[Vec3],
        gains: &[f32],
    ) -> Vec<(f32, f32)> {
        let len = sphere.length;
        let mut output = vec![(0.0f32, 0.0f32); source.len() + len - 1];
        for m in 0..source.len() {
            for (point, w) in reference_weights(sphere, directions[m]) {
                let point = &sphere.points[point];
                for k in 0..len {
                    let s = source[m] * gains[m] * w;
                    output[m + k].0 += s * point.left_hrir[k];
                    output[m + k].1 += s * point.right_hrir[k];
                }
            }
        }
        output
    }

    fn max_difference(a: &[(f32, f32)], b: &[(f32, f32)]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.0 - y.0).abs().max((x.1 - y.1).abs()))
            .fold(0.0, f32::max)
    }

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut rng = Lcg(seed);
        (0..n).map(|_| rng.next()).collect()
    }

    #[test]
    fn locate_finds_valid_weights_from_any_hint() {
        let hrir = test_sphere(4);
        let sphere = HrtfSphere::new(&hrir);
        let mut rng = Lcg(99);

        for _ in 0..2000 {
            let dir = random_direction(&mut rng);
            let hint = (rng.next().abs() * 31.99) as usize;
            let (index, w) = sphere.0.locate(dir, hint);

            assert!(w.iter().all(|&x| x >= 0.0));
            assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-5);

            // The blended corner position must point along `dir`.
            let [a, b, c] = sphere.0.triangles[index]
                .vertices
                .map(|v| hrir.points[v as usize].pos);
            let blend = a.scale(w[0]) + b.scale(w[1]) + c.scale(w[2]);
            let sin = blend.cross(dir);
            let sin = sin.dot(sin).sqrt() / (blend.dot(blend).sqrt() * dir.dot(dir).sqrt());
            assert!(
                sin < 1e-4 && blend.dot(dir) > 0.0,
                "{:?} -> {:?}",
                dir,
                blend
            );
        }
    }

    #[test]
    fn locate_handles_points_and_edges() {
        let hrir = test_sphere(4);
        let sphere = HrtfSphere::new(&hrir);

        for (i, point) in hrir.points.iter().enumerate() {
            for hint in [0, 13, 31] {
                let (_, w) = sphere.0.locate(point.pos.scale(3.0), hint);
                let corner: Vec<_> = sphere.0.triangles.iter().collect();
                // Some triangle containing point `i` was chosen, and it carries all of the weight.
                let found = corner.iter().any(|t| {
                    t.vertices
                        .iter()
                        .zip(w)
                        .any(|(&v, w)| v as usize == i && w > 0.999)
                });
                assert!(found, "point {}: {:?}", i, w);
            }
        }
    }

    #[test]
    fn impulse_at_a_vertex_reproduces_its_hrir() {
        // Longer than CHUNK, so the accumulator has to carry data between chunks.
        let hrir = test_sphere(200);
        let mut processor = HrtfProcessor::new(HrtfSphere::new(&hrir));

        let mut source = vec![0.0; 500];
        source[0] = 1.0;
        let mut output = vec![(0.0, 0.0); 500];
        processor.process(&source, &mut output, hrir.points[7].pos.scale(2.0), 1.0);

        for (k, out) in output.iter().take(200).enumerate() {
            assert!((out.0 - hrir.points[7].left_hrir[k]).abs() < 1e-6);
            assert!((out.1 - hrir.points[7].right_hrir[k]).abs() < 1e-6);
        }
        assert!(output[200..].iter().all(|&o| o == (0.0, 0.0)));
        assert_eq!(processor.tail_len(), 199);
    }

    #[test]
    fn matches_reference_for_moving_source() {
        let hrir = test_sphere(150);
        let n = 700;
        let source = noise(n, 1);
        let gains = (0..n)
            .map(|i| 0.5 + 0.5 * (i as f32 * 0.01).sin())
            .collect::<Vec<_>>();
        let directions = (0..n).map(|i| spiral(i, n)).collect::<Vec<_>>();

        let expected = reference_convolution(&hrir, &source, &directions, &gains);

        // Flush the tail by feeding zeros.
        let total = n + hrir.length - 1;
        let mut padded = source.clone();
        padded.resize(total, 0.0);
        let mut dirs = directions.clone();
        dirs.resize(total, directions[n - 1]);
        let mut gs = gains.clone();
        gs.resize(total, 0.0);

        let mut processor = HrtfProcessor::new(HrtfSphere::new(&hrir));
        let mut output = vec![(0.0, 0.0); total];
        processor.process_per_sample(&padded, &dirs, &gs, &mut output);

        let diff = max_difference(&output, &expected);
        assert!(diff < 1e-4, "max difference {}", diff);
    }

    #[test]
    fn splitting_the_input_does_not_change_the_output() {
        let hrir = test_sphere(150);
        let n = 600;
        let source = noise(n, 5);
        let gains = vec![0.8; n];
        let directions = (0..n).map(|i| spiral(i, n)).collect::<Vec<_>>();

        let mut whole = HrtfProcessor::new(HrtfSphere::new(&hrir));
        let mut expected = vec![(0.0, 0.0); n];
        whole.process_per_sample(&source, &directions, &gains, &mut expected);

        let mut split = HrtfProcessor::new(HrtfSphere::new(&hrir));
        let mut actual = vec![(0.0, 0.0); n];
        let mut start = 0;
        for len in [1, 127, 128, 129, 5, 64, 200, 1000] {
            let end = (start + len).min(n);
            split.process_per_sample(
                &source[start..end],
                &directions[start..end],
                &gains[start..end],
                &mut actual[start..end],
            );
            start = end;
        }

        assert!(max_difference(&actual, &expected) < 1e-6);
    }

    #[test]
    fn process_interpolates_direction_and_gain_per_sample() {
        let hrir = test_sphere(100);
        let (a, b) = (Vec3::new(0.3, 0.5, 0.8), Vec3::new(-0.6, 0.2, -0.7));
        let (gain_a, gain_b) = (1.0, 0.25);
        let n = 256;
        let source = noise(2 * n, 3);

        let mut blocks = HrtfProcessor::new(HrtfSphere::new(&hrir));
        let mut actual = vec![(0.0, 0.0); 2 * n];
        // The first call snaps to its values, the second ramps from them.
        blocks.process(&source[..n], &mut actual[..n], a, gain_a);
        blocks.process(&source[n..], &mut actual[n..], b, gain_b);

        let mut directions = vec![a; n];
        let mut gains = vec![gain_a; n];
        for i in 0..n {
            let t = (i + 1) as f32 / n as f32;
            directions.push(a.lerp(b, t));
            gains.push(lerpf(gain_a, gain_b, t));
        }
        let mut samples = HrtfProcessor::new(HrtfSphere::new(&hrir));
        let mut expected = vec![(0.0, 0.0); 2 * n];
        samples.process_per_sample(&source, &directions, &gains, &mut expected);

        assert!(max_difference(&actual, &expected) < 1e-6);
    }

    #[test]
    fn output_is_continuous_in_direction() {
        // Nudging every direction by a tiny amount must nudge the output by a tiny amount,
        // including at the samples where the nudge moves a direction into a different triangle.
        // A filter that switched abruptly at triangle edges would show jumps far above this.
        let hrir = test_sphere(100);
        let n = 4000;
        let source = noise(n, 11);
        let gains = vec![1.0; n];
        let run = |nudge: f32| {
            let directions = (0..n)
                .map(|i| spiral(i, n) + Vec3::new(nudge, -nudge, nudge))
                .collect::<Vec<_>>();
            let mut processor = HrtfProcessor::new(HrtfSphere::new(&hrir));
            let mut output = vec![(0.0, 0.0); n];
            processor.process_per_sample(&source, &directions, &gains, &mut output);
            output
        };

        let base = run(0.0);
        for nudge in [1e-3, 1e-4] {
            let diff = max_difference(&base, &run(nudge));
            eprintln!("nudge {}: max output change {}", nudge, diff);
            assert!(
                diff < 40.0 * nudge,
                "nudge {} changed the output by {}",
                nudge,
                diff
            );
        }
    }

    #[test]
    fn unusable_directions_keep_the_previous_one() {
        let hrir = test_sphere(50);
        let source = noise(300, 9);
        let dir = Vec3::new(0.2, 0.9, -0.4);

        let mut expected = vec![(0.0, 0.0); 300];
        let mut p = HrtfProcessor::new(HrtfSphere::new(&hrir));
        p.process(&source[..150], &mut expected[..150], dir, 1.0);
        p.process(&source[150..], &mut expected[150..], dir, 1.0);

        let mut actual = vec![(0.0, 0.0); 300];
        let mut p = HrtfProcessor::new(HrtfSphere::new(&hrir));
        p.process(&source[..150], &mut actual[..150], dir, 1.0);
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        p.process(&source[150..], &mut actual[150..], nan, 1.0);

        assert!(max_difference(&actual, &expected) < 1e-6);
    }

    #[test]
    fn nan_input_is_ignored_and_reset_clears_state() {
        let hrir = test_sphere(50);
        let mut p = HrtfProcessor::new(HrtfSphere::new(&hrir));
        let dir = Vec3::new(0.0, 0.0, 1.0);

        let mut output = vec![(0.0, 0.0); 100];
        p.process(&[f32::NAN; 100], &mut output, dir, 1.0);
        assert!(output.iter().all(|o| o.0 == 0.0 && o.1 == 0.0));

        p.process(&noise(100, 2), &mut output, dir, 1.0);
        p.reset();
        let mut after = vec![(0.0, 0.0); 100];
        p.process(&[0.0; 100], &mut after, dir, 1.0);
        assert!(after.iter().all(|o| o.0 == 0.0 && o.1 == 0.0));
    }

    #[test]
    fn sphere_links_neighbors_symmetrically() {
        let sphere = HrtfSphere::new(&test_sphere(4));
        for (i, t) in sphere.0.triangles.iter().enumerate() {
            for (slot, &n) in t.neighbors.iter().enumerate() {
                // A closed mesh: every edge is shared with exactly one other triangle.
                assert_ne!(n, NO_NEIGHBOR);
                let other = &sphere.0.triangles[n as usize];
                assert!(other.neighbors.contains(&(i as u32)));

                // The shared edge is the one opposite to `slot`.
                let edge = [t.vertices[(slot + 1) % 3], t.vertices[(slot + 2) % 3]];
                assert!(edge.iter().all(|v| other.vertices.contains(v)));
            }
        }
    }
}
