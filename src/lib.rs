//! Head-Related Transfer Function (HRTF) renderer.
//!
//! # Overview
//!
//! HRTF stands for [Head-Related Transfer Function](https://en.wikipedia.org/wiki/Head-related_transfer_function)
//! and can work only with spatial sounds. For each of such sound source after it was processed by HRTF you can
//! definitely tell from which location sound came from. In other words HRTF improves perception of sound to
//! the level of real life.
//!
//! # HRIR Spheres
//!
//! This crate uses Head-Related Impulse Response (HRIR) spheres to create HRTF spheres. HRTF sphere is a set of
//! points in 3D space which are connected into a mesh forming triangulated sphere. Each point contains spectrum
//! for left and right ears which will be used to modify samples from each spatial sound source to create binaural
//! sound. HRIR spheres can be found [here](https://github.com/mrDIMAS/hrir_sphere_builder/tree/master/hrtf_base/IRCAM).
//! HRIR spheres from the base are recorded in 44100 Hz sample rate, this crate performs **automatic** resampling to your
//! sample rate.
//!
//! # Performance
//!
//! HRTF is **heavy**, this is essential because HRTF requires some heavy math (fast Fourier transform, convolution,
//! etc.) and lots of memory copying.
//!
//! # Known problems
//!
//! This renderer still suffers from small audible clicks in very fast moving sounds, clicks sounds more like
//! "buzzing" - it is due the fact that hrtf is different from frame to frame which gives "bumps" in amplitude
//! of signal because of phase shift each impulse response have. This can be fixed by short cross fade between
//! small amount of samples from previous frame with same amount of frames of current as proposed in
//! [here](http://csoundjournal.com/issue9/newHRTFOpcodes.html)
//!
//! Clicks can be reproduced by using clean sine wave of 440 Hz on some source moving around listener.
//!
//! # Algorithm
//!
//! This crate uses overlap-save convolution to perform operations in frequency domain. Check
//! [this link](https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method) for more info.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Fast Fourier transform.
extern crate rustfft;

// File reading.
extern crate byteorder;

// Resampling.
extern crate rubato;

use byteorder::{LittleEndian, ReadBytesExt};
use rubato::Resampler;
use rustfft::{num_complex::Complex, num_traits::Zero, FFTplanner};
use std::fmt::{Debug, Formatter};
use std::ops::{Add, Sub};
use std::path::PathBuf;
use std::{
    fs::File,
    io::{BufReader, Error, Read},
    path::Path,
};

/// Simple 3d vector.
#[derive(Copy, Clone)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Initializes new vector.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Vec3 {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn normalize(self) -> Vec3 {
        let i = 1.0 / self.dot(self).sqrt();
        Vec3 {
            x: self.x * i,
            y: self.y * i,
            z: self.z * i,
        }
    }

    fn scale(self, k: f32) -> Self {
        Self {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: lerpf(self.x, other.x, t),
            y: lerpf(self.y, other.y, t),
            z: lerpf(self.z, other.z, t),
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

struct BaryCoords {
    u: f32,
    v: f32,
    w: f32,
}

impl BaryCoords {
    fn inside(&self) -> bool {
        (self.u >= 0.0) && (self.v >= 0.0) && (self.u + self.v < 1.0)
    }
}

fn get_barycentric_coords(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> BaryCoords {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    BaryCoords { u, v, w }
}

fn ray_triangle_intersection(origin: Vec3, dir: Vec3, vertices: &[Vec3; 3]) -> Option<BaryCoords> {
    let ba = vertices[1] - vertices[0];
    let ca = vertices[2] - vertices[0];

    let normal = ba.cross(ca).normalize();
    let d = -vertices[0].dot(normal);

    let u = -(origin.dot(normal) + d);
    let v = dir.dot(normal);
    let t = u / v;

    if t >= 0.0 && t <= 1.0 {
        let point = origin + dir.scale(t);
        let bary = get_barycentric_coords(point, vertices[0], vertices[1], vertices[2]);
        if bary.inside() {
            return Some(bary);
        }
    }
    None
}

/// All possible error that can occur during HRIR sphere loading.
#[derive(Debug)]
pub enum HrtfError {
    /// An io error has occurred (file does not exists, etc.)
    IoError(std::io::Error),

    /// It is not valid HRIR sphere file.
    InvalidFileFormat,

    /// HRIR has invalid length (zero).
    InvalidLength(usize),
}

impl From<std::io::Error> for HrtfError {
    fn from(io_err: Error) -> Self {
        HrtfError::IoError(io_err)
    }
}

#[derive(Clone)]
struct Face {
    a: usize,
    b: usize,
    c: usize,
}

/// See module docs.
#[derive(Clone)]
pub struct HrtfSphere {
    length: usize,
    points: Vec<HrtfPoint>,
    faces: Vec<Face>,
    source: PathBuf,
}

fn make_hrtf(
    hrir: Vec<f32>,
    pad_length: usize,
    planner: &mut FFTplanner<f32>,
) -> Vec<Complex<f32>> {
    let mut hrir = hrir
        .into_iter()
        .map(|s| Complex::new(s, 0.0))
        .collect::<Vec<Complex<f32>>>();
    for _ in hrir.len()..pad_length {
        // Pad with zeros to length of context's output buffer.
        hrir.push(Complex::zero());
    }
    let mut hrtf = vec![Complex::zero(); pad_length];
    planner
        .plan_fft(pad_length)
        .process(hrir.as_mut(), hrtf.as_mut());
    hrtf
}

fn read_hrir(reader: &mut dyn Read, len: usize) -> Result<Vec<f32>, HrtfError> {
    let mut hrir = Vec::with_capacity(len);
    for _ in 0..len {
        hrir.push(reader.read_f32::<LittleEndian>()?);
    }
    Ok(hrir)
}

fn resample_hrir(hrir: Vec<f32>, ratio: f64) -> Vec<f32> {
    if ratio.eq(&1.0) {
        hrir
    } else {
        let params = rubato::InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 160,
            interpolation: rubato::InterpolationType::Cubic,
            window: rubato::WindowFunction::BlackmanHarris2,
        };

        let mut resampler = rubato::SincFixedIn::<f32>::new(ratio, params, hrir.len(), 1);
        let result = resampler.process(&[hrir]).unwrap();
        result.into_iter().next().unwrap()
    }
}

fn read_faces(reader: &mut dyn Read, index_count: usize) -> Result<Vec<Face>, HrtfError> {
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(reader.read_u32::<LittleEndian>()?);
    }
    let faces = indices
        .chunks(3)
        .map(|f| Face {
            a: f[0] as usize,
            b: f[1] as usize,
            c: f[2] as usize,
        })
        .collect();
    Ok(faces)
}

/// Single point of HRIR sphere. See module docs for more info.
#[derive(Clone)]
pub struct HrirPoint {
    /// Position of point in cartesian coordinate space.
    pub pos: Vec3,
    left_hrir: Vec<f32>,
    right_hrir: Vec<f32>,
}

impl HrirPoint {
    /// Returns shared reference to spectrum for left ear.
    pub fn left_hrir(&self) -> &[f32] {
        &self.left_hrir
    }

    /// Returns shared reference to spectrum for right ear.
    pub fn right_hrir(&self) -> &[f32] {
        &self.right_hrir
    }
}

/// HRIR (Head-Related Impulse Response) spheres is a 3d mesh whose points contains impulse
/// responses for left and right ears. It is used for interpolation of impulse responses.
#[derive(Clone)]
pub struct HrirSphere {
    length: usize,
    points: Vec<HrirPoint>,
    faces: Vec<Face>,
    source: PathBuf,
}

impl HrirSphere {
    /// Tries to load a sphere from a file.
    pub fn from_file<P: AsRef<Path>>(path: P, device_sample_rate: u32) -> Result<Self, HrtfError> {
        let mut sphere = Self::new(
            BufReader::new(File::open(path.as_ref())?),
            device_sample_rate,
        )?;
        sphere.source = path.as_ref().to_owned();
        Ok(sphere)
    }

    /// Loads HRIR sphere from given source.
    ///
    /// # Coordinate system
    ///
    /// Hrtf spheres made in *right-handed* coordinate system. This fact can give weird positioning issues
    /// if your application uses *left-handed* coordinate system. However this can be fixed very easily:
    /// iterate over every point and invert X coordinate of it.
    ///
    /// # Sample rate
    ///
    /// HRIR spheres from [this](https://github.com/mrDIMAS/hrir_sphere_builder/tree/master/hrtf_base/IRCAM)
    /// base recorded in 44100 Hz sample rate. If your output device uses different sample rate, you have to
    /// resample initial set of .wav files and regenerate HRIR spheres. There could
    pub fn new<R: Read>(mut reader: R, device_sample_rate: u32) -> Result<Self, HrtfError> {
        let mut magic = [0; 4];
        reader.read_exact(&mut magic)?;
        if magic[0] != b'H' && magic[1] != b'R' && magic[2] != b'I' && magic[3] != b'R' {
            return Err(HrtfError::InvalidFileFormat);
        }

        let sample_rate = reader.read_u32::<LittleEndian>()?;
        let length = reader.read_u32::<LittleEndian>()? as usize;
        if length == 0 {
            return Err(HrtfError::InvalidLength(length));
        }
        let vertex_count = reader.read_u32::<LittleEndian>()? as usize;
        let index_count = reader.read_u32::<LittleEndian>()? as usize;

        let faces = read_faces(&mut reader, index_count)?;

        let ratio = sample_rate as f64 / device_sample_rate as f64;

        let mut points = Vec::with_capacity(vertex_count);
        for _ in 0..vertex_count {
            let x = reader.read_f32::<LittleEndian>()?;
            let y = reader.read_f32::<LittleEndian>()?;
            let z = reader.read_f32::<LittleEndian>()?;

            let left_hrir = resample_hrir(read_hrir(&mut reader, length)?, ratio);
            let right_hrir = resample_hrir(read_hrir(&mut reader, length)?, ratio);

            points.push(HrirPoint {
                pos: Vec3 { x, y, z },
                left_hrir,
                right_hrir,
            });
        }

        Ok(Self {
            points,
            length,
            faces,
            source: Default::default(),
        })
    }

    /// Applies specified transform to each point in sphere. Can be used to rotate or scale sphere.
    /// Transform shouldn't have translation part, otherwise result of bilinear sampling is undefined.
    pub fn transform(&mut self, matrix: &[f32; 16]) {
        for pt in self.points.iter_mut() {
            let x = pt.pos.x * matrix[0] + pt.pos.y * matrix[4] + pt.pos.z * matrix[8] + matrix[12];
            let y = pt.pos.x * matrix[1] + pt.pos.y * matrix[5] + pt.pos.z * matrix[9] + matrix[13];
            let z =
                pt.pos.x * matrix[2] + pt.pos.y * matrix[6] + pt.pos.z * matrix[10] + matrix[14];

            pt.pos.x = x;
            pt.pos.y = y;
            pt.pos.z = z;
        }
    }

    /// Returns shared reference to sphere points array.
    pub fn points(&self) -> &[HrirPoint] {
        &self.points
    }

    /// Returns mutable reference to sphere points array.
    pub fn points_mut(&mut self) -> &mut [HrirPoint] {
        &mut self.points
    }

    /// Returns length of impulse response. It is same across all points in the sphere.
    pub fn len(&self) -> usize {
        self.length
    }
}

#[derive(Clone)]
struct HrtfPoint {
    pos: Vec3,
    left_hrtf: Vec<Complex<f32>>,
    right_hrtf: Vec<Complex<f32>>,
}

impl HrtfSphere {
    fn new(hrir_sphere: HrirSphere, block_len: usize) -> Self {
        let mut planner = FFTplanner::new(false);
        let pad_length = get_pad_len(hrir_sphere.length, block_len);

        let points = hrir_sphere
            .points
            .into_iter()
            .map(|p| {
                let left_hrtf = make_hrtf(p.left_hrir, pad_length, &mut planner);
                let right_hrtf = make_hrtf(p.right_hrir, pad_length, &mut planner);

                HrtfPoint {
                    pos: p.pos,
                    left_hrtf,
                    right_hrtf,
                }
            })
            .collect();

        Self {
            points,
            length: hrir_sphere.length,
            faces: hrir_sphere.faces,
            source: hrir_sphere.source,
        }
    }

    /// Returns a path to resource from which HrtfSphere was created.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Sampling with bilinear interpolation. See more info here http://www02.smt.ufrj.br/~diniz/conf/confi117.pdf
    fn sample_bilinear(
        &self,
        left_hrtf: &mut Vec<Complex<f32>>,
        right_hrtf: &mut Vec<Complex<f32>>,
        dir: Vec3,
    ) {
        let dir = dir.scale(10.0);

        for face in self.faces.iter() {
            let a = self.points.get(face.a).unwrap();
            let b = self.points.get(face.b).unwrap();
            let c = self.points.get(face.c).unwrap();

            if let Some(bary) = ray_triangle_intersection(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                dir,
                &[a.pos, b.pos, c.pos],
            ) {
                let len = a.left_hrtf.len();

                left_hrtf.clear();
                for i in 0..len {
                    left_hrtf.push(
                        a.left_hrtf[i] * bary.u + b.left_hrtf[i] * bary.v + c.left_hrtf[i] * bary.w,
                    );
                }

                right_hrtf.clear();
                for i in 0..len {
                    right_hrtf.push(
                        a.right_hrtf[i] * bary.u
                            + b.right_hrtf[i] * bary.v
                            + c.right_hrtf[i] * bary.w,
                    );
                }
            }
        }
    }
}

#[inline]
fn copy_replace(prev_samples: &mut Vec<f32>, raw_buffer: &mut [Complex<f32>], segment_len: usize) {
    if prev_samples.len() != segment_len {
        *prev_samples = vec![0.0; segment_len];
    }

    // Copy samples from previous iteration in the beginning of the buffer.
    for (prev_sample, raw_sample) in prev_samples.iter().zip(&mut raw_buffer[..segment_len]) {
        *raw_sample = Complex::new(*prev_sample, 0.0);
    }

    // Replace last samples by samples from end of the buffer for next iteration.
    let last_start = raw_buffer.len() - segment_len;
    for (prev_sample, raw_sample) in prev_samples.iter_mut().zip(&mut raw_buffer[last_start..]) {
        *prev_sample = raw_sample.re;
    }
}

// Overlap-save convolution. See more info here:
// https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method
//
// # Notes
//
// It is much faster than direct convolution (in case for long impulse responses
// and signals). Check table here: https://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution_vs_Direct.html
//
// I measured performance and direct convolution was 8-10 times slower than overlap-save convolution with impulse
// response length of 512 and signal length of 3545 samples.
#[inline]
fn convolve_overlap_save(
    in_buffer: &mut [Complex<f32>],
    out_buffer: &mut [Complex<f32>],
    hrtf: &[Complex<f32>],
    hrtf_len: usize,
    prev_samples: &mut Vec<f32>,
    fft: &mut FFTplanner<f32>,
    ifft: &mut FFTplanner<f32>,
) {
    assert_eq!(hrtf.len(), in_buffer.len());

    copy_replace(prev_samples, in_buffer, hrtf_len);

    fft.plan_fft(in_buffer.len()).process(in_buffer, out_buffer);

    // Multiply HRIR and input signal in frequency domain.
    for (s, h) in out_buffer.iter_mut().zip(hrtf.iter()) {
        *s *= *h;
    }

    ifft.plan_fft(in_buffer.len())
        .process(out_buffer, in_buffer);
}

#[inline]
fn get_pad_len(hrtf_len: usize, block_len: usize) -> usize {
    // Total length for each temporary buffer.
    // The value defined by overlap-add convolution method:
    //
    // pad_length = M + N - 1,
    //
    // where M - signal length, N - hrtf length
    block_len + hrtf_len - 1
}

/// See module docs.
pub struct HrtfProcessor {
    hrtf_sphere: HrtfSphere,
    left_in_buffer: Vec<Complex<f32>>,
    right_in_buffer: Vec<Complex<f32>>,
    left_out_buffer: Vec<Complex<f32>>,
    right_out_buffer: Vec<Complex<f32>>,
    fft: FFTplanner<f32>,
    ifft: FFTplanner<f32>,
    left_hrtf: Vec<Complex<f32>>,
    right_hrtf: Vec<Complex<f32>>,
    block_len: usize,
    interpolation_steps: usize,
}

impl Debug for HrtfProcessor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HrtfProcessor")
    }
}

impl Clone for HrtfProcessor {
    fn clone(&self) -> Self {
        Self {
            hrtf_sphere: self.hrtf_sphere.clone(),
            left_in_buffer: self.left_in_buffer.clone(),
            right_in_buffer: self.right_in_buffer.clone(),
            left_out_buffer: self.left_out_buffer.clone(),
            right_out_buffer: self.right_out_buffer.clone(),
            fft: FFTplanner::new(false),
            ifft: FFTplanner::new(true),
            left_hrtf: self.left_hrtf.clone(),
            right_hrtf: self.right_hrtf.clone(),
            block_len: self.block_len,
            interpolation_steps: self.interpolation_steps,
        }
    }
}

/// Provides unified way of extracting single channel (left) from any set of interleaved samples
/// (LLLLL..., LRLRLRLR..., etc).
pub trait InterleavedSamples {
    /// Returns first sample from set of interleaved samples.
    fn left(&self) -> f32;
}

impl InterleavedSamples for f32 {
    fn left(&self) -> f32 {
        *self
    }
}

impl InterleavedSamples for (f32, f32) {
    fn left(&self) -> f32 {
        self.0
    }
}

#[inline]
fn get_raw_samples<T: InterleavedSamples>(
    source: &[T],
    left: &mut [Complex<f32>],
    right: &mut [Complex<f32>],
    offset: usize,
) {
    assert_eq!(left.len(), right.len());

    for ((left, right), samples) in left.iter_mut().zip(right.iter_mut()).zip(&source[offset..]) {
        // Ignore all channels except left. Only mono sounds can be processed by HRTF.
        let sample = Complex::new(samples.left(), 0.0);
        *left = sample;
        *right = sample;
    }
}

/// Contains all input parameters for HRTF signal processing.
pub struct HrtfContext<'a, 'b, 'c, T: InterleavedSamples> {
    /// Source of interleaved samples to be processed. HRTF works **only** with mono sources, so source
    /// must implement InterleavedSamples trait which must provide sample from left channel only.
    /// Source must have `interpolation_steps * block_len` length!
    pub source: &'a [T],
    /// An output buffer to write processed samples to. It must be **stereo** buffer, processed samples
    /// will be mixed with samples in output buffer.
    pub output: &'b mut [(f32, f32)],
    /// New sampling vector. It must be a vector from a sound source position to a listener. If your
    /// listener has orientation, then you should transform this vector into a listener space first.
    pub new_sample_vector: Vec3,
    /// Sampling vector from previous frame.
    pub prev_sample_vector: Vec3,
    /// Left channel samples from last frame. It is used for continuous convolution. It must point to
    /// unique buffer which associated with a single sound source.
    pub prev_left_samples: &'c mut Vec<f32>,
    /// Right channel samples from last frame. It is used for continuous convolution. It must point to
    /// unique buffer which associated with a single sound source.
    pub prev_right_samples: &'c mut Vec<f32>,
    /// New distance gain for given slice. It is used to interpolate gain so output signal will have
    /// smooth transition from frame to frame. It is very important for click-free processing.
    pub new_distance_gain: f32,
    /// Distance gain from previous frame. It is used to interpolate gain so output signal will have
    /// smooth transition from frame to frame. It is very important for click-free processing.
    pub prev_distance_gain: f32,
}

impl HrtfProcessor {
    /// Creates new HRTF processor using specified HRTF sphere. See module docs for more info.
    ///
    /// `interpolation_steps` is the amount of slices to cut source to.
    /// `block_len` is the length of each slice.
    pub fn new(hrir_sphere: HrirSphere, interpolation_steps: usize, block_len: usize) -> Self {
        let hrtf_sphere = HrtfSphere::new(hrir_sphere, block_len);

        let pad_length = get_pad_len(hrtf_sphere.length, block_len);

        // Acquire default HRTFs for left and right channels.
        let pt = hrtf_sphere.points.first().unwrap();
        let left_hrtf = pt.left_hrtf.clone();
        let right_hrtf = pt.right_hrtf.clone();

        Self {
            hrtf_sphere,
            left_in_buffer: vec![Complex::zero(); pad_length],
            right_in_buffer: vec![Complex::zero(); pad_length],
            left_out_buffer: vec![Complex::zero(); pad_length],
            right_out_buffer: vec![Complex::zero(); pad_length],
            fft: FFTplanner::new(false),
            ifft: FFTplanner::new(true),
            left_hrtf,
            right_hrtf,
            block_len,
            interpolation_steps,
        }
    }

    /// Returns shared reference to current hrtf sphere.
    pub fn hrtf_sphere(&self) -> &HrtfSphere {
        &self.hrtf_sphere
    }

    /// Processes given input samples and sums processed signal with output buffer. This method designed
    /// to be used in a loop, it requires some info from previous frame. Check `HrtfContext` for more info.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hrtf::{HrirSphere, HrtfContext, HrtfProcessor, Vec3};
    /// let hrir_sphere = HrirSphere::from_file("your_file", 44100).unwrap();
    ///
    /// let mut processor = HrtfProcessor::new(hrir_sphere, 8, 128);
    ///
    /// let source = vec![0; 1024]; // Fill with something useful.
    /// let mut output = vec![(0.0, 0.0); 1024];
    /// let mut prev_left_samples = vec![];
    /// let mut prev_right_samples = vec![];
    ///
    /// let context = HrtfContext {
    ///     source: &source,
    ///     output: &mut output,
    ///     new_sample_vector: Vec3{x: 0.0, y: 0.0, z: 1.0},
    ///     prev_sample_vector: Vec3{x:0.0,y: 0.0, z: 1.0},
    ///     prev_left_samples: &mut prev_left_samples,
    ///     prev_right_samples: &mut prev_right_samples,
    ///     // For simplicity, keep gain at 1.0 so there will be no interpolation.
    ///     new_distance_gain: 1.0,
    ///     prev_distance_gain: 1.0
    /// };
    ///
    /// processor.process_samples(context);
    /// ```
    pub fn process_samples<T: InterleavedSamples>(&mut self, context: HrtfContext<T>) {
        let HrtfContext {
            source,
            output,
            new_sample_vector: sample_vector,
            prev_sample_vector,
            prev_left_samples,
            prev_right_samples,
            prev_distance_gain,
            new_distance_gain,
        } = context;

        let expected_len = self.interpolation_steps * self.block_len;
        assert_eq!(expected_len, source.len());
        assert!(output.len() >= expected_len);

        let new_sampling_vector = sample_vector;
        let prev_sampling_vector = prev_sample_vector;

        let pad_length = get_pad_len(self.hrtf_sphere.length, self.block_len);

        // Overlap-save convolution with HRTF interpolation.
        // It divides given output buffer into N parts, fetches samples from source,
        // performs convolution and writes processed samples to output buffer. Output
        // buffer divided into parts because of HRTF interpolation which significantly
        // reduces distortion in output signal.
        for step in 0..self.interpolation_steps {
            let next = step + 1;
            let out = &mut output[(step * self.block_len)..(next * self.block_len)];

            let t = next as f32 / self.interpolation_steps as f32;
            let sampling_vector = prev_sampling_vector.lerp(new_sampling_vector, t);
            self.hrtf_sphere.sample_bilinear(
                &mut self.left_hrtf,
                &mut self.right_hrtf,
                sampling_vector,
            );

            let hrtf_len = self.hrtf_sphere.length - 1;

            get_raw_samples(
                source,
                &mut self.left_in_buffer[hrtf_len..],
                &mut self.right_in_buffer[hrtf_len..],
                step * self.block_len,
            );

            convolve_overlap_save(
                &mut self.left_in_buffer,
                &mut self.left_out_buffer,
                &self.left_hrtf,
                hrtf_len,
                prev_left_samples,
                &mut self.fft,
                &mut self.ifft,
            );

            convolve_overlap_save(
                &mut self.right_in_buffer,
                &mut self.right_out_buffer,
                &self.right_hrtf,
                hrtf_len,
                prev_right_samples,
                &mut self.fft,
                &mut self.ifft,
            );

            // Mix samples into output buffer with rescaling and apply distance gain.
            let distance_gain = lerpf(prev_distance_gain, new_distance_gain, t);
            let k = distance_gain / (pad_length as f32);

            let left_payload = &self.left_in_buffer[hrtf_len..];
            let right_payload = &self.right_in_buffer[hrtf_len..];
            for ((out_left, out_right), (processed_left, processed_right)) in
                out.iter_mut().zip(left_payload.iter().zip(right_payload))
            {
                *out_left += processed_left.re * k;
                *out_right += processed_right.re * k;
            }
        }
    }
}
