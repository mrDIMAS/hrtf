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
//! This crate uses Head-Related Impulse Response (HRIR) spheres to create HRTF spheres. An HRIR sphere is a set of
//! points in 3D space which are connected into a mesh forming triangulated sphere. Each point contains the impulse
//! responses for the left and right ears. HRIR spheres can be found
//! [here](https://github.com/mrDIMAS/hrir_sphere_builder/tree/master/hrtf_base/IRCAM).
//! HRIR spheres from the base are recorded in 44100 Hz sample rate, this crate performs **automatic** resampling to your
//! sample rate.
//!
//! # Algorithm
//!
//! Convolution is done directly in the time domain, and the sound's direction is interpolated at the
//! audio rate (once per sample) instead of once per block. Each input sample `x[m]` is filtered with the
//! impulse response of the direction the source had when that sample was emitted:
//!
//! ```text
//! y[n] = Σ_m x[m] · h_dir(m)[n - m]
//! ```
//!
//! `h_dir` is the barycentric blend of the three HRIRs at the corners of the sphere triangle that
//! the direction points into. Barycentric interpolation over a mesh is continuous across the
//! triangles' shared edges, so the effective filter changes smoothly as a source moves - there are no
//! block-boundary filter switches, which is what makes fast-moving sounds "buzz" in block-based
//! renderers.
//!
//! Each input sample is *scattered* forward into a short accumulator holding the next `len` output samples
//! (where `len` is the HRIR length). That keeps all memory accesses contiguous, allows the triangle to change on
//! any sample with no extra bookkeeping, and means there is no FFT, no block latency and no allocation while
//! processing. The cost is constant: three multiply-adds per tap per ear for every non-silent input sample.

#![allow(clippy::len_without_is_empty)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::manual_range_contains)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Resampling.
extern crate rubato;

mod processor;

pub use processor::{HrtfProcessor, HrtfSphere};

use rubato::audioadapter_buffers::direct::SequentialSlice;
use rubato::Resampler;
use std::ops::{Add, Sub};
use std::path::PathBuf;
use std::{
    fs::File,
    io::{BufReader, Error, Read},
    path::Path,
};

/// Simple 3d vector.
#[derive(Copy, Clone, Debug, PartialEq)]
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

#[derive(Copy, Clone, Debug, PartialEq)]
struct Face {
    a: usize,
    b: usize,
    c: usize,
}

fn read_u32(reader: &mut dyn Read) -> Result<u32, HrtfError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut dyn Read) -> Result<f32, HrtfError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_hrir(reader: &mut dyn Read, len: usize) -> Result<Vec<f32>, HrtfError> {
    let mut hrir = Vec::with_capacity(len);
    for _ in 0..len {
        hrir.push(read_f32(reader)?);
    }
    Ok(hrir)
}

fn resample_hrir(hrir: Vec<f32>, resampler: &mut rubato::Async<f32>) -> Vec<f32> {
    resampler.reset();
    let input_len = hrir.len();
    let input = SequentialSlice::new(&hrir, 1, input_len).unwrap();
    let mut hrir = vec![0.0; resampler.process_all_needed_output_len(input_len)];
    let output_capacity = hrir.len();
    let mut output = SequentialSlice::new_mut(&mut hrir, 1, output_capacity).unwrap();
    let (_, output_len) = resampler
        .process_all_into_buffer(&input, &mut output, input_len, None)
        .unwrap();
    hrir.truncate(output_len);
    hrir
}

fn read_faces(reader: &mut dyn Read, index_count: usize) -> Result<Vec<Face>, HrtfError> {
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(read_u32(reader)?);
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
#[derive(Clone, Debug, PartialEq)]
pub struct HrirPoint {
    /// Position of point in cartesian coordinate space.
    pub pos: Vec3,
    left_hrir: Vec<f32>,
    right_hrir: Vec<f32>,
}

impl HrirPoint {
    /// Returns shared reference to the impulse response for the left ear.
    pub fn left_hrir(&self) -> &[f32] {
        &self.left_hrir
    }

    /// Returns shared reference to the impulse response for the right ear.
    pub fn right_hrir(&self) -> &[f32] {
        &self.right_hrir
    }
}

/// HRIR (Head-Related Impulse Response) spheres is a 3d mesh whose points contains impulse
/// responses for left and right ears. It is used for interpolation of impulse responses.
#[derive(Clone, Debug, PartialEq)]
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
    /// base recorded in 44100 Hz sample rate. If your output device uses different sample rate, the impulse
    /// responses are resampled to match it.
    pub fn new<R: Read>(mut reader: R, device_sample_rate: u32) -> Result<Self, HrtfError> {
        let mut magic = [0; 4];
        reader.read_exact(&mut magic)?;
        if magic[0] != b'H' && magic[1] != b'R' && magic[2] != b'I' && magic[3] != b'R' {
            return Err(HrtfError::InvalidFileFormat);
        }

        let sample_rate = read_u32(&mut reader)?;
        let length = read_u32(&mut reader)? as usize;
        if length == 0 {
            return Err(HrtfError::InvalidLength(length));
        }
        let vertex_count = read_u32(&mut reader)? as usize;
        let index_count = read_u32(&mut reader)? as usize;

        let faces = read_faces(&mut reader, index_count)?;

        let ratio = device_sample_rate as f64 / sample_rate as f64;
        let params = rubato::SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: Some(0.95),
            oversampling_factor: 160,
            interpolation: rubato::SincInterpolationType::Cubic,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let mut resampler = rubato::Async::<f32>::new_sinc(
            ratio,
            1.0,
            &params,
            length,
            1,
            rubato::FixedAsync::Input,
        )
        .unwrap();
        let mut points = Vec::with_capacity(vertex_count);
        let mut resampled_length = None;
        for _ in 0..vertex_count {
            let x = read_f32(&mut reader)?;
            let y = read_f32(&mut reader)?;
            let z = read_f32(&mut reader)?;

            let left_hrir = read_hrir(&mut reader, length)?;
            let right_hrir = read_hrir(&mut reader, length)?;

            let left_hrir = resample_hrir(left_hrir, &mut resampler);
            let right_hrir = resample_hrir(right_hrir, &mut resampler);
            let point_length = left_hrir.len();
            if point_length == 0 || right_hrir.len() != point_length {
                return Err(HrtfError::InvalidLength(point_length));
            }
            match resampled_length {
                None => resampled_length = Some(point_length),
                Some(expected_length) if expected_length != point_length => {
                    // Convolution assumes every HRIR in the sphere has the same length.
                    return Err(HrtfError::InvalidLength(point_length));
                }
                Some(_) => {}
            }

            points.push(HrirPoint {
                pos: Vec3 { x, y, z },
                left_hrir,
                right_hrir,
            });
        }

        Ok(Self {
            points,
            length: resampled_length.unwrap_or(length),
            faces,
            source: Default::default(),
        })
    }

    /// Applies specified transform to each point in sphere. Can be used to rotate or scale sphere.
    /// Transform shouldn't have translation part, otherwise result of sampling is undefined.
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

    /// Returns source file name (if any).
    pub fn source(&self) -> &Path {
        &self.source
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    pub fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_le_bytes());
    }

    pub fn push_f32(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend(value.to_le_bytes());
    }

    pub fn test_hrir_sphere(sample_rate: u32, hrir_len: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(b"HRIR");
        push_u32(&mut bytes, sample_rate);
        push_u32(&mut bytes, hrir_len as u32);
        push_u32(&mut bytes, 1);
        push_u32(&mut bytes, 0);

        push_f32(&mut bytes, 0.0);
        push_f32(&mut bytes, 0.0);
        push_f32(&mut bytes, 1.0);

        for i in 0..hrir_len {
            push_f32(&mut bytes, if i == 0 { 1.0 } else { 0.0 });
        }
        for i in 0..hrir_len {
            push_f32(&mut bytes, if i == 0 { 1.0 } else { 0.0 });
        }

        bytes
    }

    pub fn tetrahedron_hrir_sphere_bytes_with<F>(
        sample_rate: u32,
        hrir_len: usize,
        mut hrir: F,
    ) -> Vec<u8>
    where
        F: FnMut(usize) -> f32,
    {
        let mut bytes = Vec::new();
        bytes.extend(b"HRIR");
        push_u32(&mut bytes, sample_rate);
        push_u32(&mut bytes, hrir_len as u32);
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 12);

        for index in [0_u32, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2] {
            push_u32(&mut bytes, index);
        }

        for pos in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
        ] {
            push_f32(&mut bytes, pos.x);
            push_f32(&mut bytes, pos.y);
            push_f32(&mut bytes, pos.z);

            for i in 0..hrir_len {
                push_f32(&mut bytes, hrir(i));
            }
            for i in 0..hrir_len {
                push_f32(&mut bytes, hrir(i));
            }
        }

        bytes
    }

    pub fn broadband_hrir_sample(i: usize) -> f32 {
        let t = i as f32;
        let decay = (-t / 96.0).exp();
        decay
            * ((0.17 * t).sin() * 0.45
                + (0.47 * t).sin() * 0.28
                + (1.73 * t).sin() * 0.18
                + if i == 0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::*;
    use super::*;

    #[test]
    fn resampling_updates_reported_hrir_length() {
        let bytes = test_hrir_sphere(44_100, 512);

        let original = HrirSphere::new(&bytes[..], 44_100).unwrap();
        let upsampled = HrirSphere::new(&bytes[..], 96_000).unwrap();
        let downsampled = HrirSphere::new(&bytes[..], 27_000).unwrap();

        assert_eq!(original.len(), 512);
        assert_eq!(original.points()[0].left_hrir().len(), original.len());

        assert!(upsampled.len() > original.len());
        assert_eq!(upsampled.points()[0].left_hrir().len(), upsampled.len());
        assert_eq!(upsampled.points()[0].right_hrir().len(), upsampled.len());

        assert!(downsampled.len() < original.len());
        assert_eq!(downsampled.points()[0].left_hrir().len(), downsampled.len());
        assert_eq!(
            downsampled.points()[0].right_hrir().len(),
            downsampled.len()
        );
    }

    #[test]
    fn near_identity_resampling_does_not_drop_delayed_hrir_tail() {
        for impulse_index in [0, 1, 32, 128, 256, 511] {
            let bytes = tetrahedron_hrir_sphere_bytes_with(44_100, 512, |i| {
                if i == impulse_index {
                    1.0
                } else {
                    0.0
                }
            });
            let exact_sphere = HrirSphere::new(&bytes[..], 44_100).unwrap();
            let near_sphere = HrirSphere::new(&bytes[..], 44_101).unwrap();
            assert!(near_sphere.points()[0].left_hrir().len() >= exact_sphere.len());

            let energy = |s: &HrirSphere| s.points()[0].left_hrir().iter().map(|v| v * v).sum();
            let (exact, near): (f32, f32) = (energy(&exact_sphere), energy(&near_sphere));
            assert!(
                near > exact * 0.9,
                "delayed tap {} was dropped",
                impulse_index
            );
        }
    }

    #[test]
    fn exact_sample_rate_uses_same_filter_path_as_near_identity() {
        let bytes = tetrahedron_hrir_sphere_bytes_with(44_100, 512, broadband_hrir_sample);
        let exact = HrirSphere::new(&bytes[..], 44_100).unwrap();
        let near = HrirSphere::new(&bytes[..], 44_101).unwrap();

        let exact_sum = exact.points()[0].left_hrir().iter().sum::<f32>();
        let near_sum = near.points()[0].left_hrir().iter().sum::<f32>();

        assert!((exact_sum - near_sum).abs() < 0.02);
    }
}
