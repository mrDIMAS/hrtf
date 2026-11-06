//! Benchmarks for the time-domain `HrtfProcessor`. Each iteration processes one 128-frame block,
//! so ns/iter is the per-block cost.
//!
//! To run (nightly only; the cfg keeps `cargo build --all-targets` working on stable):
//! `RUSTFLAGS="--cfg nightly_bench" cargo +nightly bench --bench hrtf`
//! Add `-C target-cpu=x86-64-v3` (or `native`) to RUSTFLAGS to enable AVX2/FMA.
//!
//! To compare with the old FFT implementation, run the same workload on the base commit: see the
//! instructions at the top of `baseline/hrtf_fft.rs`. Keep the two workloads in sync.
#![cfg(nightly_bench)]
#![feature(test)]
extern crate test;

use hrtf::{HrirSphere, HrtfProcessor, HrtfSphere, Vec3};
use test::{black_box, Bencher};

const BLOCK: usize = 128;
const HRIR_BYTES: &[u8] = include_bytes!("IRC_1003_C.bin");

/// Fraction of blocks that carry sound in the "sparse" benches: 3 of every 8 blocks = 37.5%.
const SPARSE_PATTERN: [bool; 8] = [true, false, false, true, false, false, true, false];

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

fn noise_block(rng: &mut Lcg) -> Vec<f32> {
    (0..BLOCK).map(|_| 0.5 * rng.next()).collect()
}

/// A source direction that sweeps around the head, changing every block.
fn direction(block: usize) -> Vec3 {
    let a = block as f32 * 0.07;
    Vec3::new(a.sin(), 0.4 * (a * 0.31).sin(), a.cos())
}

fn make_processor(rate: u32) -> HrtfProcessor {
    let sphere = HrirSphere::new(HRIR_BYTES, rate).unwrap();
    HrtfProcessor::new(HrtfSphere::new(&sphere))
}

/// `active(block_index)` says whether that block carries sound or silence.
fn run(b: &mut Bencher, rate: u32, active: impl Fn(usize) -> bool) {
    let mut processor = make_processor(rate);
    let mut rng = Lcg(1);
    let loud: Vec<Vec<f32>> = (0..8).map(|_| noise_block(&mut rng)).collect();
    let silent = vec![0.0f32; BLOCK];
    let mut output = vec![(0.0f32, 0.0f32); BLOCK];
    let mut block = 0usize;

    // Warm up so the first (reset) block and the lazily-grown buffers are out of the picture.
    for _ in 0..16 {
        processor.process(&loud[0], &mut output, direction(block), 1.0);
        block += 1;
    }

    b.iter(|| {
        let source = if active(block) {
            &loud[block % 8]
        } else {
            &silent
        };
        output.fill((0.0, 0.0));
        processor.process(
            black_box(source),
            black_box(&mut output),
            direction(block),
            1.0,
        );
        block += 1;
        black_box(&output);
    });
}

fn continuous(_: usize) -> bool {
    true
}
fn sparse(block: usize) -> bool {
    SPARSE_PATTERN[block % 8]
}
fn silence(_: usize) -> bool {
    false
}

#[bench]
fn continuous_44100(b: &mut Bencher) {
    run(b, 44100, continuous);
}
#[bench]
fn continuous_48000(b: &mut Bencher) {
    run(b, 48000, continuous);
}
#[bench]
fn sparse_44100(b: &mut Bencher) {
    run(b, 44100, sparse);
}
#[bench]
fn sparse_48000(b: &mut Bencher) {
    run(b, 48000, sparse);
}
#[bench]
fn silence_44100(b: &mut Bencher) {
    run(b, 44100, silence);
}
#[bench]
fn silence_48000(b: &mut Bencher) {
    run(b, 48000, silence);
}
