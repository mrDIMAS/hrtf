use criterion::{criterion_group, criterion_main, Criterion};
use hrtf::{HrirSphere, HrtfContext, HrtfProcessor, Vec3};

#[derive(Debug)]
struct SineGenerator {
    sample_rate: usize,
    frequency: f32,
    amplitude: f32,
    index: usize,
}

impl SineGenerator {
    pub fn new(sample_rate: usize, frequency: f32) -> Self {
        Self {
            sample_rate,
            frequency,
            amplitude: 0.75,
            index: 0,
        }
    }
}

impl Iterator for SineGenerator {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.amplitude
            * ((2.0 * std::f32::consts::PI * self.index as f32 * self.frequency)
                / self.sample_rate as f32)
                .sin();

        self.index += 1;

        Some(sample)
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let sample_rate = 44100;
    let source_len = 2048 + 1;
    let source = SineGenerator::new(sample_rate, 440.0)
        .take(source_len)
        .collect::<Vec<_>>();

    let sph = HrirSphere::from_file("benches/IRC_1002_C.bin", sample_rate as u32).unwrap();
    let mut processor = HrtfProcessor::new(sph, source_len);
    let mut output = vec![(0.0, 0.0); source_len];
    let mut history = vec![];

    c.bench_function(&format!("process {source_len} samples",), |b| {
        b.iter(|| {
            let context = HrtfContext {
                source: &source,
                output: &mut output,
                new_sample_vector: Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                prev_sample_vector: Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                history: &mut history,
                new_gain: 1.0,
                prev_gain: 1.0,
            };
            processor.process_samples(context);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
