#![expect(
    clippy::cast_precision_loss,
    reason = "training sizes and 24-bit PRNG samples are deliberately small and exactly bounded"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the deterministic PRNG deliberately folds its low bits into a small dataset index"
)]

use std::env;
use std::fs;
use std::path::PathBuf;

use ai4animation_rs::{
    ContactFeatures, QUADRUPED_CONTACT_HIDDEN_COUNT, QUADRUPED_CONTACT_INPUT_COUNT,
    QuadrupedContactModel,
};

const TRAIN_SAMPLES: usize = 24_000;
const TEST_SAMPLES: usize = 6_000;
const STEPS: usize = 48_000;
const BATCH: usize = 96;
const LEARNING_RATE: f32 = 0.0025;

#[derive(Clone, Copy)]
struct Sample {
    input: [f32; QUADRUPED_CONTACT_INPUT_COUNT],
    label: f32,
}

#[derive(Clone)]
struct Parameters {
    input_hidden: [f32; QUADRUPED_CONTACT_INPUT_COUNT * QUADRUPED_CONTACT_HIDDEN_COUNT],
    hidden_bias: [f32; QUADRUPED_CONTACT_HIDDEN_COUNT],
    hidden_output: [f32; QUADRUPED_CONTACT_HIDDEN_COUNT],
    output_bias: f32,
}

impl Parameters {
    fn random(random: &mut Random) -> Self {
        let scale =
            (6.0 / (QUADRUPED_CONTACT_INPUT_COUNT + QUADRUPED_CONTACT_HIDDEN_COUNT) as f32).sqrt();
        Self {
            input_hidden: std::array::from_fn(|_| random.signed() * scale),
            hidden_bias: [0.0; QUADRUPED_CONTACT_HIDDEN_COUNT],
            hidden_output: std::array::from_fn(|_| random.signed() * 0.25),
            output_bias: 0.0,
        }
    }

    fn model(&self) -> QuadrupedContactModel {
        QuadrupedContactModel::from_parts(
            self.input_hidden,
            self.hidden_bias,
            self.hidden_output,
            self.output_bias,
        )
    }
}

#[derive(Clone)]
struct Adam {
    mean: Parameters,
    variance: Parameters,
    step: i32,
}

impl Adam {
    fn zero() -> Self {
        let zero = Parameters {
            input_hidden: [0.0; QUADRUPED_CONTACT_INPUT_COUNT * QUADRUPED_CONTACT_HIDDEN_COUNT],
            hidden_bias: [0.0; QUADRUPED_CONTACT_HIDDEN_COUNT],
            hidden_output: [0.0; QUADRUPED_CONTACT_HIDDEN_COUNT],
            output_bias: 0.0,
        };
        Self {
            mean: zero.clone(),
            variance: zero,
            step: 0,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = env::args_os().nth(1).map_or_else(
        || PathBuf::from("models/quadruped-contact-synthetic-v0.a4a"),
        PathBuf::from,
    );
    let mut random = Random::new(0xA14A_2026_0720_0001);
    let train = dataset(TRAIN_SAMPLES, &mut random);
    let test = dataset(TEST_SAMPLES, &mut random);
    let mut parameters = Parameters::random(&mut random);
    let mut adam = Adam::zero();
    for step in 0..STEPS {
        let mut gradient = Adam::zero().mean;
        for _ in 0..BATCH {
            let sample = train[random.index(train.len())];
            accumulate_gradient(&parameters, sample, &mut gradient);
        }
        scale_parameters(&mut gradient, 1.0 / BATCH as f32);
        adam_update(&mut parameters, &gradient, &mut adam);
        if step % 8_000 == 7_999 {
            let metrics = evaluate(&parameters.model(), &test);
            eprintln!(
                "step={} loss={:.5} accuracy={:.4} precision={:.4} recall={:.4}",
                step + 1,
                metrics.loss,
                metrics.accuracy,
                metrics.precision,
                metrics.recall
            );
        }
    }
    let metrics = evaluate(&parameters.model(), &test);
    if metrics.accuracy < 0.965 || metrics.precision < 0.95 || metrics.recall < 0.95 {
        return Err(format!("holdout quality gate failed: {metrics:?}").into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, parameters.model().to_bytes())?;
    println!(
        "wrote={} samples={} holdout={} loss={:.6} accuracy={:.4} precision={:.4} recall={:.4}",
        output.display(),
        train.len(),
        test.len(),
        metrics.loss,
        metrics.accuracy,
        metrics.precision,
        metrics.recall
    );
    Ok(())
}

fn dataset(count: usize, random: &mut Random) -> Vec<Sample> {
    (0..count)
        .map(|_| {
            let features = ContactFeatures {
                height_over_contact_band: random.unit() * 2.0,
                vertical_motion: random.signed(),
                forward_motion: random.signed(),
                lateral_motion: random.signed(),
                was_locked: random.unit() >= 0.5,
                stance_age: random.unit(),
                swing_age: random.unit(),
            };
            Sample {
                input: features.normalized(),
                label: teacher(features),
            }
        })
        .collect()
}

fn teacher(features: ContactFeatures) -> f32 {
    let contact = if features.was_locked {
        let protected_debounce = features.stance_age < 0.12;
        protected_debounce
            || (features.height_over_contact_band < 1.25
                && features.vertical_motion < 0.45
                && !(features.forward_motion > 0.18 && features.height_over_contact_band > 0.62))
    } else {
        features.swing_age >= 0.12
            && features.height_over_contact_band < 0.72
            && features.vertical_motion < 0.35
            && features.forward_motion < 0.08
    };
    if contact { 0.99 } else { 0.01 }
}

fn accumulate_gradient(parameters: &Parameters, sample: Sample, gradient: &mut Parameters) {
    let hidden = std::array::from_fn::<_, QUADRUPED_CONTACT_HIDDEN_COUNT, _>(|row| {
        let offset = row * QUADRUPED_CONTACT_INPUT_COUNT;
        parameters.input_hidden[offset..offset + QUADRUPED_CONTACT_INPUT_COUNT]
            .iter()
            .zip(sample.input)
            .fold(parameters.hidden_bias[row], |sum, (&weight, input)| {
                weight.mul_add(input, sum)
            })
            .tanh()
    });
    let logit = parameters
        .hidden_output
        .iter()
        .zip(hidden)
        .fold(parameters.output_bias, |sum, (&weight, value)| {
            weight.mul_add(value, sum)
        });
    let probability = sigmoid(logit);
    let output_delta = probability - sample.label;
    gradient.output_bias += output_delta;
    for (row, &hidden_value) in hidden.iter().enumerate() {
        gradient.hidden_output[row] += output_delta * hidden_value;
        let hidden_delta =
            output_delta * parameters.hidden_output[row] * (1.0 - hidden_value * hidden_value);
        gradient.hidden_bias[row] += hidden_delta;
        let offset = row * QUADRUPED_CONTACT_INPUT_COUNT;
        for (column, input) in sample.input.into_iter().enumerate() {
            gradient.input_hidden[offset + column] += hidden_delta * input;
        }
    }
}

fn adam_update(parameters: &mut Parameters, gradient: &Parameters, adam: &mut Adam) {
    adam.step += 1;
    update_slice(
        &mut parameters.input_hidden,
        &gradient.input_hidden,
        &mut adam.mean.input_hidden,
        &mut adam.variance.input_hidden,
        adam.step,
    );
    update_slice(
        &mut parameters.hidden_bias,
        &gradient.hidden_bias,
        &mut adam.mean.hidden_bias,
        &mut adam.variance.hidden_bias,
        adam.step,
    );
    update_slice(
        &mut parameters.hidden_output,
        &gradient.hidden_output,
        &mut adam.mean.hidden_output,
        &mut adam.variance.hidden_output,
        adam.step,
    );
    update_value(
        &mut parameters.output_bias,
        gradient.output_bias,
        &mut adam.mean.output_bias,
        &mut adam.variance.output_bias,
        adam.step,
    );
}

fn update_slice(
    parameters: &mut [f32],
    gradients: &[f32],
    means: &mut [f32],
    variances: &mut [f32],
    step: i32,
) {
    for (((parameter, &gradient), mean), variance) in parameters
        .iter_mut()
        .zip(gradients)
        .zip(means)
        .zip(variances)
    {
        update_value(parameter, gradient, mean, variance, step);
    }
}

fn update_value(parameter: &mut f32, gradient: f32, mean: &mut f32, variance: &mut f32, step: i32) {
    *mean = mean.mul_add(0.9, gradient * 0.1);
    *variance = variance.mul_add(0.999, gradient * gradient * 0.001);
    let corrected_mean = *mean / (1.0 - 0.9_f32.powi(step));
    let corrected_variance = *variance / (1.0 - 0.999_f32.powi(step));
    *parameter -= LEARNING_RATE * corrected_mean / (corrected_variance.sqrt() + 1.0e-8);
}

fn scale_parameters(parameters: &mut Parameters, scale: f32) {
    for value in parameters
        .input_hidden
        .iter_mut()
        .chain(&mut parameters.hidden_bias)
        .chain(&mut parameters.hidden_output)
        .chain(std::iter::once(&mut parameters.output_bias))
    {
        *value *= scale;
    }
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    loss: f32,
    accuracy: f32,
    precision: f32,
    recall: f32,
}

fn evaluate(model: &QuadrupedContactModel, samples: &[Sample]) -> Metrics {
    let mut loss = 0.0;
    let (mut correct, mut true_positive, mut false_positive, mut false_negative) = (0, 0, 0, 0);
    for sample in samples {
        let features = ContactFeatures {
            height_over_contact_band: sample.input[0] + 1.0,
            vertical_motion: sample.input[1],
            forward_motion: sample.input[2],
            lateral_motion: sample.input[3],
            was_locked: sample.input[4] > 0.0,
            stance_age: (sample.input[5] + 1.0) * 0.5,
            swing_age: (sample.input[6] + 1.0) * 0.5,
        };
        let probability = model
            .predict_probability(features)
            .clamp(1.0e-6, 1.0 - 1.0e-6);
        loss -= sample.label.mul_add(
            probability.ln(),
            (1.0 - sample.label) * (1.0 - probability).ln(),
        );
        let predicted = probability >= 0.5;
        let expected = sample.label >= 0.5;
        correct += usize::from(predicted == expected);
        true_positive += usize::from(predicted && expected);
        false_positive += usize::from(predicted && !expected);
        false_negative += usize::from(!predicted && expected);
    }
    let ratio = |numerator: usize, denominator: usize| numerator as f32 / denominator.max(1) as f32;
    Metrics {
        loss: loss / samples.len() as f32,
        accuracy: ratio(correct, samples.len()),
        precision: ratio(true_positive, true_positive + false_positive),
        recall: ratio(true_positive, true_positive + false_negative),
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

struct Random(u64);

impl Random {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        let bits = (self.next() >> 40) as u32;
        bits as f32 / 0xFF_FFFF as f32
    }

    fn signed(&mut self) -> f32 {
        self.unit().mul_add(2.0, -1.0)
    }

    fn index(&mut self, length: usize) -> usize {
        self.next() as usize % length
    }
}
