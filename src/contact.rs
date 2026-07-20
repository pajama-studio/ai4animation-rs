use std::fmt;

pub const QUADRUPED_CONTACT_INPUT_COUNT: usize = 9;
pub const QUADRUPED_CONTACT_HIDDEN_COUNT: usize = 24;
const MODEL_MAGIC: &[u8; 8] = b"A4AQCM02";
const HEADER_BYTES: usize = 14;
const MODEL_FLOAT_COUNT: usize = QUADRUPED_CONTACT_HIDDEN_COUNT * QUADRUPED_CONTACT_INPUT_COUNT
    + QUADRUPED_CONTACT_HIDDEN_COUNT
    + QUADRUPED_CONTACT_HIDDEN_COUNT
    + 1;

/// Pose-derived inputs for the experimental learned stance classifier.
///
/// Every value is dimensionless so one model can be evaluated on differently
/// sized quadrupeds. Distances are normalized by the current leg reach and
/// durations by the controller's stance/swing debounce windows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactFeatures {
    pub height_over_contact_band: f32,
    pub vertical_motion: f32,
    pub forward_motion: f32,
    pub lateral_motion: f32,
    /// World-space horizontal foot displacement per sample, normalized by the
    /// contact band. A planted foot should approach zero even while the body moves.
    pub world_horizontal_motion: f32,
    /// World-space vertical foot displacement per sample, normalized by the
    /// contact band. This captures terrain/body movement missing from clip space.
    pub world_vertical_motion: f32,
    pub was_locked: bool,
    pub stance_age: f32,
    pub swing_age: f32,
}

impl ContactFeatures {
    #[must_use]
    pub fn normalized(self) -> [f32; QUADRUPED_CONTACT_INPUT_COUNT] {
        [
            self.height_over_contact_band.clamp(0.0, 2.0) - 1.0,
            self.vertical_motion.clamp(-1.0, 1.0),
            self.forward_motion.clamp(-1.0, 1.0),
            self.lateral_motion.clamp(-1.0, 1.0),
            self.world_horizontal_motion.clamp(0.0, 2.0) - 1.0,
            self.world_vertical_motion.clamp(-1.0, 1.0),
            if self.was_locked { 1.0 } else { -1.0 },
            self.stance_age.clamp(0.0, 1.0).mul_add(2.0, -1.0),
            self.swing_age.clamp(0.0, 1.0).mul_add(2.0, -1.0),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContactModelError {
    InvalidMagic,
    UnsupportedShape {
        inputs: usize,
        hidden: usize,
        outputs: usize,
    },
    InvalidLength {
        expected: usize,
        actual: usize,
    },
    NonFiniteWeight,
}

impl fmt::Display for ContactModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ContactModelError {}

/// A tiny, allocation-free MLP trained to propose quadruped foot contact.
///
/// This is deliberately an advisory model: games should retain reach, bone
/// length, and minimum stance-time constraints around its probability output.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadrupedContactModel {
    input_hidden: [f32; QUADRUPED_CONTACT_INPUT_COUNT * QUADRUPED_CONTACT_HIDDEN_COUNT],
    hidden_bias: [f32; QUADRUPED_CONTACT_HIDDEN_COUNT],
    hidden_output: [f32; QUADRUPED_CONTACT_HIDDEN_COUNT],
    output_bias: f32,
}

impl QuadrupedContactModel {
    /// Loads the small neutral binary format emitted by the bundled trainer.
    ///
    /// # Errors
    ///
    /// Rejects unknown versions, shapes, lengths, or non-finite parameters.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContactModelError> {
        if bytes.get(..MODEL_MAGIC.len()) != Some(MODEL_MAGIC) {
            return Err(ContactModelError::InvalidMagic);
        }
        if bytes.len() < HEADER_BYTES {
            return Err(ContactModelError::InvalidLength {
                expected: HEADER_BYTES + MODEL_FLOAT_COUNT * size_of::<f32>(),
                actual: bytes.len(),
            });
        }
        let inputs = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        let hidden = usize::from(u16::from_le_bytes([bytes[10], bytes[11]]));
        let outputs = usize::from(u16::from_le_bytes([bytes[12], bytes[13]]));
        if (inputs, hidden, outputs)
            != (
                QUADRUPED_CONTACT_INPUT_COUNT,
                QUADRUPED_CONTACT_HIDDEN_COUNT,
                1,
            )
        {
            return Err(ContactModelError::UnsupportedShape {
                inputs,
                hidden,
                outputs,
            });
        }
        let expected = HEADER_BYTES + MODEL_FLOAT_COUNT * size_of::<f32>();
        if bytes.len() != expected {
            return Err(ContactModelError::InvalidLength {
                expected,
                actual: bytes.len(),
            });
        }
        let mut cursor = HEADER_BYTES;
        let mut next = || {
            let value = f32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]);
            cursor += size_of::<f32>();
            value
        };
        let input_hidden = std::array::from_fn(|_| next());
        let hidden_bias = std::array::from_fn(|_| next());
        let hidden_output = std::array::from_fn(|_| next());
        let output_bias = next();
        if input_hidden
            .iter()
            .chain(&hidden_bias)
            .chain(&hidden_output)
            .chain(std::iter::once(&output_bias))
            .any(|value| !value.is_finite())
        {
            return Err(ContactModelError::NonFiniteWeight);
        }
        Ok(Self {
            input_hidden,
            hidden_bias,
            hidden_output,
            output_bias,
        })
    }

    /// Creates a model from trainer-owned parameter arrays.
    #[must_use]
    pub const fn from_parts(
        input_hidden: [f32; QUADRUPED_CONTACT_INPUT_COUNT * QUADRUPED_CONTACT_HIDDEN_COUNT],
        hidden_bias: [f32; QUADRUPED_CONTACT_HIDDEN_COUNT],
        hidden_output: [f32; QUADRUPED_CONTACT_HIDDEN_COUNT],
        output_bias: f32,
    ) -> Self {
        Self {
            input_hidden,
            hidden_bias,
            hidden_output,
            output_bias,
        }
    }

    /// Loads the reproducible synthetic baseline shipped with the crate.
    ///
    /// # Errors
    ///
    /// The checked-in artifact is validated through the same parser as external
    /// weights, so corrupt package contents are reported rather than trusted.
    pub fn bundled_synthetic_v0() -> Result<Self, ContactModelError> {
        Self::from_bytes(include_bytes!(
            "../models/quadruped-contact-synthetic-v0.a4a"
        ))
    }

    #[must_use]
    pub fn predict_probability(&self, features: ContactFeatures) -> f32 {
        let input = features.normalized();
        let hidden = std::array::from_fn::<_, QUADRUPED_CONTACT_HIDDEN_COUNT, _>(|row| {
            let offset = row * QUADRUPED_CONTACT_INPUT_COUNT;
            let sum = self.input_hidden[offset..offset + QUADRUPED_CONTACT_INPUT_COUNT]
                .iter()
                .zip(input)
                .fold(self.hidden_bias[row], |accumulator, (&weight, value)| {
                    weight.mul_add(value, accumulator)
                });
            sum.tanh()
        });
        let logit = self
            .hidden_output
            .iter()
            .zip(hidden)
            .fold(self.output_bias, |accumulator, (&weight, value)| {
                weight.mul_add(value, accumulator)
            });
        sigmoid(logit)
    }

    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_BYTES + MODEL_FLOAT_COUNT * size_of::<f32>());
        bytes.extend_from_slice(MODEL_MAGIC);
        let input_count = u16::try_from(QUADRUPED_CONTACT_INPUT_COUNT).unwrap_or_default();
        let hidden_count = u16::try_from(QUADRUPED_CONTACT_HIDDEN_COUNT).unwrap_or_default();
        bytes.extend_from_slice(&input_count.to_le_bytes());
        bytes.extend_from_slice(&hidden_count.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        for value in self
            .input_hidden
            .iter()
            .chain(&self.hidden_bias)
            .chain(&self.hidden_output)
            .chain(std::iter::once(&self.output_bias))
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_round_trip_preserves_predictions() {
        let model = QuadrupedContactModel::from_parts(
            [0.01; QUADRUPED_CONTACT_INPUT_COUNT * QUADRUPED_CONTACT_HIDDEN_COUNT],
            [0.02; QUADRUPED_CONTACT_HIDDEN_COUNT],
            [0.03; QUADRUPED_CONTACT_HIDDEN_COUNT],
            -0.04,
        );
        let decoded = QuadrupedContactModel::from_bytes(&model.to_bytes()).unwrap();
        let features = ContactFeatures {
            height_over_contact_band: 0.3,
            vertical_motion: -0.1,
            forward_motion: -0.2,
            lateral_motion: 0.0,
            world_horizontal_motion: 0.05,
            world_vertical_motion: -0.02,
            was_locked: false,
            stance_age: 0.0,
            swing_age: 0.8,
        };
        assert!(
            (model.predict_probability(features) - decoded.predict_probability(features)).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn bundled_model_separates_clear_stance_and_swing() {
        let model = QuadrupedContactModel::bundled_synthetic_v0().unwrap();
        let stance = model.predict_probability(ContactFeatures {
            height_over_contact_band: 0.15,
            vertical_motion: -0.1,
            forward_motion: -0.3,
            lateral_motion: 0.0,
            world_horizontal_motion: 0.02,
            world_vertical_motion: -0.01,
            was_locked: true,
            stance_age: 0.7,
            swing_age: 0.0,
        });
        let swing = model.predict_probability(ContactFeatures {
            height_over_contact_band: 1.7,
            vertical_motion: 0.5,
            forward_motion: 0.5,
            lateral_motion: 0.1,
            world_horizontal_motion: 0.9,
            world_vertical_motion: 0.5,
            was_locked: false,
            stance_age: 0.0,
            swing_age: 0.3,
        });
        assert!(stance > 0.8, "stance probability {stance}");
        assert!(swing < 0.2, "swing probability {swing}");
    }
}
