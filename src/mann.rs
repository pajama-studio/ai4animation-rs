use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseExpert {
    pub input_count: usize,
    pub output_count: usize,
    /// Row-major `[output][input]` weights.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MannError {
    EmptyExperts,
    InvalidExpertShape,
    InputShape { expected: usize, actual: usize },
    GateShape { expected: usize, actual: usize },
    LayerShape { expected: usize, actual: usize },
}

impl fmt::Display for MannError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MannError {}

#[derive(Clone, Debug, PartialEq)]
pub struct ModeAdaptiveLayer {
    experts: Vec<DenseExpert>,
    elu: bool,
}

impl ModeAdaptiveLayer {
    /// Creates a layer whose experts all share one dense shape.
    ///
    /// # Errors
    ///
    /// Returns [`MannError::EmptyExperts`] or [`MannError::InvalidExpertShape`]
    /// when the expert bank cannot be evaluated safely.
    pub fn new(experts: Vec<DenseExpert>, elu: bool) -> Result<Self, MannError> {
        let Some(first) = experts.first() else {
            return Err(MannError::EmptyExperts);
        };
        if first.input_count == 0
            || first.output_count == 0
            || experts.iter().any(|expert| {
                expert.input_count != first.input_count
                    || expert.output_count != first.output_count
                    || expert.weights.len() != expert.input_count * expert.output_count
                    || expert.bias.len() != expert.output_count
                    || expert
                        .weights
                        .iter()
                        .chain(&expert.bias)
                        .any(|value| !value.is_finite())
            })
        {
            return Err(MannError::InvalidExpertShape);
        }
        Ok(Self { experts, elu })
    }

    #[must_use]
    pub fn input_count(&self) -> usize {
        self.experts[0].input_count
    }

    #[must_use]
    pub fn output_count(&self) -> usize {
        self.experts[0].output_count
    }

    /// Blends experts using normalized non-negative gates and evaluates the layer.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the supplied slices do not match the layer.
    pub fn evaluate(
        &self,
        input: &[f32],
        gates: &[f32],
        output: &mut [f32],
    ) -> Result<(), MannError> {
        if input.len() != self.input_count() {
            return Err(MannError::InputShape {
                expected: self.input_count(),
                actual: input.len(),
            });
        }
        if gates.len() != self.experts.len() {
            return Err(MannError::GateShape {
                expected: self.experts.len(),
                actual: gates.len(),
            });
        }
        if output.len() != self.output_count() {
            return Err(MannError::LayerShape {
                expected: self.output_count(),
                actual: output.len(),
            });
        }
        output.fill(0.0);
        let gate_sum: f32 = gates.iter().map(|gate| gate.max(0.0)).sum();
        let normalizer = if gate_sum > 1.0e-8 {
            gate_sum.recip()
        } else {
            0.0
        };
        for (expert, &raw_gate) in self.experts.iter().zip(gates) {
            let gate = raw_gate.max(0.0) * normalizer;
            for (row, value) in output.iter_mut().enumerate() {
                let weights =
                    &expert.weights[row * expert.input_count..(row + 1) * expert.input_count];
                let sum = weights
                    .iter()
                    .zip(input)
                    .fold(expert.bias[row], |accumulator, (&weight, &input)| {
                        weight.mul_add(input, accumulator)
                    });
                *value += gate * sum;
            }
        }
        if self.elu {
            for value in output {
                if *value < 0.0 {
                    *value = value.exp() - 1.0;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModeAdaptiveNetwork {
    layers: Vec<ModeAdaptiveLayer>,
}

impl ModeAdaptiveNetwork {
    /// Creates a network from shape-compatible mode-adaptive layers.
    ///
    /// # Errors
    ///
    /// Returns [`MannError::LayerShape`] when adjacent layers do not connect.
    pub fn new(layers: Vec<ModeAdaptiveLayer>) -> Result<Self, MannError> {
        for pair in layers.windows(2) {
            if pair[0].output_count() != pair[1].input_count() {
                return Err(MannError::LayerShape {
                    expected: pair[0].output_count(),
                    actual: pair[1].input_count(),
                });
            }
        }
        Ok(Self { layers })
    }

    /// Evaluates the complete network with a shared expert gate vector.
    ///
    /// # Errors
    ///
    /// Returns a shape error reported by any layer.
    pub fn evaluate(&self, input: &[f32], gates: &[f32]) -> Result<Vec<f32>, MannError> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            let mut output = vec![0.0; layer.output_count()];
            layer.evaluate(&current, gates, &mut output)?;
            current = output;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert(scale: f32) -> DenseExpert {
        DenseExpert {
            input_count: 2,
            output_count: 1,
            weights: vec![scale, scale],
            bias: vec![0.0],
        }
    }

    #[test]
    fn gates_blend_expert_parameters() {
        let layer = ModeAdaptiveLayer::new(vec![expert(1.0), expert(3.0)], false).unwrap();
        let mut output = [0.0];
        layer
            .evaluate(&[2.0, 1.0], &[0.75, 0.25], &mut output)
            .unwrap();
        assert!((output[0] - 4.5).abs() < 1.0e-6);
    }
}
