# Quadruped contact model card

## `quadruped-contact-synthetic-v0.a4a`

- Purpose: propose stance/contact probability before deterministic foot-lock IK.
- Inputs: normalized foot height, vertical/forward/lateral motion, previous lock
  state, stance age, and swing age.
- Network: 7 inputs, one 24-unit `tanh` layer, one sigmoid output.
- Training data: 24,000 deterministic synthetic samples generated entirely by
  this repository. No third-party motion, model weights, or animal assets.
- Holdout: 6,000 separately generated deterministic samples.
- Metrics: 98.48% accuracy, 97.44% precision, 97.15% recall, 0.100444 binary
  cross-entropy.
- Artifact SHA-256:
  `80fab46ce07bc20b76516c6531d6419be89a503ddd7485f05d638c73399f78fb`.

This baseline learns the trainer's geometric contact policy. It validates the
data/training/export/Rust-inference path, but it is not evidence that learned
contact already outperforms authored clip timing. Integrations must measure
foot slide and retain hard bone-length, reach, debounce, and fallback rules.
