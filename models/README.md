# Quadruped contact model card

## `quadruped-contact-synthetic-v0.a4a`

- Purpose: propose stance/contact probability before deterministic foot-lock IK.
- Inputs: normalized foot height; clip-space vertical/forward/lateral motion;
  world-space horizontal/vertical motion; previous lock state; stance age; and
  swing age.
- Network: 9 inputs, one 24-unit `tanh` layer, one sigmoid output.
- Training data: 24,000 deterministic synthetic samples generated entirely by
  this repository. No third-party motion, model weights, or animal assets.
- Holdout: 6,000 separately generated deterministic samples.
- Metrics: 97.82% accuracy, 96.32% precision, 99.43% recall, 0.112127 binary
  cross-entropy.
- Artifact SHA-256:
  `fa84e32c5327cbaef5fe9225a9b3579f6b760e75f347b4decf645b0b925a329a`.

This baseline learns the trainer's geometric contact policy. It validates the
data/training/export/Rust-inference path, but it is not evidence that learned
contact already outperforms authored clip timing. Integrations must measure
foot slide and retain hard bone-length, reach, debounce, and fallback rules.
