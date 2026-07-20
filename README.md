# ai4animation-rs

Clean-room Rust building blocks for responsive, data-driven character animation.
The first release focuses on the runtime pieces a game needs every frame:

- fixed-length FABRIK with optional pole constraints;
- an analytic, joint-limited two-bone solver for animal legs;
- walk, trot, pace, and canter contact scheduling;
- a small mode-adaptive expert network evaluator;
- a versioned, allocation-free learned contact model and deterministic trainer;
- deterministic, allocation-free gait sampling.

```rust
use ai4animation_rs::{TwoBoneConfig, solve_two_bone};

let solved = solve_two_bone(
    [0.0, 1.0, 0.0],
    [0.2, 0.0, 0.5],
    [0.0, 1.0, 1.0],
    TwoBoneConfig::new(0.65, 0.62),
).unwrap();
assert!((solved.upper_length() - 0.65).abs() < 1e-4);
```

## Relationship to `AI4AnimationPy`

This project interoperates with the concepts used by
[AI4AnimationPy](https://github.com/facebookresearch/ai4animationpy) and the
paper *Mode-Adaptive Neural Networks for Quadruped Motion Control*. It is not
an official Meta project.

`AI4AnimationPy` and its bundled model weights/assets are CC BY-NC 4.0. This
repository contains no translated `AI4AnimationPy` source, pretrained weights,
motion data, or character assets. Its implementation was written independently
from public algorithm descriptions and standard animation mathematics, allowing
this crate to remain MIT-licensed and suitable for commercial games.

The `ModeAdaptiveNetwork` evaluator accepts weights supplied by the application.
Applications are responsible for ensuring that their model and training data
licenses permit the intended use.

## Status

`0.3` is a runtime and training experiment, not a `PyTorch` replacement. Planned
follow-ups include a documented neutral weight format, ONNX/safetensors import,
trajectory feature schemas, retargeting, and offline dataset tools.

## Reproducible contact experiment

The first training experiment learns a stance/contact proposal from synthetic,
dimensionless pose signals. It proves the complete train/export/load/infer path
without importing restricted motion data. Rebuild its checked-in weights with:

```sh
cargo run --release --bin train-quadruped-contact
cargo test --all-targets
```

Applications can train the same runtime model from a private, tab-separated
clip corpus without publishing licensed animation samples:

```sh
cargo run --release --bin train-quadruped-contact -- \
  --dataset /path/to/contact-samples.tsv /path/to/game-contact.a4a
```

The trainer rejects a model unless both the aggregate holdout and every dataset
group clear accuracy, precision, and recall gates. A group can represent a
species, motion family, body type, or any application-defined rollout boundary.

The `.a4a` artifact starts with the versioned `A4AQCM01` header followed by its
fixed shape and little-endian `f32` parameters. The model is advisory by design:
production integrations must preserve hard reach, bone-length, debounce, and
fallback constraints around its probability output.

## License

Licensed under MIT.
