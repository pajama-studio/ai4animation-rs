# Audited external motion sources

The repository contains import code and provenance, not third-party raw motion
or derived training tables. Consumers must retain the attribution required by
each source and independently review whether a license fits their product.

## Lifelike Agility dog BVH corpus

- Source: Lei Han et al., *Lifelike Agility and Play in Quadrupedal Robots using
  Reinforcement Learning and Generative Pre-trained Models*.
- Permanent identifier: <https://doi.org/10.6084/m9.figshare.24968946>
- Publisher record: Springer Nature / Figshare, posted 2024-07-06.
- Declared data license: CC BY 4.0.
- Published file: `raw_bvh_data.zip`, file id `43971117`, 73,545,752 bytes.
- Published and locally verified MD5: `746d592726eb7411f084518bfa1f3791`.
- Contents observed by the importer: 33 BVH clips, 117,636 frames at 120 Hz,
  980.3 seconds (16.34 minutes) of dog motion.
- Motion families: forward/backward walk and run, straight/zig-zag/star paths,
  idle, jump, high jump, hit reaction, and play.

Reproducible download and integrity check:

```sh
curl -L --fail --retry 3 \
  https://ndownloader.figshare.com/files/43971117 \
  -o raw_bvh_data.zip
md5 raw_bvh_data.zip
```

`prepare-quadruped-bvh` downsamples the source to 30 Hz and emits 117,596
per-foot samples. Its deterministic FNV-1a whole-clip split currently produces:

- training: 26 clips, 87,308 samples, 52.78% contact;
- holdout: 7 unseen clips, 30,288 samples, 55.71% contact.

Contact pseudo-labels depend on the paw remaining low and stable in world space,
while clip-space motion is retained as separate model input. This distinction is
important: a foot moving backward through a rig can be planted if that motion
cancels body travel. Per-paw ground height is estimated from low-and-stable
samples rather than a raw minimum, which is sensitive to marker spikes.

In the initial external-only experiment, the 9→24→1 model achieved 99.04%
accuracy, 98.86% precision, and 99.42% recall on seven entirely unseen clips.
When mixed with ROOTWALKER's eight-species game corpus using group-balanced
mini-batches, it achieved 98.07% aggregate holdout accuracy; every individual
game species and the external corpus passed the trainer's quality gates.

## Evaluated but not imported

- RGBD-Dog provides five dogs, solved BVH skeletons, meshes, markers, and video,
  but access is academic/request-based; it is not part of the commercial corpus.
- 3DDogs provides 37 usable dogs and 143 recordings with optical ground truth,
  but its license is non-commercial and forbids redistribution.
- AnimalML3D provides 1,240 skeletal animal motions and text labels, but derives
  from DeformingThings4D and requires a separate license-chain audit.
- QuadFM advertises 11,784 text-labelled quadruped clips under an Apache-2.0 code
  repository, but as of 2026-07-20 its repository still says the dataset will be
  released soon. No unreleased data is assumed available.
