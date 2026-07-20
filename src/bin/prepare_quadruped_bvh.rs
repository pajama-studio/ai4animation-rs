#![expect(
    clippy::cast_precision_loss,
    reason = "bounded frame counters are converted to seconds and normalized ages"
)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a validated positive finite sampling ratio becomes a small stride"
)]

use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

use ai4animation_rs::bvh::{BvhClip, BvhPose, BvhTranslationMode};

const TARGET_HZ: f32 = 30.0;
const EXTERNAL_GROUP: usize = 8;
const MIN_STANCE_SECONDS: f32 = 0.10;
const MIN_SWING_SECONDS: f32 = 0.065;

#[derive(Clone, Copy)]
struct FootSpec {
    contact_candidates: &'static [&'static str],
    upper: &'static str,
    middle: &'static str,
}

const FEET: [FootSpec; 4] = [
    FootSpec {
        contact_candidates: &["b__LeftFinger"],
        upper: "b_LeftArm",
        middle: "b_LeftForeArm",
    },
    FootSpec {
        contact_candidates: &["b_RightFinger"],
        upper: "b_RightArm",
        middle: "b_RightForeArm",
    },
    FootSpec {
        contact_candidates: &["b_LeftToe002"],
        upper: "b_LeftLegUpper",
        middle: "b_LeftLeg",
    },
    FootSpec {
        contact_candidates: &["b_RightToe002"],
        upper: "b_RightLegUpper",
        middle: "b_RightLeg",
    },
];

#[derive(Clone, Copy, Default)]
struct FootFrame {
    world: [f32; 3],
    local_vertical: f32,
    local_forward: f32,
    local_lateral: f32,
}

#[derive(Clone, Copy)]
struct SampleFrame {
    feet: [FootFrame; 4],
}

#[derive(Clone, Copy, Default)]
struct ContactState {
    contact: bool,
    stance_frames: usize,
    swing_frames: usize,
}

#[derive(Clone, Copy, Default)]
struct Counts {
    rows: usize,
    contacts: usize,
    clips: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args_os().skip(1).collect::<Vec<_>>();
    let [input, output] = arguments.as_slice() else {
        return Err("usage: prepare-quadruped-bvh INPUT_DIRECTORY OUTPUT.tsv".into());
    };
    let input = PathBuf::from(input);
    let output = PathBuf::from(output);
    let mut clips = fs::read_dir(&input)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("bvh"))
        })
        .collect::<Vec<_>>();
    clips.sort();
    if clips.is_empty() {
        return Err(format!("no .bvh files found in {}", input.display()).into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(&output)?);
    writeln!(
        writer,
        "split\tgroup\tclip\theight_over_contact_band\tvertical_motion\tforward_motion\tlateral_motion\tworld_horizontal_motion\tworld_vertical_motion\twas_locked\tstance_age\tswing_age\tcontact"
    )?;
    let mut train = Counts::default();
    let mut test = Counts::default();
    for path in clips {
        let file = File::open(&path)?;
        let clip = BvhClip::from_reader(BufReader::new(file))?;
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or("BVH clip name is not UTF-8")?;
        let split = if stable_hash(name).is_multiple_of(5) {
            "test"
        } else {
            "train"
        };
        let counts = write_clip(&mut writer, name, split, &clip)?;
        let aggregate = if split == "test" {
            &mut test
        } else {
            &mut train
        };
        aggregate.rows += counts.rows;
        aggregate.contacts += counts.contacts;
        aggregate.clips += 1;
        eprintln!(
            "clip={name} split={split} frames={} samples={} contacts={}",
            clip.frame_count, counts.rows, counts.contacts
        );
    }
    writer.flush()?;
    if train.clips == 0 || test.clips == 0 || train.contacts == 0 || test.contacts == 0 {
        return Err(
            "whole-clip split did not produce non-empty contact train and test sets".into(),
        );
    }
    println!(
        "wrote={} train_clips={} train_rows={} train_contact_ratio={:.4} test_clips={} test_rows={} test_contact_ratio={:.4}",
        output.display(),
        train.clips,
        train.rows,
        train.contacts as f32 / train.rows as f32,
        test.clips,
        test.rows,
        test.contacts as f32 / test.rows as f32,
    );
    Ok(())
}

fn write_clip(
    writer: &mut impl Write,
    name: &str,
    split: &str,
    clip: &BvhClip,
) -> Result<Counts, Box<dyn std::error::Error>> {
    let hip = required_joint(clip, "b_Hips")?;
    let spine = required_joint(clip, "b_Spine3")?;
    let foot_candidates = FEET
        .map(|foot| {
            foot.contact_candidates
                .iter()
                .map(|name| required_joint(clip, name))
                .collect::<Result<Vec<_>, _>>()
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let upper = FEET
        .map(|foot| required_joint(clip, foot.upper))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let middle = FEET
        .map(|foot| required_joint(clip, foot.middle))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let step = (1.0 / (clip.frame_time * TARGET_HZ)).round().max(1.0) as usize;
    let effective_hz = 1.0 / (clip.frame_time * step as f32);
    if (effective_hz - TARGET_HZ).abs() > 0.5 {
        return Err(format!(
            "{name}: cannot downsample {} Hz to {TARGET_HZ} Hz",
            1.0 / clip.frame_time
        )
        .into());
    }
    let indices = (0..clip.frame_count).step_by(step).collect::<Vec<_>>();
    if indices.len() < 3 {
        return Err(format!("{name}: clip is too short").into());
    }
    let first_pose = clip.pose(indices[0], BvhTranslationMode::ChannelsReplaceOffset)?;
    let bands: [f32; 4] = std::array::from_fn(|index| {
        let contact = lowest_candidate(&first_pose, &foot_candidates[index]);
        let reach = distance(
            first_pose.positions[upper[index]],
            first_pose.positions[middle[index]],
        ) + distance(first_pose.positions[middle[index]], contact);
        (reach * 0.09).max(0.001)
    });
    let mut frames = Vec::with_capacity(indices.len());
    for frame_index in indices {
        let pose = clip.pose(frame_index, BvhTranslationMode::ChannelsReplaceOffset)?;
        frames.push(sample_pose(&pose, hip, spine, &foot_candidates));
    }
    let floors: [f32; 4] = std::array::from_fn(|foot| estimate_floor(&frames, foot, bands[foot]));
    let delta_seconds = clip.frame_time * step as f32;
    let mut states = [ContactState::default(); 4];
    let mut counts = Counts::default();
    for frame_index in 1..frames.len() {
        let current = frames[frame_index];
        let previous = frames[frame_index - 1];
        for foot in 0..4 {
            let state = &mut states[foot];
            let was_locked = state.contact;
            let stance_age = state.stance_frames as f32 * delta_seconds / MIN_STANCE_SECONDS;
            let swing_age = state.swing_frames as f32 * delta_seconds / MIN_SWING_SECONDS;
            let band = bands[foot];
            let height = ((current.feet[foot].world[1] - floors[foot]) / band).max(0.0);
            let vertical =
                (current.feet[foot].local_vertical - previous.feet[foot].local_vertical) / band;
            let forward =
                (current.feet[foot].local_forward - previous.feet[foot].local_forward) / band;
            let lateral =
                (current.feet[foot].local_lateral - previous.feet[foot].local_lateral) / band;
            let world_x = current.feet[foot].world[0] - previous.feet[foot].world[0];
            let world_z = current.feet[foot].world[2] - previous.feet[foot].world[2];
            let world_horizontal = world_x.hypot(world_z) / band;
            let world_vertical =
                (current.feet[foot].world[1] - previous.feet[foot].world[1]) / band;
            let enter = height < 0.90 && world_vertical.abs() < 0.28 && world_horizontal < 0.24;
            let retain = height < 1.45 && world_vertical.abs() < 0.48 && world_horizontal < 0.38;
            if state.contact {
                state.stance_frames += 1;
                if !retain && stance_age >= 1.0 {
                    state.contact = false;
                    state.stance_frames = 0;
                    state.swing_frames = 1;
                }
            } else {
                state.swing_frames += 1;
                if enter && swing_age >= 1.0 {
                    state.contact = true;
                    state.stance_frames = 1;
                    state.swing_frames = 0;
                }
            }
            writeln!(
                writer,
                "{split}\t{EXTERNAL_GROUP}\t{name}\t{height:.9}\t{vertical:.9}\t{forward:.9}\t{lateral:.9}\t{world_horizontal:.9}\t{world_vertical:.9}\t{}\t{stance_age:.9}\t{swing_age:.9}\t{}",
                u8::from(was_locked),
                u8::from(state.contact),
            )?;
            counts.rows += 1;
            counts.contacts += usize::from(state.contact);
        }
    }
    Ok(counts)
}

fn sample_pose(
    pose: &BvhPose,
    hip: usize,
    spine: usize,
    foot_candidates: &[Vec<usize>],
) -> SampleFrame {
    let hips = pose.positions[hip];
    let spine = pose.positions[spine];
    let mut forward = [spine[0] - hips[0], 0.0, spine[2] - hips[2]];
    let forward_length = forward[0].hypot(forward[2]).max(1.0e-5);
    forward[0] /= forward_length;
    forward[2] /= forward_length;
    let lateral = [forward[2], 0.0, -forward[0]];
    SampleFrame {
        feet: std::array::from_fn(|index| {
            let world = lowest_candidate(pose, &foot_candidates[index]);
            let relative = [world[0] - hips[0], world[1] - hips[1], world[2] - hips[2]];
            FootFrame {
                world,
                local_vertical: relative[1],
                local_forward: dot(relative, forward),
                local_lateral: dot(relative, lateral),
            }
        }),
    }
}

/// Estimates a paw's ground level from its low-and-stable world-space samples.
/// A raw minimum is too sensitive to marker spikes; a plain height percentile is
/// biased by gait duty factor. Stability makes this independent of whether the
/// source clip is a walk, run, idle, turn, or jump.
fn estimate_floor(frames: &[SampleFrame], foot: usize, band: f32) -> f32 {
    let mut all_heights = frames
        .iter()
        .map(|frame| frame.feet[foot].world[1])
        .collect::<Vec<_>>();
    all_heights.sort_by(f32::total_cmp);
    let low_limit = all_heights[(all_heights.len() * 3 / 5).min(all_heights.len() - 1)];
    let mut candidates = (1..frames.len())
        .filter_map(|index| {
            let current = frames[index].feet[foot].world;
            if current[1] > low_limit {
                return None;
            }
            let previous = frames[index - 1].feet[foot].world;
            let speed = distance(current, previous) / band;
            Some((speed, current[1]))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    candidates.truncate((candidates.len() / 3).max(1));
    let mut stable_heights = candidates
        .into_iter()
        .map(|(_, height)| height)
        .collect::<Vec<_>>();
    stable_heights.sort_by(f32::total_cmp);
    stable_heights[stable_heights.len() / 2]
}

fn lowest_candidate(pose: &BvhPose, candidates: &[usize]) -> [f32; 3] {
    candidates
        .iter()
        .map(|&index| pose.positions[index])
        .min_by(|left, right| left[1].total_cmp(&right[1]))
        .unwrap_or([0.0; 3])
}

fn required_joint(clip: &BvhClip, name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    clip.joint_index(name)
        .ok_or_else(|| format!("required joint {name:?} is missing").into())
}

fn distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    (left[0] - right[0])
        .hypot(left[1] - right[1])
        .hypot(left[2] - right[2])
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0].mul_add(right[0], left[1].mul_add(right[1], left[2] * right[2]))
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_hash_is_stable_and_not_frame_based() {
        assert_eq!(stable_hash("dog_quad_walk_001"), 0xebd8_4feb_80ec_163f);
        assert_eq!(
            stable_hash("dog_quad_walk_001"),
            stable_hash("dog_quad_walk_001")
        );
    }
}
