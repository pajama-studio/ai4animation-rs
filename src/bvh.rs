//! Minimal, dependency-free BVH ingestion for offline animation datasets.

use std::fmt;
use std::io::BufRead;

type Mat3 = [[f32; 3]; 3];

const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvhChannel {
    XPosition,
    YPosition,
    ZPosition,
    XRotation,
    YRotation,
    ZRotation,
}

impl BvhChannel {
    fn parse(token: &str) -> Option<Self> {
        match token {
            "Xposition" => Some(Self::XPosition),
            "Yposition" => Some(Self::YPosition),
            "Zposition" => Some(Self::ZPosition),
            "Xrotation" => Some(Self::XRotation),
            "Yrotation" => Some(Self::YRotation),
            "Zrotation" => Some(Self::ZRotation),
            _ => None,
        }
    }

    const fn position_axis(self) -> Option<usize> {
        match self {
            Self::XPosition => Some(0),
            Self::YPosition => Some(1),
            Self::ZPosition => Some(2),
            Self::XRotation | Self::YRotation | Self::ZRotation => None,
        }
    }

    const fn rotation_axis(self) -> Option<usize> {
        match self {
            Self::XRotation => Some(0),
            Self::YRotation => Some(1),
            Self::ZRotation => Some(2),
            Self::XPosition | Self::YPosition | Self::ZPosition => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BvhJoint {
    pub name: String,
    pub parent: Option<usize>,
    pub offset: [f32; 3],
    pub channels: Vec<BvhChannel>,
    channel_start: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvhTranslationMode {
    /// Standard BVH: animated translation is added to the joint's static offset.
    OffsetPlusChannels,
    /// Some DCC exporters repeat the offset in translation channels. In that
    /// dialect the animated channel value replaces, rather than adds to, OFFSET.
    ChannelsReplaceOffset,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BvhPose {
    pub positions: Vec<[f32; 3]>,
    pub rotations: Vec<[[f32; 3]; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BvhClip {
    pub joints: Vec<BvhJoint>,
    pub frame_time: f32,
    pub frame_count: usize,
    channel_count: usize,
    frames: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BvhError {
    line: usize,
    message: String,
}

impl BvhError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for BvhError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "BVH line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for BvhError {}

impl BvhClip {
    /// Parses a BVH stream without retaining its textual representation.
    ///
    /// # Errors
    ///
    /// Returns a line-numbered error for malformed hierarchy, channel, frame
    /// metadata, or motion values.
    #[expect(
        clippy::too_many_lines,
        reason = "the streaming state machine keeps hierarchy and motion line numbers in one pass"
    )]
    pub fn from_reader(reader: impl BufRead) -> Result<Self, BvhError> {
        let mut joints = Vec::<BvhJoint>::new();
        let mut stack = Vec::<usize>::new();
        let mut pending_joint = None;
        let mut saw_hierarchy = false;
        let mut saw_motion = false;
        let mut frame_count = None;
        let mut frame_time = None;
        let mut frames = Vec::new();
        let mut motion_values_started = false;
        let mut last_line = 0;

        for (line_index, line) in reader.lines().enumerate() {
            let line_number = line_index + 1;
            last_line = line_number;
            let line = line.map_err(|error| BvhError::new(line_number, error.to_string()))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if motion_values_started {
                for token in trimmed.split_whitespace() {
                    let value = token.parse::<f32>().map_err(|error| {
                        BvhError::new(
                            line_number,
                            format!("invalid motion value {token:?}: {error}"),
                        )
                    })?;
                    if !value.is_finite() {
                        return Err(BvhError::new(line_number, "non-finite motion value"));
                    }
                    frames.push(value);
                }
                continue;
            }
            if trimmed == "HIERARCHY" {
                saw_hierarchy = true;
                continue;
            }
            if trimmed == "MOTION" {
                if !saw_hierarchy || !stack.is_empty() || pending_joint.is_some() {
                    return Err(BvhError::new(line_number, "incomplete hierarchy"));
                }
                saw_motion = true;
                continue;
            }
            if saw_motion {
                if let Some(value) = trimmed.strip_prefix("Frames:") {
                    frame_count = Some(parse_usize(value.trim(), line_number, "frame count")?);
                    continue;
                }
                if let Some(value) = trimmed.strip_prefix("Frame Time:") {
                    let parsed = parse_f32(value.trim(), line_number, "frame time")?;
                    if parsed <= 0.0 {
                        return Err(BvhError::new(line_number, "frame time must be positive"));
                    }
                    frame_time = Some(parsed);
                    motion_values_started = true;
                    continue;
                }
                return Err(BvhError::new(
                    line_number,
                    "expected Frames: or Frame Time:",
                ));
            }
            if !saw_hierarchy {
                return Err(BvhError::new(line_number, "expected HIERARCHY"));
            }

            if trimmed == "{" {
                let joint = pending_joint
                    .take()
                    .ok_or_else(|| BvhError::new(line_number, "unexpected opening brace"))?;
                stack.push(joint);
                continue;
            }
            if trimmed == "}" {
                if pending_joint.is_some() {
                    return Err(BvhError::new(line_number, "joint is missing opening brace"));
                }
                stack
                    .pop()
                    .ok_or_else(|| BvhError::new(line_number, "unexpected closing brace"))?;
                continue;
            }
            if let Some(name) = trimmed
                .strip_prefix("ROOT ")
                .or_else(|| trimmed.strip_prefix("JOINT "))
            {
                if pending_joint.is_some() {
                    return Err(BvhError::new(line_number, "joint is missing opening brace"));
                }
                let parent = stack.last().copied();
                if trimmed.starts_with("ROOT ") && (!joints.is_empty() || parent.is_some()) {
                    return Err(BvhError::new(line_number, "ROOT must be the first joint"));
                }
                let index = joints.len();
                joints.push(BvhJoint {
                    name: name.trim().to_owned(),
                    parent,
                    offset: [0.0; 3],
                    channels: Vec::new(),
                    channel_start: 0,
                });
                pending_joint = Some(index);
                continue;
            }
            if trimmed == "End Site" {
                let parent = stack
                    .last()
                    .copied()
                    .ok_or_else(|| BvhError::new(line_number, "End Site has no parent"))?;
                let index = joints.len();
                joints.push(BvhJoint {
                    name: format!("{} End Site", joints[parent].name),
                    parent: Some(parent),
                    offset: [0.0; 3],
                    channels: Vec::new(),
                    channel_start: 0,
                });
                pending_joint = Some(index);
                continue;
            }
            let joint_index = stack
                .last()
                .copied()
                .ok_or_else(|| BvhError::new(line_number, "joint property outside braces"))?;
            if let Some(values) = trimmed.strip_prefix("OFFSET ") {
                let values = values.split_whitespace().collect::<Vec<_>>();
                if values.len() != 3 {
                    return Err(BvhError::new(line_number, "OFFSET requires three values"));
                }
                joints[joint_index].offset = [
                    parse_f32(values[0], line_number, "X offset")?,
                    parse_f32(values[1], line_number, "Y offset")?,
                    parse_f32(values[2], line_number, "Z offset")?,
                ];
                continue;
            }
            if let Some(values) = trimmed.strip_prefix("CHANNELS ") {
                let values = values.split_whitespace().collect::<Vec<_>>();
                let count = values
                    .first()
                    .ok_or_else(|| BvhError::new(line_number, "CHANNELS requires a count"))
                    .and_then(|value| parse_usize(value, line_number, "channel count"))?;
                if values.len() != count + 1 {
                    return Err(BvhError::new(
                        line_number,
                        format!(
                            "CHANNELS declares {count} values but provides {}",
                            values.len() - 1
                        ),
                    ));
                }
                joints[joint_index].channels = values[1..]
                    .iter()
                    .map(|value| {
                        BvhChannel::parse(value).ok_or_else(|| {
                            BvhError::new(line_number, format!("unknown channel {value:?}"))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                continue;
            }
            return Err(BvhError::new(
                line_number,
                format!("unrecognized hierarchy statement {trimmed:?}"),
            ));
        }

        let frame_count = frame_count.ok_or_else(|| BvhError::new(last_line, "missing Frames:"))?;
        let frame_time =
            frame_time.ok_or_else(|| BvhError::new(last_line, "missing Frame Time:"))?;
        if joints.is_empty() {
            return Err(BvhError::new(last_line, "hierarchy contains no joints"));
        }
        let mut channel_count = 0;
        for joint in &mut joints {
            joint.channel_start = channel_count;
            channel_count += joint.channels.len();
        }
        let expected_values = frame_count
            .checked_mul(channel_count)
            .ok_or_else(|| BvhError::new(last_line, "motion value count overflow"))?;
        if frames.len() != expected_values {
            return Err(BvhError::new(
                last_line,
                format!(
                    "expected {expected_values} motion values, found {}",
                    frames.len()
                ),
            ));
        }
        Ok(Self {
            joints,
            frame_time,
            frame_count,
            channel_count,
            frames,
        })
    }

    #[must_use]
    pub fn joint_index(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|joint| joint.name == name)
    }

    /// Evaluates a frame in world space, respecting each joint's declared Euler
    /// channel order.
    ///
    /// # Errors
    ///
    /// Returns an error when `frame_index` is outside the clip.
    pub fn pose(
        &self,
        frame_index: usize,
        translation_mode: BvhTranslationMode,
    ) -> Result<BvhPose, BvhError> {
        if frame_index >= self.frame_count {
            return Err(BvhError::new(
                0,
                format!("frame {frame_index} is outside 0..{}", self.frame_count),
            ));
        }
        let frame_start = frame_index * self.channel_count;
        let frame = &self.frames[frame_start..frame_start + self.channel_count];
        let mut positions = Vec::<[f32; 3]>::with_capacity(self.joints.len());
        let mut rotations = Vec::<Mat3>::with_capacity(self.joints.len());
        for joint in &self.joints {
            let mut animated_translation = [0.0; 3];
            let mut has_animated_translation = false;
            let mut local_rotation = IDENTITY;
            for (offset, channel) in joint.channels.iter().copied().enumerate() {
                let value = frame[joint.channel_start + offset];
                if let Some(axis) = channel.position_axis() {
                    animated_translation[axis] = value;
                    has_animated_translation = true;
                } else if let Some(axis) = channel.rotation_axis() {
                    local_rotation = multiply(local_rotation, axis_rotation(axis, value));
                }
            }
            let local_translation = if has_animated_translation {
                match translation_mode {
                    BvhTranslationMode::OffsetPlusChannels => {
                        add(joint.offset, animated_translation)
                    }
                    BvhTranslationMode::ChannelsReplaceOffset => animated_translation,
                }
            } else {
                joint.offset
            };
            if let Some(parent) = joint.parent {
                positions.push(add(
                    positions[parent],
                    transform(rotations[parent], local_translation),
                ));
                rotations.push(multiply(rotations[parent], local_rotation));
            } else {
                positions.push(local_translation);
                rotations.push(local_rotation);
            }
        }
        Ok(BvhPose {
            positions,
            rotations,
        })
    }
}

fn parse_f32(token: &str, line: usize, label: &str) -> Result<f32, BvhError> {
    let value = token
        .parse::<f32>()
        .map_err(|error| BvhError::new(line, format!("invalid {label}: {error}")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(BvhError::new(line, format!("{label} is not finite")))
    }
}

fn parse_usize(token: &str, line: usize, label: &str) -> Result<usize, BvhError> {
    token
        .parse::<usize>()
        .map_err(|error| BvhError::new(line, format!("invalid {label}: {error}")))
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn transform(matrix: Mat3, vector: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|row| {
        matrix[row][0].mul_add(
            vector[0],
            matrix[row][1].mul_add(vector[1], matrix[row][2] * vector[2]),
        )
    })
}

fn multiply(left: Mat3, right: Mat3) -> Mat3 {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            left[row][0].mul_add(
                right[0][column],
                left[row][1].mul_add(right[1][column], left[row][2] * right[2][column]),
            )
        })
    })
}

fn axis_rotation(axis: usize, degrees: f32) -> Mat3 {
    let (sine, cosine) = degrees.to_radians().sin_cos();
    match axis {
        0 => [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        1 => [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        _ => [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const TWO_JOINT: &str = "HIERARCHY\nROOT Root\n{\nOFFSET 0 0 0\nCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\nJOINT Paw\n{\nOFFSET 1 0 0\nCHANNELS 0\nEnd Site\n{\nOFFSET 0.5 0 0\n}\n}\n}\nMOTION\nFrames: 1\nFrame Time: 0.0333333\n2 3 4 90 0 0\n";

    #[test]
    fn parses_and_evaluates_declared_rotation_order() {
        let clip = BvhClip::from_reader(Cursor::new(TWO_JOINT)).unwrap();
        assert_eq!(clip.frame_count, 1);
        assert_eq!(clip.joints.len(), 3);
        assert_eq!(clip.joint_index("Paw"), Some(1));
        let pose = clip
            .pose(0, BvhTranslationMode::OffsetPlusChannels)
            .unwrap();
        assert!((pose.positions[1][0] - 2.0).abs() < 1.0e-5);
        assert!((pose.positions[1][1] - 4.0).abs() < 1.0e-5);
        assert!((pose.positions[1][2] - 4.0).abs() < 1.0e-5);
        assert!((pose.positions[2][0] - 2.0).abs() < 1.0e-5);
        assert!((pose.positions[2][1] - 4.5).abs() < 1.0e-5);
    }

    #[test]
    fn replacement_translation_supports_redundant_dcc_offsets() {
        let text = TWO_JOINT
            .replace("CHANNELS 0", "CHANNELS 3 Xposition Yposition Zposition")
            .replace("2 3 4 90 0 0", "2 3 4 90 0 0 1 0 0");
        let clip = BvhClip::from_reader(Cursor::new(text)).unwrap();
        let replaced = clip
            .pose(0, BvhTranslationMode::ChannelsReplaceOffset)
            .unwrap();
        let added = clip
            .pose(0, BvhTranslationMode::OffsetPlusChannels)
            .unwrap();
        assert!((replaced.positions[1][1] - 4.0).abs() < 1.0e-5);
        assert!((added.positions[1][1] - 5.0).abs() < 1.0e-5);
    }

    #[test]
    fn rejects_wrong_motion_value_count() {
        let text = TWO_JOINT.replace("2 3 4 90 0 0", "2 3 4");
        let error = BvhClip::from_reader(Cursor::new(text)).unwrap_err();
        assert!(error.to_string().contains("expected 6 motion values"));
    }
}
