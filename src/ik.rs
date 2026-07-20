use crate::math::{Vec3, add, cross, distance, dot, length, lerp, normalize_or, scale, sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FabrikConfig {
    pub max_iterations: u8,
    pub tolerance: f32,
    pub pole_weight: f32,
}

impl Default for FabrikConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 1.0e-3,
            pole_weight: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FabrikReport {
    pub iterations: u8,
    pub error: f32,
    pub reachable: bool,
}

/// Solves a position chain in place while preserving every supplied segment length.
/// `lengths[i]` is the distance from joint `i - 1` to joint `i`; index zero is ignored.
pub fn solve_fabrik(
    joints: &mut [Vec3],
    lengths: &[f32],
    target: Vec3,
    pole: Option<Vec3>,
    config: FabrikConfig,
) -> Option<FabrikReport> {
    if joints.len() < 2
        || lengths.len() != joints.len()
        || joints.iter().flatten().any(|v| !v.is_finite())
    {
        return None;
    }
    let segment_lengths: Vec<f32> = lengths
        .iter()
        .enumerate()
        .map(|(index, &value)| if index == 0 { 0.0 } else { value })
        .collect();
    if segment_lengths[1..]
        .iter()
        .any(|value| !value.is_finite() || *value <= 1.0e-6)
    {
        return None;
    }
    let root = joints[0];
    let total: f32 = segment_lengths[1..].iter().sum();
    let root_to_target = distance(root, target);
    if root_to_target >= total {
        let direction = normalize_or(sub(target, root), [0.0, -1.0, 0.0]);
        for index in 1..joints.len() {
            joints[index] = add(joints[index - 1], scale(direction, segment_lengths[index]));
        }
        return Some(FabrikReport {
            iterations: 1,
            error: distance(*joints.last()?, target),
            reachable: false,
        });
    }

    let mut iterations = 0;
    let tolerance = config.tolerance.max(1.0e-6);
    for iteration in 0..config.max_iterations.max(1) {
        iterations = iteration + 1;
        *joints.last_mut()? = target;
        for index in (1..joints.len()).rev() {
            let direction = normalize_or(sub(joints[index - 1], joints[index]), [0.0, 1.0, 0.0]);
            joints[index - 1] = add(joints[index], scale(direction, segment_lengths[index]));
        }
        joints[0] = root;
        for index in 1..joints.len() {
            let direction = normalize_or(sub(joints[index], joints[index - 1]), [0.0, -1.0, 0.0]);
            joints[index] = add(joints[index - 1], scale(direction, segment_lengths[index]));
        }
        if let Some(pole) = pole {
            apply_pole(joints, pole, config.pole_weight.clamp(0.0, 1.0));
        }
        if distance(*joints.last()?, target) <= tolerance {
            break;
        }
    }
    Some(FabrikReport {
        iterations,
        error: distance(*joints.last()?, target),
        reachable: true,
    })
}

fn apply_pole(joints: &mut [Vec3], pole: Vec3, weight: f32) {
    if joints.len() <= 2 || weight <= 0.0 {
        return;
    }
    let root = joints[0];
    let axis = normalize_or(
        sub(*joints.last().expect("non-empty chain"), root),
        [0.0, 1.0, 0.0],
    );
    let pole_offset = sub(pole, root);
    let pole_direction = normalize_or(
        sub(pole_offset, scale(axis, dot(pole_offset, axis))),
        [1.0, 0.0, 0.0],
    );
    let interior_end = joints.len() - 1;
    for joint in &mut joints[1..interior_end] {
        let offset = sub(*joint, root);
        let axial = scale(axis, dot(offset, axis));
        let radial_length = length(sub(offset, axial));
        if radial_length > 1.0e-6 {
            let desired = add(root, add(axial, scale(pole_direction, radial_length)));
            *joint = lerp(*joint, desired, weight);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBoneConfig {
    pub upper_length: f32,
    pub lower_length: f32,
    /// Prevents a perfectly straight limb, which has an undefined bend plane.
    pub maximum_extension: f32,
    /// Prevents the knee/elbow folding completely through itself.
    pub minimum_reach: f32,
}

impl TwoBoneConfig {
    #[must_use]
    pub fn new(upper_length: f32, lower_length: f32) -> Self {
        Self {
            upper_length,
            lower_length,
            maximum_extension: 0.985,
            minimum_reach: 0.035,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBoneSolution {
    pub root: Vec3,
    pub joint: Vec3,
    pub end: Vec3,
    pub requested_end: Vec3,
    pub clamped: bool,
}

impl TwoBoneSolution {
    #[must_use]
    pub fn upper_length(self) -> f32 {
        distance(self.root, self.joint)
    }

    #[must_use]
    pub fn lower_length(self) -> f32 {
        distance(self.joint, self.end)
    }
}

/// Analytic fixed-length leg solve with a stable pole direction and extension limits.
#[must_use]
pub fn solve_two_bone(
    root: Vec3,
    requested_end: Vec3,
    pole: Vec3,
    config: TwoBoneConfig,
) -> Option<TwoBoneSolution> {
    let upper = config.upper_length;
    let lower = config.lower_length;
    if !upper.is_finite() || !lower.is_finite() || upper <= 1.0e-6 || lower <= 1.0e-6 {
        return None;
    }
    let target_delta = sub(requested_end, root);
    let raw_distance = length(target_delta);
    let direction = normalize_or(target_delta, [0.0, -1.0, 0.0]);
    let geometric_minimum = (upper - lower).abs() + 1.0e-4;
    let minimum = geometric_minimum.max(config.minimum_reach.max(0.0));
    let maximum = ((upper + lower) * config.maximum_extension.clamp(0.5, 0.9999)).max(minimum);
    let solved_distance = raw_distance.clamp(minimum, maximum);
    let end = add(root, scale(direction, solved_distance));
    let along = (upper.mul_add(upper, -lower * lower) + solved_distance * solved_distance)
        / (2.0 * solved_distance);
    let bend_height = upper.mul_add(upper, -along * along).max(0.0).sqrt();

    let pole_offset = sub(pole, root);
    let mut bend = sub(pole_offset, scale(direction, dot(pole_offset, direction)));
    if length(bend) <= 1.0e-6 {
        bend = cross(
            direction,
            if direction[1].abs() < 0.9 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            },
        );
    }
    bend = normalize_or(bend, [1.0, 0.0, 0.0]);
    let joint = add(root, add(scale(direction, along), scale(bend, bend_height)));
    Some(TwoBoneSolution {
        root,
        joint,
        end,
        requested_end,
        clamped: (raw_distance - solved_distance).abs() > 1.0e-5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() < 1.0e-4, "{left} != {right}");
    }

    #[test]
    fn two_bone_never_stretches_authored_segments() {
        let config = TwoBoneConfig::new(0.7, 0.6);
        let solved = solve_two_bone([0.0; 3], [4.0, 0.0, 0.0], [0.0, 1.0, 0.0], config)
            .expect("valid chain");
        close(solved.upper_length(), 0.7);
        close(solved.lower_length(), 0.6);
        assert!(solved.clamped);
        assert!(distance(solved.root, solved.end) < 1.3);
    }

    #[test]
    fn fabrik_preserves_every_segment() {
        let mut joints = [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -2.0, 0.0]];
        let report = solve_fabrik(
            &mut joints,
            &[0.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
            Some([0.0, -1.0, 1.0]),
            FabrikConfig::default(),
        )
        .expect("valid chain");
        assert!(report.error < 0.01);
        close(distance(joints[0], joints[1]), 1.0);
        close(distance(joints[1], joints[2]), 1.0);
    }
}
