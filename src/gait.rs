#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leg {
    FrontLeft,
    FrontRight,
    RearLeft,
    RearRight,
}

impl Leg {
    const fn index(self) -> usize {
        match self {
            Self::FrontLeft => 0,
            Self::FrontRight => 1,
            Self::RearLeft => 2,
            Self::RearRight => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GaitKind {
    Walk,
    Trot,
    Pace,
    Canter,
}

impl GaitKind {
    const fn phase_offsets(self) -> [f32; 4] {
        match self {
            Self::Walk => [0.0, 0.5, 0.75, 0.25],
            Self::Trot => [0.0, 0.5, 0.5, 0.0],
            Self::Pace => [0.0, 0.5, 0.0, 0.5],
            Self::Canter => [0.0, 0.12, 0.58, 0.72],
        }
    }

    const fn stance_fraction(self) -> f32 {
        match self {
            Self::Walk => 0.68,
            Self::Trot => 0.56,
            Self::Pace => 0.54,
            Self::Canter => 0.43,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaitSample {
    pub phase: f32,
    pub contact: f32,
    pub swing: f32,
    pub lift: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadrupedGait {
    phase: f32,
    kind: GaitKind,
    stride_length: f32,
}

impl QuadrupedGait {
    #[must_use]
    pub fn new(kind: GaitKind, stride_length: f32) -> Self {
        Self {
            phase: 0.0,
            kind,
            stride_length: stride_length.max(0.05),
        }
    }

    pub fn update(&mut self, delta_seconds: f32, speed: f32) {
        let cycles_per_second = speed.max(0.0) / self.stride_length;
        self.phase = (self.phase + delta_seconds.max(0.0) * cycles_per_second).rem_euclid(1.0);
    }

    pub fn set_kind(&mut self, kind: GaitKind) {
        self.kind = kind;
    }

    #[must_use]
    pub fn sample(self, leg: Leg) -> GaitSample {
        let phase = (self.phase + self.kind.phase_offsets()[leg.index()]).rem_euclid(1.0);
        let stance = self.kind.stance_fraction();
        let edge = 0.08_f32.min(stance * 0.25);
        let contact = if phase < stance {
            smoothstep(0.0, edge, phase) * (1.0 - smoothstep(stance - edge, stance, phase))
        } else {
            0.0
        };
        let swing = if phase >= stance {
            (phase - stance) / (1.0 - stance)
        } else {
            0.0
        };
        let lift = (std::f32::consts::PI * swing).sin().max(0.0);
        GaitSample {
            phase,
            contact,
            swing,
            lift,
        }
    }
}

fn smoothstep(minimum: f32, maximum: f32, value: f32) -> f32 {
    let amount = ((value - minimum) / (maximum - minimum).max(1.0e-6)).clamp(0.0, 1.0);
    amount * amount * (3.0 - 2.0 * amount)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trot_pairs_diagonal_legs() {
        let gait = QuadrupedGait::new(GaitKind::Trot, 1.0);
        assert_eq!(gait.sample(Leg::FrontLeft), gait.sample(Leg::RearRight));
        assert_eq!(gait.sample(Leg::FrontRight), gait.sample(Leg::RearLeft));
        assert_ne!(gait.sample(Leg::FrontLeft), gait.sample(Leg::FrontRight));
    }

    #[test]
    fn gait_update_wraps_deterministically() {
        let mut gait = QuadrupedGait::new(GaitKind::Walk, 2.0);
        gait.update(1.0, 2.5);
        assert!((gait.sample(Leg::FrontLeft).phase - 0.25).abs() < 1.0e-6);
    }
}
