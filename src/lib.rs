#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

mod gait;
mod ik;
mod mann;
mod math;

pub use gait::{GaitKind, GaitSample, Leg, QuadrupedGait};
pub use ik::{
    FabrikConfig, FabrikReport, TwoBoneConfig, TwoBoneSolution, solve_fabrik, solve_two_bone,
};
pub use mann::{DenseExpert, MannError, ModeAdaptiveLayer, ModeAdaptiveNetwork};
