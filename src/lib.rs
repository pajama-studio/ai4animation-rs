#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

pub mod bvh;
mod contact;
mod gait;
mod ik;
mod mann;
mod math;

pub use contact::{
    ContactFeatures, ContactModelError, QUADRUPED_CONTACT_HIDDEN_COUNT,
    QUADRUPED_CONTACT_INPUT_COUNT, QuadrupedContactModel,
};
pub use gait::{GaitKind, GaitSample, Leg, QuadrupedGait};
pub use ik::{
    FabrikConfig, FabrikReport, TwoBoneConfig, TwoBoneSolution, solve_fabrik, solve_two_bone,
};
pub use mann::{DenseExpert, MannError, ModeAdaptiveLayer, ModeAdaptiveNetwork};
