//! Noise model representations
//!
//! Represent Gaussian noise models in a factor graph, specifically used when
//! constructing a [factor](crate::containers::Factor).

use std::fmt::Debug;

use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::DynClone;
use nalgebra::Dim;

use crate::linalg::{MatrixX, VectorX};

/// The trait for a noise model.
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait NoiseModel: Debug + DynClone + Downcast + Send {
    /// The dimension of the noise model
    type Dim: Dim
    where
        Self: Sized;

    fn dim(&self) -> usize;

    /// Whiten a vector
    fn whiten_vec(&self, v: VectorX) -> VectorX;

    /// Whiten a matrix
    fn whiten_mat(&self, m: MatrixX) -> MatrixX;
}

dyn_clone::clone_trait_object!(NoiseModel);
impl_downcast!(NoiseModel);

#[cfg(feature = "serde")]
pub use register_noisemodel as tag_noise;

mod gaussian;
pub use gaussian::{GaussianNoise, GaussianNoiseDyn};

mod unit;
pub use unit::{UnitNoise, UnitNoiseDyn};
