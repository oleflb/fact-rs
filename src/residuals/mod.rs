//! Struct & traits for implementing residuals
//!
//! Residuals are the main building block of a
//! [factor](crate::containers::Factor).
//!
//! # Examples
//! Here we make a custom residual for a z-position measurement.
//! Residuals define one associated input pack implementing [`VarPack`]. For a
//! unary residual this can be a single variable type, and for multi-variable
//! residuals this can be a tuple of variable types.
//!
//! `Differ` is the object that computes our auto-differentation. Out of the box
//! factrs comes with [ForwardProp](factrs::linalg::ForwardProp) and
//! [NumericalDiff](factrs::linalg::NumericalDiff). We recommend
//! [ForwardProp](factrs::linalg::ForwardProp) as it should be faster and more
//! accurate.
//!
//! Finally, the residual is defined through a single function that is generic
//! over the datatype. That's it! factrs handles the rest for you.
//!
//! ```
//! use std::fmt;
//!
//! use factrs::{
//!     dtype,
//!     linalg::{Const, ForwardProp, Numeric, VectorX},
//!     residuals,
//!     variables::SE3,
//! };
//!
//! #[derive(Debug, Clone)]
//! # #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
//! struct ZResidual {
//!     value: dtype,
//! }
//!
//! #[factrs::mark]
//! impl residuals::Residual for ZResidual {
//!     type Input = SE3;
//!     type Differ = ForwardProp<Const<6>>;
//!
//!     fn residual<T: Numeric>(&self, x1: SE3<T>) -> VectorX<T> {
//!         VectorX::from_element(1, T::from(self.value) - x1.xyz().z)
//!     }
//! }
//! ```
mod traits;
mod var_pack;
#[cfg(feature = "serde")]
pub use traits::tag_residual;
pub use traits::{DiffPack, DynResidual, ErasedResidual, FixedOutputDim, Residual};
pub use var_pack::{DynVarPack, FactorInput, KeyPack, ResidualError, VarPack};

mod prior;
pub use prior::PriorResidual;

mod between;
pub use between::BetweenResidual;

pub mod imu_preint;
pub use imu_preint::{Accel, Gravity, Gyro, ImuCovariance, ImuPreintegrator};
