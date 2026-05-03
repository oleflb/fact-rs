use core::fmt;

use nalgebra::{Const, Dyn};

use super::NoiseModel;
use crate::linalg::{MatrixX, VectorX};

/// A unit noise model.
///
/// Represents a noise model that does not modify the input, or equal weighting
/// in a [factor](crate::containers::Factor).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnitNoise<const D: usize>;

#[factrs::mark]
impl<const D: usize> NoiseModel for UnitNoise<D> {
    type Dim = Const<D>;

    fn dim(&self) -> usize {
        D
    }

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        v
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        m
    }
}

impl<const N: usize> fmt::Display for UnitNoise<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A unit noise model.
///
/// Represents a noise model that does not modify the input, or equal weighting
/// in a [factor](crate::containers::Factor).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnitNoiseDyn {
    dimension: usize,
}

impl UnitNoiseDyn {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[factrs::mark]
impl NoiseModel for UnitNoiseDyn {
    type Dim = Dyn;

    fn dim(&self) -> usize {
        self.dimension
    }

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        v
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        m
    }
}

impl fmt::Display for UnitNoiseDyn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
