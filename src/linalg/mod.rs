//! Various helpers for linear algebra structures.
//!
//! Specifically this module contains the following,
//! - re-alias all nalgebra types to use our dtype by default.
//! - re-alias num-dual types for identical reasons
//! - a [MatrixBlock] struct to help with block matrix operations after
//!   linearization
//! - a [Diff] trait to help with numerical and forward-mode differentiation
//! - Forward mode differentiator [ForwardProp]
//! - Numerical differentiator [NumericalDiff]
use crate::dtype;

mod dual;
pub use dual::{DualAllocator, DualScalar, DualVector, Numeric};
// Dual numbers
pub use num_dual::Derivative;

mod nalgebra_wrap;
pub use nalgebra_wrap::*;

// ------------------------- MatrixBlocks ------------------------- //
/// A struct to help with block matrix operations after linearization
///
/// This struct is used to store a matrix and a set of indices that
/// represent the start of each block in the matrix. This is useful
/// when linearizing a factor graph, where the Jacobian is a block
/// matrix with each block corresponding to a different variable.
#[derive(Debug, Clone)]
pub struct MatrixBlock {
    mat: MatrixX,
    idx: Vec<usize>,
}

impl MatrixBlock {
    pub fn new(mat: MatrixX, idx: Vec<usize>) -> Self {
        Self { mat, idx }
    }

    pub fn get_block(&self, idx: usize) -> MatrixViewX<'_> {
        let idx_start = self.idx[idx];
        let idx_end = if idx + 1 < self.idx.len() {
            self.idx[idx + 1]
        } else {
            self.mat.ncols()
        };
        self.mat.columns(idx_start, idx_end - idx_start)
    }

    pub fn mul(&self, idx: usize, x: VectorViewX<'_>) -> VectorX {
        self.get_block(idx) * x
    }

    pub fn mat(&self) -> MatrixViewX<'_> {
        self.mat.as_view()
    }

    pub fn idx(&self) -> &[usize] {
        &self.idx
    }
}

// ------------------------- Derivatives ------------------------- //
use nalgebra::{DimNameAdd, DimNameSum};

use crate::variables::{Variable, VariableDtype};

/// A struct to hold the result of a differentiation operation
#[derive(Debug, Clone)]
pub struct DiffResult<V, G> {
    pub value: V,
    pub diff: G,
}

/// Input value that can be differentiated.
pub trait DiffInput: Clone {
    type Packed<T: Numeric>;

    fn dim(&self) -> usize;

    fn pack(&self) -> Self::Packed<dtype>;

    fn perturb(&self, col: usize, eps: dtype) -> Self;
}

/// Input value that can be seeded with fixed-size dual vectors.
pub trait StaticDiffInput: DiffInput {
    type Dim: DimName;

    fn dual(&self) -> Self::Packed<DualVector<Self::Dim>>
    where
        AllocatorBuffer<Self::Dim>: Sync + Send,
        DefaultAllocator: DualAllocator<Self::Dim>,
        DualVector<Self::Dim>: Copy;
}

fn perturb_component<V>(v: &V, col: &mut usize, done: &mut bool, eps: dtype) -> V
where
    V: VariableDtype,
{
    if *done {
        return v.clone();
    }

    let dim = Variable::dim(v);
    if *col < dim {
        let mut delta = VectorX::zeros(dim);
        delta[*col] = eps;
        *done = true;
        v.oplus(delta.as_view())
    } else {
        *col -= dim;
        v.clone()
    }
}

fn dual_component<V, N>(v: &V, offset: &mut usize) -> V::Alias<DualVector<N>>
where
    V: VariableDtype,
    N: DimName,
    AllocatorBuffer<N>: Sync + Send,
    DefaultAllocator: DualAllocator<N>,
    DualVector<N>: Copy,
{
    let out = v.dual::<N>(*offset);
    *offset += Variable::dim(v);
    out
}

impl<V> DiffInput for V
where
    V: VariableDtype,
{
    type Packed<T: Numeric> = V::Alias<T>;

    fn dim(&self) -> usize {
        Variable::dim(self)
    }

    fn pack(&self) -> Self::Packed<dtype> {
        self.clone()
    }

    fn perturb(&self, col: usize, eps: dtype) -> Self {
        let mut delta = VectorX::zeros(Variable::dim(self));
        delta[col] = eps;
        self.oplus(delta.as_view())
    }
}

impl<V> StaticDiffInput for V
where
    V: VariableDtype,
    AllocatorBuffer<V::Dim>: Sync + Send,
    DefaultAllocator: DualAllocator<V::Dim>,
    DualVector<V::Dim>: Copy,
{
    type Dim = V::Dim;

    fn dual(&self) -> Self::Packed<DualVector<Self::Dim>> {
        self.dual::<Self::Dim>(0)
    }
}

macro_rules! impl_tuple_diff_input {
    ($($var:ident: $name:ident),+ $(,)?) => {
        impl<$($var),+> DiffInput for ($($var,)+)
        where
            $($var: VariableDtype,)+
        {
            type Packed<T: Numeric> = ($($var::Alias<T>,)+);

            fn dim(&self) -> usize {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                0 $(+ Variable::dim($name))+
            }

            fn pack(&self) -> Self::Packed<dtype> {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                ($($name.clone(),)+)
            }

            fn perturb(&self, col: usize, eps: dtype) -> Self {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                let mut col = col;
                let mut done = false;
                let out = ($(
                    perturb_component($name, &mut col, &mut done, eps),
                )+);
                assert!(done, "perturbation column out of bounds");
                out
            }
        }
    };
}

impl_tuple_diff_input!(V1: v1, V2: v2);
impl_tuple_diff_input!(V1: v1, V2: v2, V3: v3);
impl_tuple_diff_input!(V1: v1, V2: v2, V3: v3, V4: v4);
impl_tuple_diff_input!(V1: v1, V2: v2, V3: v3, V4: v4, V5: v5);
impl_tuple_diff_input!(V1: v1, V2: v2, V3: v3, V4: v4, V5: v5, V6: v6);

macro_rules! impl_tuple_static_diff_input {
    (($($var:ident: $name:ident),+), $dim:ty, [$($bound:tt)*]) => {
        impl<$($var),+> StaticDiffInput for ($($var,)+)
        where
            $($var: VariableDtype,)+
            $($bound)*
            AllocatorBuffer<$dim>: Sync + Send,
            DefaultAllocator: DualAllocator<$dim>,
            DualVector<$dim>: Copy,
        {
            type Dim = $dim;

            fn dual(&self) -> Self::Packed<DualVector<Self::Dim>> {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                let mut offset = 0;
                ($(
                    dual_component::<$var, Self::Dim>($name, &mut offset),
                )+)
            }
        }
    };
}

impl_tuple_static_diff_input!(
    (V1: v1, V2: v2),
    DimNameSum<V1::Dim, V2::Dim>,
    [V1::Dim: DimNameAdd<V2::Dim>,]
);
impl_tuple_static_diff_input!(
    (V1: v1, V2: v2, V3: v3),
    DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>,
    [V1::Dim: DimNameAdd<V2::Dim>, DimNameSum<V1::Dim, V2::Dim>: DimNameAdd<V3::Dim>,]
);
impl_tuple_static_diff_input!(
    (V1: v1, V2: v2, V3: v3, V4: v4),
    DimNameSum<DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>, V4::Dim>,
    [V1::Dim: DimNameAdd<V2::Dim>, DimNameSum<V1::Dim, V2::Dim>: DimNameAdd<V3::Dim>, DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>: DimNameAdd<V4::Dim>,]
);
impl_tuple_static_diff_input!(
    (V1: v1, V2: v2, V3: v3, V4: v4, V5: v5),
    DimNameSum<DimNameSum<DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>, V4::Dim>, V5::Dim>,
    [V1::Dim: DimNameAdd<V2::Dim>, DimNameSum<V1::Dim, V2::Dim>: DimNameAdd<V3::Dim>, DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>: DimNameAdd<V4::Dim>, DimNameSum<DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>, V4::Dim>: DimNameAdd<V5::Dim>,]
);
impl_tuple_static_diff_input!(
    (V1: v1, V2: v2, V3: v3, V4: v4, V5: v5, V6: v6),
    DimNameSum<DimNameSum<DimNameSum<DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>, V4::Dim>, V5::Dim>, V6::Dim>,
    [V1::Dim: DimNameAdd<V2::Dim>, DimNameSum<V1::Dim, V2::Dim>: DimNameAdd<V3::Dim>, DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>: DimNameAdd<V4::Dim>, DimNameSum<DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>, V4::Dim>: DimNameAdd<V5::Dim>, DimNameSum<DimNameSum<DimNameSum<DimNameSum<V1::Dim, V2::Dim>, V3::Dim>, V4::Dim>, V5::Dim>: DimNameAdd<V6::Dim>,]
);

/// A trait to abstract over different differentiation methods
///
/// Specifically, this trait works for single-variable or tuple-pack functions
/// with scalar output (gradient) or vector output (Jacobian).
///
/// This trait is implemented for both numerical and forward-mode in
/// [NumericalDiff] and [ForwardProp], respectively. Where possible, we
/// recommend [ForwardProp] which functions using dual numbers.
pub trait Diff<I: DiffInput> {
    /// The dtype of the variables
    type T: Numeric;

    fn jacobian<F>(f: F, input: &I) -> DiffResult<VectorX, MatrixX>
    where
        F: Fn(I::Packed<Self::T>) -> VectorX<Self::T>;

    fn gradient<F>(f: F, input: &I) -> DiffResult<dtype, VectorX>
    where
        F: Fn(I::Packed<Self::T>) -> Self::T,
    {
        let f_wrapped = |input| VectorX::from_element(1, f(input));
        let DiffResult { value, diff } = Self::jacobian(f_wrapped, input);
        let diff = VectorX::from_iterator(diff.len(), diff.iter().cloned());
        DiffResult {
            value: value[0],
            diff,
        }
    }
}

/// Compute the derivative of a scalar function using numerical derivatives.
pub fn numerical_derivative<F: Fn(dtype) -> dtype>(
    f: F,
    x: dtype,
    eps: dtype,
) -> DiffResult<dtype, dtype> {
    let r = f(x);
    let d = (f(x + eps) - f(x - eps)) / (2.0 * eps);

    DiffResult { value: r, diff: d }
}

/// Compute the derivative of a scalar function using forward derivatives.
pub fn forward_prop_derivative<F: Fn(DualScalar) -> DualScalar>(
    f: F,
    x: dtype,
) -> DiffResult<dtype, dtype> {
    let xd = x.into();
    let r = f(xd);
    DiffResult {
        value: r.re,
        diff: r.eps,
    }
}

mod numerical_diff;
pub use numerical_diff::NumericalDiff;

mod forward_prop;
pub use forward_prop::ForwardProp;
