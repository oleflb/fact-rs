use crate::{
    dtype,
    linalg::{Diff, DiffInput, DiffResult, MatrixX, VectorX},
    variables::VariableDtype,
};

/// Numerical differentiator.
///
/// It operates on functions with regular dtype inputs and outputs. The generic
/// parameter `PWR` specifies the finite-difference step size as `10^-PWR`.
///
/// ```
/// use factrs::{
///     linalg::{Diff, DiffResult, NumericalDiff, VectorX, vectorx},
///     traits::*,
///     variables::SO2,
/// };
///
/// fn f((x, y): (SO2, SO2)) -> VectorX {
///     x.ominus(&y)
/// }
///
/// let x = SO2::from_theta(2.0);
/// let y = SO2::from_theta(1.0);
///
/// let DiffResult { value, diff } = NumericalDiff::<6>::jacobian(f, &(x, y));
/// assert_eq!(value, vectorx![1.0]);
/// assert_eq!(diff.ncols(), 2);
/// ```
pub struct NumericalDiff<const PWR: i32 = 6>;

impl<I, const PWR: i32> Diff<I> for NumericalDiff<PWR>
where
    I: DiffInput,
{
    type T = dtype;

    fn jacobian<F>(f: F, input: &I) -> DiffResult<VectorX, MatrixX>
    where
        F: Fn(I::Packed<Self::T>) -> VectorX<Self::T>,
    {
        let eps = dtype::powi(10.0, -PWR);
        let value = f(input.pack());
        let mut jac = MatrixX::zeros(value.len(), input.dim());

        for col in 0..input.dim() {
            let plus = input.perturb(col, eps);
            let minus = input.perturb(col, -eps);
            let delta = (f(plus.pack()) - f(minus.pack())) / (2.0 * eps);
            jac.column_mut(col).copy_from(&delta);
        }

        DiffResult { value, diff: jac }
    }
}

impl<const PWR: i32> NumericalDiff<PWR> {
    pub fn jacobian_variable<I, VOut, F>(f: F, input: &I) -> DiffResult<VOut, MatrixX>
    where
        I: DiffInput,
        VOut: VariableDtype,
        F: Fn(I::Packed<dtype>) -> VOut,
    {
        let eps = dtype::powi(10.0, -PWR);
        let value = f(input.pack());
        let mut jac = MatrixX::zeros(VOut::DIM, input.dim());

        for col in 0..input.dim() {
            let plus = input.perturb(col, eps);
            let minus = input.perturb(col, -eps);
            let delta = f(plus.pack()).ominus(&f(minus.pack())) / (2.0 * eps);
            jac.column_mut(col).copy_from(&delta);
        }

        DiffResult { value, diff: jac }
    }
}
