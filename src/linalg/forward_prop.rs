use super::{
    AllocatorBuffer, Diff, MatrixDim, StaticDiffInput,
    dual::{DualAllocator, DualVector},
};
use crate::linalg::{Const, DefaultAllocator, DiffResult, Dyn, MatrixX, VectorDim, VectorX};
use nalgebra::DimName;

/// Forward mode differentiator.
///
/// It seeds a pack of variables with fixed-size dual vectors and computes the
/// Jacobian of a vector-valued function.
///
/// ```
/// use factrs::{
///     linalg::{Diff, DiffResult, ForwardProp, Numeric, VectorX, vectorx},
///     traits::*,
///     variables::SO2,
/// };
///
/// fn f<T: Numeric>((x, y): (SO2<T>, SO2<T>)) -> VectorX<T> {
///     x.ominus(&y)
/// }
///
/// let x = SO2::from_theta(2.0);
/// let y = SO2::from_theta(1.0);
///
/// let DiffResult { value, diff } = ForwardProp::jacobian(f, &(x, y));
/// assert_eq!(value, vectorx![1.0]);
/// assert_eq!(diff.ncols(), 2);
/// ```
pub struct ForwardProp;

impl<I> Diff<I> for ForwardProp
where
    I: StaticDiffInput,
    AllocatorBuffer<I::Dim>: Sync + Send,
    DefaultAllocator: DualAllocator<I::Dim>,
    DualVector<I::Dim>: Copy,
{
    type T = DualVector<I::Dim>;

    fn jacobian<F>(f: F, input: &I) -> DiffResult<VectorX, MatrixX>
    where
        F: Fn(I::Packed<Self::T>) -> VectorX<Self::T>,
    {
        let res = f(input.dual());

        let n = VectorDim::<I::Dim>::zeros().shape_generic().0;
        let eps1 = MatrixDim::<Dyn, I::Dim>::from_rows(
            res.map(|r| r.eps.unwrap_generic(n, Const::<1>).transpose())
                .as_slice(),
        );

        let mut eps = MatrixX::zeros(res.len(), I::Dim::DIM);
        eps.copy_from(&eps1);

        DiffResult {
            value: res.map(|r| r.re),
            diff: eps,
        }
    }
}
