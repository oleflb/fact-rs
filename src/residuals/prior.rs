use crate::{
    linalg::{
        AllocatorBuffer, DefaultAllocator, DualAllocator, DualVector, ForwardProp, Numeric, VectorX,
    },
    residuals::{FixedOutputDim, Residual},
    variables::{Variable, VariableDtype},
};

/// Unary factor for a prior on a variable.
///
/// This residual is used to enforce a prior on a variable. Specifically it
/// computes $$
/// z \ominus v
/// $$
/// where $z$ is the prior value and $v$ is the variable being estimated.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PriorResidual<P> {
    prior: P,
}

impl<P: VariableDtype> PriorResidual<P> {
    pub fn new(prior: P) -> Self {
        Self { prior }
    }
}

#[factrs::mark]
impl<P> Residual for PriorResidual<P>
where
    P: VariableDtype + 'static + Send,
    AllocatorBuffer<P::Dim>: Sync + Send,
    DefaultAllocator: DualAllocator<P::Dim>,
    DualVector<P::Dim>: Copy,
{
    type Input = P;
    type Differ = ForwardProp;

    fn residual<T: Numeric>(&self, v: <P as Variable>::Alias<T>) -> VectorX<T> {
        self.prior.cast::<T>().ominus(&v)
    }
}

impl<P> FixedOutputDim for PriorResidual<P>
where
    P: VariableDtype + 'static + Send,
{
    type DimOut = P::Dim;
}

#[cfg(test)]
mod test {

    use matrixcompare::assert_matrix_eq;

    use super::*;
    use crate::{
        containers::Values,
        linalg::{DefaultAllocator, Diff, DualAllocator, NumericalDiff, vectorx},
        symbols::X,
        variables::{SE3, SO3, VectorVar3},
    };

    #[cfg(not(feature = "f32"))]
    const PWR: i32 = 6;
    #[cfg(not(feature = "f32"))]
    const TOL: f64 = 1e-6;

    #[cfg(feature = "f32")]
    const PWR: i32 = 4;
    #[cfg(feature = "f32")]
    const TOL: f32 = 1e-2;

    fn test_prior_jacobian<
        #[cfg(feature = "serde")] P: VariableDtype + 'static + typetag::Tagged,
        #[cfg(not(feature = "serde"))] P: VariableDtype + 'static,
    >(
        prior: P,
    ) where
        AllocatorBuffer<P::Dim>: Sync + Send,
        DefaultAllocator: DualAllocator<P::Dim>,
        DualVector<P::Dim>: Copy,
    {
        let prior_residual = PriorResidual::new(prior);

        let x1 = P::identity();
        let mut values = Values::new();
        values.insert_unchecked(X(0), x1.clone());
        let jac = crate::residuals::ErasedResidual::residual_jacobian(
            &prior_residual,
            &values,
            &[X(0).into()],
        )
        .expect("prior residual should linearize")
        .diff;

        let f = |v: P| {
            let mut vals = Values::new();
            vals.insert_unchecked(X(0), v.clone());
            crate::residuals::ErasedResidual::residual(&prior_residual, &vals, &[X(0).into()])
                .expect("prior residual should evaluate")
        };
        let jac_n = NumericalDiff::<PWR>::jacobian(f, &x1).diff;

        eprintln!("jac: {jac:.3}");
        eprintln!("jac_n: {jac_n:.3}");

        assert_matrix_eq!(jac, jac_n, comp = abs, tol = TOL);
    }

    #[test]
    fn prior_linear() {
        test_prior_jacobian(VectorVar3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn prior_so3() {
        let prior = SO3::exp(vectorx![0.1, 0.2, 0.3].as_view());
        test_prior_jacobian(prior);
    }

    #[test]
    fn prior_se3() {
        let prior = SE3::exp(vectorx![0.1, 0.2, 0.3, 1.0, 2.0, 3.0].as_view());
        test_prior_jacobian(prior);
    }
}
