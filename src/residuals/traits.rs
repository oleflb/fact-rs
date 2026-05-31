use std::fmt::Debug;

use crate::{
    containers::{Key, Values},
    dtype,
    linalg::{Diff, DiffInput, DiffResult, MatrixX, Numeric, VectorX},
    residuals::{DynValues, ResidualError, VarPack},
};
use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::DynClone;

/// Typed residual authoring trait.
pub trait Residual: Debug + Clone + Send + 'static {
    type Input: VarPack + DiffInput;
    type Differ: Diff<Self::Input>;

    /// Output dimension for this residual instance.
    ///
    /// This must remain invariant while a graph is optimized. The graph sparse
    /// structure is built from these dimensions.
    fn dim_out(&self) -> usize;

    fn residual<T: Numeric>(&self, input: <Self::Input as DiffInput>::Packed<T>) -> VectorX<T>;

    fn residual_jacobian(&self, input: Self::Input) -> DiffResult<VectorX, MatrixX> {
        <Self::Differ as Diff<Self::Input>>::jacobian(
            |input| <Self as Residual>::residual(self, input),
            &input,
        )
    }
}

/// Object-safe residual trait stored by factors.
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait ErasedResidual: Debug + DynClone + Downcast + Send {
    fn dim_in(&self, values: &Values, keys: &[Key]) -> Result<usize, ResidualError>;

    fn dim_out(&self, values: &Values, keys: &[Key]) -> Result<usize, ResidualError>;

    fn residual(&self, values: &Values, keys: &[Key]) -> Result<VectorX, ResidualError>;

    fn residual_jacobian(
        &self,
        values: &Values,
        keys: &[Key],
    ) -> Result<DiffResult<VectorX, MatrixX>, ResidualError>;
}

dyn_clone::clone_trait_object!(ErasedResidual);
impl_downcast!(ErasedResidual);

#[cfg(feature = "serde")]
pub use register_erasedresidual as tag_residual;

/// Dynamic residuals receive values plus a runtime key pack.
pub trait DynResidual: Debug + Clone + Send + 'static {
    fn dim_out(&self, keys: &[Key]) -> Result<usize, ResidualError>;

    fn residual(&self, input: &DynValues) -> VectorX;

    fn residual_jacobian(&self, input: &DynValues) -> DiffResult<VectorX, MatrixX> {
        numerical_jacobian_dyn(self, input)
    }
}

fn numerical_jacobian_dyn<R: DynResidual>(
    residual: &R,
    input: &DynValues,
) -> DiffResult<VectorX, MatrixX> {
    let eps = dtype::powi(10.0, -6);
    let keys = input.keys();
    let dims = keys
        .iter()
        .map(|key| {
            input
                .get_raw(*key)
                .unwrap_or_else(|| panic!("missing key in dynamic residual: {key:?}"))
                .dim()
        })
        .collect::<Vec<_>>();
    let dim_total = dims.iter().sum();
    let value = residual.residual(input);
    let mut jac = MatrixX::zeros(value.len(), dim_total);

    let mut col = 0;
    for (key, dim) in keys.iter().zip(dims) {
        for j in 0..dim {
            let mut delta = VectorX::zeros(dim);
            delta[j] = eps;

            let mut plus_values = input.clone();
            plus_values
                .get_raw_mut(*key)
                .unwrap_or_else(|| panic!("missing key in dynamic residual: {key:?}"))
                .oplus_mut(delta.as_view());
            let plus = residual.residual(&plus_values);

            delta[j] = -eps;
            let mut minus_values = input.clone();
            minus_values
                .get_raw_mut(*key)
                .unwrap_or_else(|| panic!("missing key in dynamic residual: {key:?}"))
                .oplus_mut(delta.as_view());
            let minus = residual.residual(&minus_values);

            let deriv = (plus - minus) / (2.0 * eps);
            jac.column_mut(col).copy_from(&deriv);
            col += 1;
        }
    }

    DiffResult { value, diff: jac }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assign_symbols,
        containers::{FactorBuilder, Values},
        linalg::{MatrixX, VectorX, vectorx},
        residuals::DynVarPack,
        variables::{Variable, VectorVar2, VectorVar3},
    };

    assign_symbols!(X: VectorVar2; Y: VectorVar3);

    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    struct DimResidual;

    #[factrs::mark]
    impl DynResidual for DimResidual {
        fn dim_out(&self, keys: &[Key]) -> Result<usize, ResidualError> {
            Ok(keys.len())
        }

        fn residual(&self, input: &DynValues) -> VectorX {
            VectorX::from_iterator(
                input.keys().len(),
                input.keys().iter().map(|key| {
                    input
                        .get_raw(*key)
                        .expect("dynamic residual key must exist")
                        .dim() as dtype
                }),
            )
        }
    }

    #[test]
    fn dyn_residual_receives_values_and_key_pack() {
        let keys: Vec<Key> = vec![X(0).into(), Y(0).into()];
        let input = DynVarPack::new(keys).expect("valid dynamic variable pack");
        let mut values = Values::new();
        values.insert(X(0), VectorVar2::identity());
        values.insert(Y(0), VectorVar3::identity());

        let input = DynValues::new(&values, input.keys()).expect("dynamic values");
        let residual = DynResidual::residual(&DimResidual, &input);
        assert_eq!(residual.as_slice(), &[2.0, 3.0]);
    }

    #[test]
    fn dyn_residual_default_numerical_jacobian_has_expected_shape() {
        let keys: Vec<Key> = vec![X(0).into(), Y(0).into()];
        let input = DynVarPack::new(keys).expect("valid dynamic variable pack");
        let factor = FactorBuilder::new_dyn(DimResidual, input).build();
        let mut values = Values::new();
        values.insert(X(0), VectorVar2::identity());
        values.insert(Y(0), VectorVar3::identity());

        let linear = factor.linearize(&values);
        assert_eq!(linear.b.len(), 2);
        assert_eq!(linear.a.mat().shape(), (2, 5));
    }

    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    struct AnalyticResidual;

    #[factrs::mark]
    impl Residual for AnalyticResidual {
        type Input = VectorVar2;
        type Differ = crate::linalg::ForwardProp;

        fn dim_out(&self) -> usize {
            1
        }

        fn residual<T: Numeric>(&self, input: crate::variables::VectorVar<2, T>) -> VectorX<T> {
            vectorx![input.0[0] + T::from(2.0) * input.0[1]]
        }

        fn residual_jacobian(&self, input: VectorVar2) -> DiffResult<VectorX, MatrixX> {
            DiffResult {
                value: <Self as Residual>::residual(self, input),
                diff: MatrixX::from_row_slice(1, 2, &[10.0, 20.0]),
            }
        }
    }

    #[test]
    fn typed_residual_uses_analytic_jacobian_override() {
        let mut values = Values::new();
        values.insert(X(0), VectorVar2::new(1.0, 2.0));
        let factor = FactorBuilder::new(AnalyticResidual, X(0)).build();

        let linear = factor.linearize(&values);

        assert_eq!(linear.a.mat(), MatrixX::from_row_slice(1, 2, &[10.0, 20.0]));
        assert_eq!(linear.b, vectorx![-5.0]);
    }
}
