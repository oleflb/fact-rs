use std::fmt::Debug;

use crate::{
    containers::{Key, Values},
    dtype,
    linalg::{Diff, DiffInput, DiffResult, MatrixX, Numeric, VectorX},
    residuals::{DynVarPack, ResidualError, VarPack},
};
use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::DynClone;

/// Typed residual authoring trait.
pub trait Residual: Debug + Clone + Send + 'static {
    type Input: VarPack + DiffInput;
    type Differ: Diff<Self::Input>;

    fn residual<T: Numeric>(&self, input: <Self::Input as DiffInput>::Packed<T>) -> VectorX<T>;
}

/// Optional marker for residuals with a compile-time output dimension.
pub trait FixedOutputDim {
    type DimOut: crate::linalg::DimName;
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
    fn residual(&self, values: &Values, input: &DynVarPack) -> VectorX;

    fn residual_jacobian(
        &self,
        values: &Values,
        input: &DynVarPack,
    ) -> DiffResult<VectorX, MatrixX> {
        numerical_jacobian_dyn(self, values, input)
    }
}

fn numerical_jacobian_dyn<R: DynResidual>(
    residual: &R,
    values: &Values,
    input: &DynVarPack,
) -> DiffResult<VectorX, MatrixX> {
    let eps = dtype::powi(10.0, -6);
    let keys = input.keys();
    let dims = keys
        .iter()
        .map(|key| {
            values
                .get_raw(*key)
                .unwrap_or_else(|| panic!("missing key in dynamic residual: {key:?}"))
                .dim()
        })
        .collect::<Vec<_>>();
    let dim_total = dims.iter().sum();
    let value = residual.residual(values, input);
    let mut jac = MatrixX::zeros(value.len(), dim_total);

    let mut col = 0;
    for (key, dim) in keys.iter().zip(dims) {
        for j in 0..dim {
            let mut delta = VectorX::zeros(dim);
            delta[j] = eps;

            let mut plus_values = values.clone();
            plus_values
                .get_raw_mut(*key)
                .unwrap_or_else(|| panic!("missing key in dynamic residual: {key:?}"))
                .oplus_mut(delta.as_view());
            let plus = residual.residual(&plus_values, input);

            delta[j] = -eps;
            let mut minus_values = values.clone();
            minus_values
                .get_raw_mut(*key)
                .unwrap_or_else(|| panic!("missing key in dynamic residual: {key:?}"))
                .oplus_mut(delta.as_view());
            let minus = residual.residual(&minus_values, input);

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
        linalg::VectorX,
        variables::{Variable, VectorVar2, VectorVar3},
    };

    assign_symbols!(X: VectorVar2; Y: VectorVar3);

    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    struct DimResidual;

    #[factrs::mark]
    impl DynResidual for DimResidual {
        fn residual(&self, values: &Values, input: &DynVarPack) -> VectorX {
            VectorX::from_iterator(
                input.keys().len(),
                input.keys().iter().map(|key| {
                    values
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

        let residual = DynResidual::residual(&DimResidual, &values, &input);
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
}
