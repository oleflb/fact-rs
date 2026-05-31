use std::{
    fmt::{self, Write},
    marker::PhantomData,
};

use pad_adapter::PadAdapter;

use super::{DefaultSymbolHandler, KeyFormatter};
use crate::{
    containers::{Key, Values},
    dtype,
    linalg::{DiffResult, MatrixBlock},
    linear::LinearFactor,
    noise::{NoiseModel, UnitNoiseDyn},
    residuals::{
        DynResidual, DynVarPack, ErasedResidual, FactorInput, KeyPack, Residual, ResidualError,
    },
    robust::{L2, RobustCost},
};

/// Main structure to represent a factor in the graph.
///
/// $$ \blue{\rho_i}(||\purple{r_i}(\green{\Theta})||_{\red{\Sigma_i}} ) $$
///
/// Factors are the main building block of the factor graph. They are composed
/// of four pieces:
/// - <green>Keys</green>: The variables that the factor depends on, given by a
///   slice of [Keys](Key).
/// - <purple>Residual</purple>: The vector-valued function that computes the
///   error of the factor given a set of values, from the
///   [residual](crate::residuals) module.
/// - <red>Noise Model</red>: The noise model describes the uncertainty of the
///   residual, given by the traits in the [noise](crate::noise) module.
/// - <blue>Robust Kernel</blue>: The robust kernel weights the error of the
///   factor, given by the traits in the [robust](crate::robust) module.
///
/// The easiest way to construct a factor is using the [fac](factrs::fac) macro,
/// or alternatively, using [FactorBuilder].
///
/// During optimization the factor is linearized around a set of values into a
/// [LinearFactor].
///
///  ```
/// # use factrs::{
///    assign_symbols,
///    containers::FactorBuilder,
///    noise::GaussianNoise,
///    optimizers::GaussNewton,
///    residuals::{PriorResidual},
///    robust::GemanMcClure,
///    variables::VectorVar3,
/// };
/// # assign_symbols!(X: VectorVar3);
/// let prior = VectorVar3::new(1.0, 2.0, 3.0);
/// let residual = PriorResidual::new(prior);
/// let noise = GaussianNoise::<3>::from_diag_sigmas(1e-1, 2e-1, 3e-1);
/// let robust = GemanMcClure::default();
/// let factor = FactorBuilder::new(residual, X(0)).noise(noise).robust(robust).build();
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Factor {
    pub(crate) keys: Vec<Key>,
    pub(crate) residual: Box<dyn ErasedResidual>,
    pub(crate) noise: Box<dyn NoiseModel>,
    pub(crate) robust: Box<dyn RobustCost>,
}

impl Factor {
    /// Compute the error of the factor given a set of values.
    pub fn error(&self, values: &Values) -> dtype {
        self.try_error(values)
            .expect("failed to evaluate factor error")
    }

    pub fn try_error(&self, values: &Values) -> Result<dtype, ResidualError> {
        let r = self.residual.residual(values, &self.keys)?;
        self.validate_noise_dim(r.len())?;
        let r = self.noise.whiten_vec(r);
        let norm2 = r.norm_squared();
        Ok(self.robust.loss(norm2))
    }

    /// Compute the dimension of the output of the factor.
    pub fn dim_out(&self, values: &Values) -> usize {
        self.try_dim_out(values)
            .expect("failed to evaluate factor output dimension")
    }

    pub fn try_dim_out(&self, values: &Values) -> Result<usize, ResidualError> {
        self.residual.dim_out(values, &self.keys)
    }

    /// Linearize the factor given a set of values into a [LinearFactor].
    pub fn linearize(&self, values: &Values) -> LinearFactor {
        self.try_linearize(values)
            .expect("failed to linearize factor")
    }

    pub fn try_linearize(&self, values: &Values) -> Result<LinearFactor, ResidualError> {
        // Compute residual and jacobian
        let DiffResult { value: r, diff: a } =
            self.residual.residual_jacobian(values, &self.keys)?;
        self.validate_linearization(values, r.len(), a.nrows(), a.ncols())?;

        // Whiten residual and jacobian
        let r = self.noise.whiten_vec(r);
        let a = self.noise.whiten_mat(a);

        // Weight according to robust cost
        let norm2 = r.norm_squared();
        let weight = self.robust.weight(norm2).sqrt();
        let a = weight * a;
        let b = -weight * r;

        // Turn A into a MatrixBlock
        let idx = self
            .keys
            .iter()
            .scan(0, |sum, k| {
                let out = Some(*sum);
                *sum += values.get_raw(*k).expect("Key missing in values").dim();
                out
            })
            .collect::<Vec<_>>();
        let a = MatrixBlock::new(a, idx);

        Ok(LinearFactor::new(self.keys.clone(), a, b))
    }

    /// Get the keys of the factor.
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    pub fn is_residual<R>(&self) -> bool
    where
        R: ErasedResidual + 'static,
    {
        self.residual_as::<R>().is_some()
    }

    pub fn residual_as<R>(&self) -> Option<&R>
    where
        R: ErasedResidual + 'static,
    {
        self.residual.downcast_ref::<R>()
    }

    pub fn residual_as_mut<R>(&mut self) -> Option<&mut R>
    where
        R: ErasedResidual + 'static,
    {
        self.residual.downcast_mut::<R>()
    }

    pub fn is_noise<N>(&self) -> bool
    where
        N: NoiseModel + 'static,
    {
        self.noise_as::<N>().is_some()
    }

    pub fn noise_as<N>(&self) -> Option<&N>
    where
        N: NoiseModel + 'static,
    {
        self.noise.downcast_ref::<N>()
    }

    pub fn noise_as_mut<N>(&mut self) -> Option<&mut N>
    where
        N: NoiseModel + 'static,
    {
        self.noise.downcast_mut::<N>()
    }

    pub fn is_robust<C>(&self) -> bool
    where
        C: RobustCost + 'static,
    {
        self.robust_as::<C>().is_some()
    }

    pub fn robust_as<C>(&self) -> Option<&C>
    where
        C: RobustCost + 'static,
    {
        self.robust.downcast_ref::<C>()
    }

    pub fn robust_as_mut<C>(&mut self) -> Option<&mut C>
    where
        C: RobustCost + 'static,
    {
        self.robust.downcast_mut::<C>()
    }

    pub fn set_noise<N>(&mut self, noise: N)
    where
        N: NoiseModel + 'static,
    {
        self.noise = Box::new(noise);
    }

    pub fn set_robust<C>(&mut self, robust: C)
    where
        C: RobustCost + 'static,
    {
        self.robust = Box::new(robust);
    }

    pub fn replace<R, K>(&mut self, residual: R, keys: K)
    where
        R: Residual + ErasedResidual + 'static,
        K: FactorInput<R::Input>,
    {
        self.keys = keys.into_keys();
        self.residual = Box::new(residual);
    }

    pub fn replace_unchecked<R, K>(&mut self, residual: R, keys: K)
    where
        R: Residual + ErasedResidual + 'static,
        K: KeyPack,
    {
        self.keys = keys.into_keys();
        self.residual = Box::new(residual);
    }

    pub fn replace_dyn<R>(&mut self, residual: R, input: DynVarPack)
    where
        R: DynResidual + ErasedResidual + 'static,
    {
        self.keys = input.into_keys();
        self.residual = Box::new(residual);
    }

    fn validate_noise_dim(&self, residual_dim: usize) -> Result<(), ResidualError> {
        let noise_dim = self.noise.dim();
        if noise_dim == 0 || noise_dim == residual_dim {
            Ok(())
        } else {
            Err(ResidualError::NoiseDimensionMismatch {
                expected: residual_dim,
                actual: noise_dim,
            })
        }
    }

    fn validate_linearization(
        &self,
        values: &Values,
        residual_dim: usize,
        jac_rows: usize,
        jac_cols: usize,
    ) -> Result<(), ResidualError> {
        self.validate_noise_dim(residual_dim)?;
        let expected_cols = self.keys.iter().try_fold(0, |dim, key| {
            values
                .get_raw(*key)
                .map(|value| dim + value.dim())
                .ok_or(ResidualError::MissingKey(*key))
        })?;
        if jac_rows == residual_dim && jac_cols == expected_cols {
            Ok(())
        } else {
            Err(ResidualError::JacobianShapeMismatch {
                expected_rows: residual_dim,
                actual_rows: jac_rows,
                expected_cols,
                actual_cols: jac_cols,
            })
        }
    }
}

impl fmt::Debug for Factor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        FactorFormatter::<DefaultSymbolHandler>::new(self).fmt(f)
    }
}

/// Formatter for a factor
///
/// Specifically, this can be used if custom symbols are desired. See
/// [tests/custom_key](https://github.com/rpl-cmu/factrs/blob/dev/tests/custom_key.rs) for examples.
pub struct FactorFormatter<'f, KF> {
    factor: &'f Factor,
    kf: PhantomData<KF>,
}

impl<'f, KF> FactorFormatter<'f, KF> {
    pub fn new(factor: &'f Factor) -> Self {
        Self {
            factor,
            kf: Default::default(),
        }
    }
}

impl<KF: KeyFormatter> fmt::Debug for FactorFormatter<'_, KF> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.write_str("Factor {\n")?;
            let mut pad = PadAdapter::new(f);
            // Keys
            pad.write_str("key: [")?;
            for (i, key) in self.factor.keys().iter().enumerate() {
                if i > 0 {
                    pad.write_str(", ")?;
                }
                KF::fmt(&mut pad, *key)?;
            }
            pad.write_str("]\n")?;
            // Residual
            writeln!(pad, "res: {:#?}", self.factor.residual)?;
            // Noise
            writeln!(pad, "noi: {:#?}", self.factor.noise)?;
            // Robust
            writeln!(pad, "rob: {:#?}", self.factor.robust)?;
            f.write_str("}")?;
        } else {
            f.write_str("Factor { ")?;
            for (i, key) in self.factor.keys().iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                KF::fmt(f, *key)?;
            }
            write!(
                f,
                "], residual: {:?}, noise: {:?}, robust: {:?} }}",
                self.factor.residual, self.factor.noise, self.factor.robust
            )?;
        }

        Ok(())
    }
}

/// Builder for a factor.
///
/// If the noise model or robust kernel aren't set, they default to
/// [UnitNoiseDyn] and [L2] respectively.
pub struct FactorBuilder {
    keys: Vec<Key>,
    residual: Box<dyn ErasedResidual>,
    noise: Option<Box<dyn NoiseModel>>,
    robust: Option<Box<dyn RobustCost>>,
}

impl FactorBuilder {
    /// Create a typed factor while verifying key types at compile time.
    pub fn new<R, K>(residual: R, keys: K) -> Self
    where
        R: Residual + ErasedResidual + 'static,
        K: FactorInput<R::Input>,
    {
        Self {
            keys: keys.into_keys(),
            residual: Box::new(residual),
            noise: None,
            robust: None,
        }
    }

    /// Create a typed factor without compile-time key type validation.
    pub fn new_unchecked<R, K>(residual: R, keys: K) -> Self
    where
        R: Residual + ErasedResidual + 'static,
        K: KeyPack,
    {
        Self {
            keys: keys.into_keys(),
            residual: Box::new(residual),
            noise: None,
            robust: None,
        }
    }

    /// Create a dynamic factor from a dynamic residual and key pack.
    pub fn new_dyn<R>(residual: R, input: DynVarPack) -> Self
    where
        R: DynResidual + ErasedResidual + 'static,
    {
        Self {
            keys: input.into_keys(),
            residual: Box::new(residual),
            noise: None,
            robust: None,
        }
    }

    /// Add a noise model to the factor.
    pub fn noise<N>(mut self, noise: N) -> Self
    where
        N: 'static + NoiseModel,
    {
        self.noise = Some(Box::new(noise));
        self
    }

    /// Add a robust kernel to the factor.
    pub fn robust<C>(mut self, robust: C) -> Self
    where
        C: 'static + RobustCost,
    {
        self.robust = Some(Box::new(robust));
        self
    }

    /// Build the factor.
    pub fn build(self) -> Factor {
        let noise = self
            .noise
            .unwrap_or_else(|| Box::new(UnitNoiseDyn::default()));
        let robust = self.robust.unwrap_or_else(|| Box::new(L2));
        Factor {
            keys: self.keys,
            residual: self.residual,
            noise,
            robust,
        }
    }
}

#[cfg(test)]
mod tests {

    use factrs_proc::fac;
    use matrixcompare::assert_matrix_eq;

    use super::*;
    use crate::{
        assign_symbols,
        linalg::{Diff, NumericalDiff},
        noise::GaussianNoise,
        residuals::{BetweenResidual, PriorResidual},
        robust::GemanMcClure,
        variables::{Variable, VectorVar3},
    };

    #[cfg(not(feature = "f32"))]
    const PWR: i32 = 6;
    #[cfg(not(feature = "f32"))]
    const TOL: f64 = 1e-6;

    #[cfg(feature = "f32")]
    const PWR: i32 = 3;
    #[cfg(feature = "f32")]
    const TOL: f32 = 1e-3;

    assign_symbols!(X: VectorVar3);

    #[test]
    fn linearize_a() {
        let prior = VectorVar3::new(1.0, 2.0, 3.0);
        let x = VectorVar3::identity();

        let residual = PriorResidual::new(prior);
        let noise = GaussianNoise::<3>::from_diag_sigmas(1e-1, 2e-1, 3e-1);
        let robust = GemanMcClure::default();

        let factor: Factor = fac![residual, X(0), noise, robust];

        let f = |x: VectorVar3| {
            let mut values = Values::new();
            values.insert_unchecked(X(0), x);
            factor.error(&values)
        };

        let mut values = Values::new();
        values.insert_unchecked(X(0), x.clone());

        let linear = factor.linearize(&values);
        let grad_got = -linear.a.mat().transpose() * linear.b;
        println!("Received {grad_got:}");

        let grad_num = NumericalDiff::<PWR>::gradient(f, &x).diff;
        println!("Expected {grad_num:}");

        assert_matrix_eq!(grad_got, grad_num, comp = abs, tol = TOL);
    }

    #[test]
    fn linearize_block() {
        let bet = VectorVar3::new(1.0, 2.0, 3.0);
        let x = <VectorVar3 as Variable>::identity();

        let residual = BetweenResidual::new(bet);
        let noise = GaussianNoise::<3>::from_diag_sigmas(1e-1, 2e-1, 3e-1);
        let robust = GemanMcClure::default();

        let factor: Factor = fac![residual, (X(0), X(1)), noise, robust];

        let mut values = Values::new();
        values.insert_unchecked(X(0), x.clone());
        values.insert_unchecked(X(1), x);

        let linear = factor.linearize(&values);

        println!("Full Mat {:}", linear.a.mat());
        println!("First Block {:}", linear.a.get_block(0));
        println!("Second Block {:}", linear.a.get_block(1));

        assert_matrix_eq!(
            linear.a.get_block(0),
            linear.a.mat().columns(0, 3),
            comp = float
        );
        assert_matrix_eq!(
            linear.a.get_block(1),
            linear.a.mat().columns(3, 3),
            comp = float
        );
    }
}
