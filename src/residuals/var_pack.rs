use std::{any::type_name, collections::HashSet, fmt};

use crate::{
    containers::{Key, Symbol, TypedSymbol, Values},
    linalg::Numeric,
    variables::{Variable, VariableDtype},
};

/// Errors produced while building or evaluating residual inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidualError {
    WrongKeyCount {
        expected: usize,
        actual: usize,
    },
    DuplicateKey(Key),
    MissingKey(Key),
    WrongVariableType {
        key: Key,
        expected: &'static str,
    },
    NoiseDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    JacobianShapeMismatch {
        expected_rows: usize,
        actual_rows: usize,
        expected_cols: usize,
        actual_cols: usize,
    },
}

impl fmt::Display for ResidualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongKeyCount { expected, actual } => {
                write!(f, "expected {expected} keys, got {actual}")
            }
            Self::DuplicateKey(key) => write!(f, "duplicate key: {key:?}"),
            Self::MissingKey(key) => write!(f, "missing key: {key:?}"),
            Self::WrongVariableType { key, expected } => {
                write!(f, "key {key:?} does not contain variable type {expected}")
            }
            Self::NoiseDimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "noise dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::JacobianShapeMismatch {
                expected_rows,
                actual_rows,
                expected_cols,
                actual_cols,
            } => write!(
                f,
                "jacobian shape mismatch: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}"
            ),
        }
    }
}

impl std::error::Error for ResidualError {}

/// Residual input shape.
///
/// Typed packs use variable types, for example `(SE3, ImuBias)`. Dynamic packs
/// use [`DynVarPack`]. Runtime keys are supplied separately by factor builders.
pub trait VarPack: Send + 'static {
    type Packed<T: Numeric>;

    fn dim_in(values: &Values, keys: &[Key]) -> Result<usize, ResidualError>;

    fn input(values: &Values, keys: &[Key]) -> Result<Self, ResidualError>
    where
        Self: Sized;

    fn pack<T: Numeric>(values: &Values, keys: &[Key]) -> Result<Self::Packed<T>, ResidualError>;
}

fn check_key_count(keys: &[Key], expected: usize) -> Result<(), ResidualError> {
    if keys.len() == expected {
        Ok(())
    } else {
        Err(ResidualError::WrongKeyCount {
            expected,
            actual: keys.len(),
        })
    }
}

fn get_typed<V: VariableDtype + 'static>(values: &Values, key: Key) -> Result<&V, ResidualError> {
    values
        .get_unchecked(key)
        .ok_or(ResidualError::WrongVariableType {
            key,
            expected: type_name::<V>(),
        })
}

impl<V> VarPack for V
where
    V: VariableDtype + 'static,
{
    type Packed<T: Numeric> = V::Alias<T>;

    fn dim_in(values: &Values, keys: &[Key]) -> Result<usize, ResidualError> {
        check_key_count(keys, 1)?;
        Ok(Variable::dim(get_typed::<V>(values, keys[0])?))
    }

    fn input(values: &Values, keys: &[Key]) -> Result<Self, ResidualError> {
        check_key_count(keys, 1)?;
        Ok(get_typed::<V>(values, keys[0])?.clone())
    }

    fn pack<T: Numeric>(values: &Values, keys: &[Key]) -> Result<Self::Packed<T>, ResidualError> {
        check_key_count(keys, 1)?;
        Ok(get_typed::<V>(values, keys[0])?.cast::<T>())
    }
}

macro_rules! impl_tuple_var_pack {
    ($count:expr, $(($idx:tt, $var:ident)),+ $(,)?) => {
        impl<$($var),+> VarPack for ($($var,)+)
        where
            $($var: VariableDtype + 'static,)+
        {
            type Packed<T: Numeric> = ($($var::Alias<T>,)+);

            fn dim_in(values: &Values, keys: &[Key]) -> Result<usize, ResidualError> {
                check_key_count(keys, $count)?;
                let mut dim = 0;
                $(
                    dim += Variable::dim(get_typed::<$var>(values, keys[$idx])?);
                )+
                Ok(dim)
            }

            fn input(values: &Values, keys: &[Key]) -> Result<Self, ResidualError> {
                check_key_count(keys, $count)?;
                Ok(($(
                    get_typed::<$var>(values, keys[$idx])?.clone(),
                )+))
            }

            fn pack<T: Numeric>(
                values: &Values,
                keys: &[Key],
            ) -> Result<Self::Packed<T>, ResidualError> {
                check_key_count(keys, $count)?;
                Ok(($(
                    get_typed::<$var>(values, keys[$idx])?.cast::<T>(),
                )+))
            }
        }
    };
}

impl_tuple_var_pack!(2, (0, V1), (1, V2));
impl_tuple_var_pack!(3, (0, V1), (1, V2), (2, V3));
impl_tuple_var_pack!(4, (0, V1), (1, V2), (2, V3), (3, V4));
impl_tuple_var_pack!(5, (0, V1), (1, V2), (2, V3), (3, V4), (4, V5));
impl_tuple_var_pack!(6, (0, V1), (1, V2), (2, V3), (3, V4), (4, V5), (5, V6));

/// Concrete dynamic residual input pack.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DynVarPack {
    keys: Vec<Key>,
}

impl DynVarPack {
    pub fn new<I, K>(keys: I) -> Result<Self, ResidualError>
    where
        I: IntoIterator<Item = K>,
        K: Into<Key>,
    {
        let keys = keys.into_iter().map(Into::into).collect::<Vec<_>>();
        let mut seen = HashSet::with_capacity(keys.len());
        for key in &keys {
            if !seen.insert(*key) {
                return Err(ResidualError::DuplicateKey(*key));
            }
        }
        Ok(Self { keys })
    }

    pub(crate) fn from_keys_unchecked(keys: Vec<Key>) -> Self {
        Self { keys }
    }

    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    pub(crate) fn into_keys(self) -> Vec<Key> {
        self.keys
    }
}

impl VarPack for DynVarPack {
    type Packed<T: Numeric> = DynVarPack;

    fn dim_in(values: &Values, keys: &[Key]) -> Result<usize, ResidualError> {
        keys.iter().try_fold(0, |dim, key| {
            values
                .get_raw(*key)
                .map(|value| dim + value.dim())
                .ok_or(ResidualError::MissingKey(*key))
        })
    }

    fn input(_values: &Values, keys: &[Key]) -> Result<Self, ResidualError> {
        Ok(Self::from_keys_unchecked(keys.to_vec()))
    }

    fn pack<T: Numeric>(_values: &Values, keys: &[Key]) -> Result<Self::Packed<T>, ResidualError> {
        Ok(Self::from_keys_unchecked(keys.to_vec()))
    }
}

/// Converts user supplied factor input into ordered factor keys while validating
/// typed symbol packs at compile time when possible.
pub trait FactorInput<P: VarPack> {
    fn into_keys(self) -> Vec<Key>;
}

impl<K, V> FactorInput<V> for K
where
    K: TypedSymbol<V>,
    V: VariableDtype + 'static,
{
    fn into_keys(self) -> Vec<Key> {
        vec![self.into()]
    }
}

impl FactorInput<DynVarPack> for DynVarPack {
    fn into_keys(self) -> Vec<Key> {
        self.into_keys()
    }
}

/// Converts unchecked key packs into ordered factor keys.
pub trait KeyPack {
    fn into_keys(self) -> Vec<Key>;
}

impl<K> KeyPack for K
where
    K: Symbol,
{
    fn into_keys(self) -> Vec<Key> {
        vec![self.into()]
    }
}

impl<K> KeyPack for Vec<K>
where
    K: Into<Key>,
{
    fn into_keys(self) -> Vec<Key> {
        self.into_iter().map(Into::into).collect()
    }
}

impl<K, const N: usize> KeyPack for [K; N]
where
    K: Into<Key>,
{
    fn into_keys(self) -> Vec<Key> {
        self.into_iter().map(Into::into).collect()
    }
}

impl KeyPack for DynVarPack {
    fn into_keys(self) -> Vec<Key> {
        self.into_keys()
    }
}

macro_rules! impl_tuple_factor_input {
    ($(($key:ident, $var:ident)),+ $(,)?) => {
        impl<$($key, $var),+> FactorInput<($($var,)+)> for ($($key,)+)
        where
            $($key: TypedSymbol<$var>,)+
            $($var: VariableDtype + 'static,)+
        {
            fn into_keys(self) -> Vec<Key> {
                #[allow(non_snake_case)]
                let ($($key,)+) = self;
                vec![$($key.into(),)+]
            }
        }

        impl<$($key),+> KeyPack for ($($key,)+)
        where
            $($key: Symbol,)+
        {
            fn into_keys(self) -> Vec<Key> {
                #[allow(non_snake_case)]
                let ($($key,)+) = self;
                vec![$($key.into(),)+]
            }
        }
    };
}

impl_tuple_factor_input!((K1, V1), (K2, V2));
impl_tuple_factor_input!((K1, V1), (K2, V2), (K3, V3));
impl_tuple_factor_input!((K1, V1), (K2, V2), (K3, V3), (K4, V4));
impl_tuple_factor_input!((K1, V1), (K2, V2), (K3, V3), (K4, V4), (K5, V5));
impl_tuple_factor_input!((K1, V1), (K2, V2), (K3, V3), (K4, V4), (K5, V5), (K6, V6));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assign_symbols,
        variables::{Variable, VectorVar2, VectorVar3},
    };

    assign_symbols!(X: VectorVar2; Y: VectorVar3);

    #[test]
    fn dyn_var_pack_preserves_key_order() {
        let keys: Vec<Key> = vec![X(2).into(), X(0).into(), X(1).into()];
        let pack = DynVarPack::new(keys).expect("valid dynamic variable pack");
        assert_eq!(pack.keys(), &[X(2).into(), X(0).into(), X(1).into()]);
    }

    #[test]
    fn dyn_var_pack_rejects_duplicate_keys() {
        let keys: Vec<Key> = vec![X(0).into(), X(0).into()];
        let err = DynVarPack::new(keys).expect_err("duplicate key must be rejected");
        assert_eq!(err, ResidualError::DuplicateKey(X(0).into()));
    }

    #[test]
    fn typed_tuple_pack_dim_in_sums_value_dims() {
        let mut values = Values::new();
        values.insert(X(0), VectorVar2::identity());
        values.insert(Y(0), VectorVar3::identity());
        let keys = <(X, Y) as FactorInput<(VectorVar2, VectorVar3)>>::into_keys((X(0), Y(0)));

        let dim = <(VectorVar2, VectorVar3) as VarPack>::dim_in(&values, &keys)
            .expect("valid typed tuple pack");
        assert_eq!(dim, 5);
    }

    #[test]
    fn typed_tuple_pack_rejects_wrong_variable_type() {
        let mut values = Values::new();
        values.insert_unchecked(X(0), VectorVar3::identity());
        values.insert(Y(0), VectorVar3::identity());
        let keys = vec![X(0).into(), Y(0).into()];

        let err = <(VectorVar2, VectorVar3) as VarPack>::dim_in(&values, &keys)
            .expect_err("wrong variable type must be rejected");
        assert!(matches!(err, ResidualError::WrongVariableType { .. }));
    }
}
