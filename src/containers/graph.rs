use std::{
    fmt::{Debug, Write},
    marker::PhantomData,
};

use faer::sparse::{Pair, SymbolicSparseColMat};
use foldhash::HashMap;
use pad_adapter::PadAdapter;

use super::{DefaultSymbolHandler, Idx, Key, KeyFormatter, Values, ValuesOrder};
// Once "debug_closure_helpers" is stabilized, we won't need this anymore
// Need custom debug to handle pretty key printing at the moment
// Pad adapter helps with the pretty printing
use crate::containers::factor::FactorFormatter;
use crate::{
    containers::Factor,
    dtype,
    linalg::{MatrixX, VectorX},
    linear::{LinearGraph, accumulate_dense_normal_factor},
    residuals::{ErasedResidual, QueryInput, QueryKeys, Residual},
};

/// Structure to represent a nonlinear factor graph
///
/// Main usage will be via `add_factor` to add new [factors](Factor) to the
/// graph. Also of note is the `linearize` function that returns a [linear (aka
/// Gaussian) factor graph](LinearGraph).
///
/// Since the graph represents a nonlinear least-squares problem, during
/// optimization it will be iteratively linearized about a set of variables and
/// solved iteratively.
///
/// ```
/// # use factrs::{
///    assign_symbols,
///    containers::{Graph, FactorBuilder},
///    residuals::PriorResidual,
///    robust::GemanMcClure,
///    traits::*,
///    variables::SO2,
/// };
/// # assign_symbols!(X: SO2);
/// # let factor = FactorBuilder::new(PriorResidual::new(SO2::identity()), X(0)).build();
/// let mut graph = Graph::new();
/// graph.add_factor(factor);
/// ```
#[derive(Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Graph {
    factors: Vec<Factor>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            factors: Vec::with_capacity(capacity),
        }
    }

    pub fn at(&self, idx: usize) -> &Factor {
        &self.factors[idx]
    }

    pub fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    pub fn factors_for<'a, K>(&'a self, keys: K) -> impl Iterator<Item = &'a Factor> + 'a
    where
        K: QueryKeys + 'a,
        K::Storage: 'a,
    {
        let keys = keys.into_storage();
        self.factors
            .iter()
            .filter(move |factor| factor.keys() == keys.as_ref())
    }

    pub fn factors_for_mut<'a, K>(
        &'a mut self,
        keys: K,
    ) -> impl Iterator<Item = &'a mut Factor> + 'a
    where
        K: QueryKeys + 'a,
        K::Storage: 'a,
    {
        let keys = keys.into_storage();
        self.factors
            .iter_mut()
            .filter(move |factor| factor.keys() == keys.as_ref())
    }

    pub fn factors_for_residual<'a, R, K>(
        &'a self,
        keys: K,
    ) -> impl Iterator<Item = &'a Factor> + 'a
    where
        R: Residual + ErasedResidual + 'static,
        K: QueryInput<R::Input> + 'a,
        K::Storage: 'a,
    {
        self.factors_for(keys)
            .filter(|factor| factor.is_residual::<R>())
    }

    pub fn factors_for_residual_mut<'a, R, K>(
        &'a mut self,
        keys: K,
    ) -> impl Iterator<Item = &'a mut Factor> + 'a
    where
        R: Residual + ErasedResidual + 'static,
        K: QueryInput<R::Input> + 'a,
        K::Storage: 'a,
    {
        self.factors_for_mut(keys)
            .filter(|factor| factor.is_residual::<R>())
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    pub fn error(&self, values: &Values) -> dtype {
        self.factors.iter().map(|f| f.error(values)).sum()
    }

    pub fn linearize(&self, values: &Values) -> LinearGraph {
        let factors = self.factors.iter().map(|f| f.linearize(values)).collect();
        LinearGraph::from_vec(factors)
    }

    pub(crate) fn dense_normal_equations(
        &self,
        values: &Values,
        order: &ValuesOrder,
    ) -> (MatrixX, VectorX) {
        let dim = order.dim();
        let mut hessian = MatrixX::zeros(dim, dim);
        let mut rhs = VectorX::zeros(dim);

        for nonlinear_factor in &self.factors {
            let factor = nonlinear_factor.linearize(values);
            accumulate_dense_normal_factor(&factor, order, &mut hessian, &mut rhs);
        }

        (hessian, rhs)
    }

    pub fn structure_hash(&self, values: &Values) -> GraphStructureHash {
        self.structure(values).hash
    }

    pub fn sparsity_pattern(&self, values: &Values) -> GraphOrder {
        let structure = self.structure(values);
        self.sparsity_pattern_from_structure(structure)
    }

    pub(crate) fn structure(&self, values: &Values) -> GraphStructure {
        let mut order_map = HashMap::default();
        let mut order_entries = Vec::new();
        let mut col = 0;
        let mut total_rows = 0;
        let mut factor_rows = Vec::with_capacity(self.factors.len());

        for factor in &self.factors {
            let dim_out = factor.dim_out(values);
            factor_rows.push(dim_out);
            total_rows += dim_out;

            for key in factor.keys() {
                if order_map.contains_key(key) {
                    continue;
                }

                let dim = values
                    .get_raw(*key)
                    .unwrap_or_else(|| panic!("key {key:?} missing in values"))
                    .dim();
                order_map.insert(*key, Idx { idx: col, dim });
                order_entries.push((*key, dim));
                col += dim;
            }
        }

        let order = ValuesOrder::new(order_map);
        let hash = self.hash_structure(&order_entries, &factor_rows, total_rows, col);

        GraphStructure {
            hash,
            order,
            factor_rows,
            total_rows,
            total_cols: col,
        }
    }

    pub(crate) fn sparsity_pattern_from_structure(&self, structure: GraphStructure) -> GraphOrder {
        let GraphStructure {
            hash,
            order,
            factor_rows,
            total_rows,
            total_cols,
        } = structure;
        let mut indices = Vec::<Pair<usize, usize>>::new();

        let _ = self
            .factors
            .iter()
            .zip(&factor_rows)
            .fold(0, |row, (f, dim_out)| {
                f.keys().iter().for_each(|key| {
                    (0..*dim_out).for_each(|i| {
                        let Idx {
                            idx: col,
                            dim: col_dim,
                        } = order.get(*key).expect("Key missing in values");
                        (0..*col_dim).for_each(|j| {
                            indices.push(Pair::new(row + i, col + j));
                        });
                    });
                });
                row + *dim_out
            });

        let (sparsity_pattern, sparsity_order) =
            SymbolicSparseColMat::try_new_from_indices(total_rows, total_cols, &indices)
                .expect("Failed to make sparse matrix");
        GraphOrder {
            structure_hash: hash,
            order,
            sparsity_pattern,
            sparsity_order,
        }
    }

    fn hash_structure(
        &self,
        order_entries: &[(Key, usize)],
        factor_rows: &[usize],
        total_rows: usize,
        total_cols: usize,
    ) -> GraphStructureHash {
        let mut hasher = StructureHasher::new();
        hasher.write_usize(self.factors.len());
        hasher.write_usize(total_rows);
        hasher.write_usize(total_cols);
        hasher.write_usize(order_entries.len());

        for (key, dim) in order_entries {
            hasher.write_u64(key.0);
            hasher.write_usize(*dim);
        }

        for (factor, dim_out) in self.factors.iter().zip(factor_rows) {
            hasher.write_usize(*dim_out);
            hasher.write_usize(factor.keys().len());
            for key in factor.keys() {
                hasher.write_u64(key.0);
            }
        }

        hasher.finish()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Factor> {
        self.factors.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Factor> {
        self.factors.iter_mut()
    }

    pub fn remove_factors<F>(&mut self, should_remove: F) -> Vec<Factor>
    where
        F: Fn(&Factor) -> bool,
    {
        self.factors
            .extract_if(.., |factor| should_remove(factor))
            .collect::<Vec<_>>()
    }
}

impl Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        GraphFormatter::<DefaultSymbolHandler>::new(self).fmt(f)
    }
}

impl IntoIterator for Graph {
    type Item = Factor;
    type IntoIter = std::vec::IntoIter<Factor>;

    fn into_iter(self) -> Self::IntoIter {
        self.factors.into_iter()
    }
}

/// Formatter for a graph
///
/// Specifically, this can be used if custom symbols are desired. See
/// [tests/custom_key](https://github.com/rpl-cmu/factrs/blob/dev/tests/custom_key.rs) for examples.
pub struct GraphFormatter<'g, KF> {
    pub graph: &'g Graph,
    kf: PhantomData<KF>,
}

impl<'g, KF> GraphFormatter<'g, KF> {
    pub fn new(graph: &'g Graph) -> Self {
        Self {
            graph,
            kf: Default::default(),
        }
    }
}

impl<KF: KeyFormatter> Debug for GraphFormatter<'_, KF> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.write_str("Graph [\n")?;
            let mut pad = PadAdapter::new(f);
            for factor in self.graph.factors.iter() {
                writeln!(pad, "{:#?},", FactorFormatter::<KF>::new(factor))?;
            }
            f.write_str("]")
        } else {
            f.write_str("Graph [ ")?;
            for factor in self.graph.factors.iter() {
                write!(f, "{:?}, ", FactorFormatter::<KF>::new(factor))?;
            }
            f.write_str("]")
        }
    }
}

/// Simple structure to hold the order of the graph
///
/// Specifically this is used to cache linearization results such as the order
/// of the graph and the sparsity pattern of the Jacobian (allows use to avoid
/// resorting indices).
pub struct GraphOrder {
    pub structure_hash: GraphStructureHash,
    // Contains the order of the variables
    pub order: ValuesOrder,
    // Contains the sparsity pattern of the jacobian
    pub sparsity_pattern: SymbolicSparseColMat<usize>,
    // Contains the order of values to put into the sparsity pattern
    pub sparsity_order: faer::sparse::Argsort<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphStructureHash(pub u64);

impl GraphStructureHash {
    pub const EMPTY: Self = Self(0);
}

pub(crate) struct GraphStructure {
    pub hash: GraphStructureHash,
    pub order: ValuesOrder,
    pub factor_rows: Vec<usize>,
    pub total_rows: usize,
    pub total_cols: usize,
}

struct StructureHasher {
    state: u64,
}

impl StructureHasher {
    const SEED: u64 = 0x517c_c1b7_2722_0a95;
    const PRIME: u64 = 0x9e37_79b9_7f4a_7c15;

    fn new() -> Self {
        Self { state: Self::SEED }
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_u64(&mut self, value: u64) {
        self.state ^= value
            .wrapping_add(Self::PRIME)
            .wrapping_add(self.state << 6)
            .wrapping_add(self.state >> 2);
        self.state = Self::mix(self.state);
    }

    fn finish(self) -> GraphStructureHash {
        GraphStructureHash(Self::mix(self.state))
    }

    fn mix(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assign_symbols,
        containers::{FactorBuilder, FactorQuery, FactorQueryMut, Values},
        linalg::{ForwardProp, Numeric, VectorX},
        noise::{GaussianNoise, UnitNoiseDyn},
        residuals::{BetweenResidual, PriorResidual, Residual},
        robust::{GemanMcClure, L2},
        variables::{Variable, VectorVar2, VectorVar3},
    };

    assign_symbols!(X: VectorVar2; Y: VectorVar3);

    fn values_x0() -> Values {
        let mut values = Values::new();
        values.insert(X(0), VectorVar2::identity());
        values
    }

    fn prior_x(idx: u32) -> Factor {
        FactorBuilder::new(PriorResidual::new(VectorVar2::new(1.0, 2.0)), X(idx)).build()
    }

    fn between_x(lhs: u32, rhs: u32) -> Factor {
        FactorBuilder::new(
            BetweenResidual::new(VectorVar2::new(1.0, 2.0)),
            (X(lhs), X(rhs)),
        )
        .build()
    }

    #[test]
    fn factors_for_matches_exact_ordered_keys() {
        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));
        graph.add_factor(between_x(0, 1));

        assert_eq!(graph.factors_for(X(0)).count(), 1);
        assert_eq!(graph.factors_for((X(0), X(1))).count(), 1);
        assert_eq!(graph.factors_for((X(1), X(0))).count(), 0);
    }

    #[test]
    fn factors_for_filters_by_component_type() {
        let mut graph = Graph::new();
        graph.add_factor(
            FactorBuilder::new(PriorResidual::new(VectorVar2::new(1.0, 2.0)), X(0))
                .noise(GaussianNoise::<2>::from_diag_sigmas(1.0, 2.0))
                .robust(GemanMcClure::default())
                .build(),
        );
        graph.add_factor(prior_x(0));

        assert_eq!(graph.factors_for(X(0)).count(), 2);
        assert_eq!(
            graph
                .factors_for(X(0))
                .residual::<PriorResidual<VectorVar2>>()
                .count(),
            2
        );
        assert_eq!(
            graph
                .factors_for(X(0))
                .noise::<GaussianNoise<2>>()
                .robust::<GemanMcClure>()
                .count(),
            1
        );
        assert_eq!(graph.factors_for(X(0)).noise::<UnitNoiseDyn>().count(), 1);
        assert_eq!(graph.factors_for(X(0)).robust::<L2>().count(), 1);
    }

    #[test]
    fn factors_for_mut_updates_matching_factors() {
        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));
        graph.add_factor(prior_x(1));

        for factor in graph
            .factors_for_mut(X(0))
            .residual::<PriorResidual<VectorVar2>>()
        {
            factor.set_robust(GemanMcClure::default());
        }

        assert_eq!(graph.factors_for(X(0)).robust::<GemanMcClure>().count(), 1);
        assert_eq!(graph.factors_for(X(1)).robust::<GemanMcClure>().count(), 0);
    }

    #[test]
    fn typed_residual_query_validates_key_input() {
        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));

        assert_eq!(
            graph
                .factors_for_residual::<PriorResidual<VectorVar2>, _>(X(0))
                .count(),
            1
        );
    }

    #[test]
    fn replacing_queried_factor_changes_structure_hash() {
        let mut values = values_x0();
        values.insert(X(1), VectorVar2::identity());

        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));
        let hash = graph.structure_hash(&values);

        for factor in graph.factors_for_mut(X(0)) {
            factor.replace(PriorResidual::new(VectorVar2::new(3.0, 4.0)), X(1));
        }

        assert_ne!(hash, graph.structure_hash(&values));
        assert_eq!(graph.factors_for(X(0)).count(), 0);
        assert_eq!(graph.factors_for(X(1)).count(), 1);
    }

    #[test]
    fn graph_order_is_graph_induced() {
        let mut values = values_x0();
        values.insert(Y(0), VectorVar3::identity());

        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));

        let graph_order = graph.sparsity_pattern(&values);

        assert_eq!(graph_order.order.len(), 1);
        assert_eq!(graph_order.order.dim(), 2);
        assert!(graph_order.order.get(X(0)).is_some());
        assert!(graph_order.order.get(Y(0)).is_none());
        assert_eq!(graph_order.structure_hash, graph.structure_hash(&values));
    }

    #[test]
    fn dense_normal_equations_match_linearized_graph() {
        let mut values = values_x0();
        values.insert(X(1), VectorVar2::new(0.5, -1.0));

        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));
        graph.add_factor(between_x(0, 1));

        let order = ValuesOrder::from_values(&values);
        let (direct_hessian, direct_rhs) = graph.dense_normal_equations(&values, &order);
        let (linear_hessian, linear_rhs) = graph.linearize(&values).dense_normal_equations(&order);

        assert!((&direct_hessian - &linear_hessian).norm() < 1e-12);
        assert!((&direct_rhs - &linear_rhs).norm() < 1e-12);
    }

    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    struct PanicResidual;

    #[factrs::mark]
    impl Residual for PanicResidual {
        type Input = VectorVar2;
        type Differ = ForwardProp;

        fn dim_out(&self) -> usize {
            2
        }

        fn residual<T: Numeric>(&self, _input: VectorVar2<T>) -> VectorX<T> {
            panic!("structure construction must not evaluate residuals")
        }
    }

    #[test]
    fn sparsity_pattern_uses_dim_out_without_evaluating_residual() {
        let values = values_x0();
        let mut graph = Graph::new();
        graph.add_factor(FactorBuilder::new(PanicResidual, X(0)).build());

        let graph_order = graph.sparsity_pattern(&values);

        assert_eq!(graph_order.order.dim(), 2);
    }

    #[test]
    fn structure_hash_ignores_unused_values_and_numeric_changes() {
        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));

        let values = values_x0();
        let hash = graph.structure_hash(&values);

        let mut with_unused = values.clone();
        with_unused.insert(Y(0), VectorVar3::new(1.0, 2.0, 3.0));
        assert_eq!(hash, graph.structure_hash(&with_unused));

        let mut with_new_x0 = values.clone();
        with_new_x0.insert(X(0), VectorVar2::new(10.0, 20.0));
        assert_eq!(hash, graph.structure_hash(&with_new_x0));
    }

    #[test]
    fn structure_hash_changes_with_factor_structure() {
        let mut values = values_x0();
        values.insert(X(1), VectorVar2::identity());

        let mut graph = Graph::new();
        graph.add_factor(prior_x(0));
        let hash = graph.structure_hash(&values);

        graph.add_factor(prior_x(1));
        assert_ne!(hash, graph.structure_hash(&values));

        let mut other_graph = Graph::new();
        other_graph.add_factor(prior_x(1));
        assert_ne!(hash, other_graph.structure_hash(&values));
    }
}
