use faer_ext::IntoNalgebra;

use super::{BaseOptParams, OptError, OptObserverVec, OptResult, Optimizer};
use crate::{
    containers::{Graph, GraphOrder, GraphStructureHash, Values, ValuesOrder},
    dtype,
    linalg::DiffResult,
    linear::{LinearSolver, LinearValues},
};

/// The Gauss-Newton optimizer
///
/// Solves $A \Delta \Theta = b$ directly for each optimizer steps. It defaults
/// to using [CholeskySolver](crate::linear::CholeskySolver) under the hood, but
/// this can be changed using [set_solver](GaussNewton::set_solver). See
/// the [linear](crate::linear) module for more linear solver options.
pub struct GaussNewton {
    graph: Graph,
    solver: Box<dyn LinearSolver>,
    /// Basic parameters for the optimizer
    params: BaseOptParams,
    /// Observers for the optimizer
    observers: OptObserverVec,
    // For caching computation between steps
    graph_order: Option<GraphOrder>,
    dense_order: Option<(GraphStructureHash, ValuesOrder)>,
    use_dense_normal_equations: bool,
}

impl GaussNewton {
    /// Sets the linear solver to use for the optimizer.
    pub fn set_solver(&mut self, solver: impl LinearSolver + 'static) {
        self.solver = Box::new(solver);
        self.use_dense_normal_equations = false;
    }

    /// Uses dense normal equations for small systems instead of the configured sparse solver.
    pub fn set_dense_normal_equations(&mut self, enabled: bool) {
        self.use_dense_normal_equations = enabled;
    }
}

impl Optimizer for GaussNewton {
    type Params = BaseOptParams;

    fn new(params: Self::Params, graph: Graph) -> Self {
        Self {
            graph,
            solver: Default::default(),
            observers: OptObserverVec::default(),
            params,
            graph_order: None,
            dense_order: None,
            use_dense_normal_equations: false,
        }
    }

    fn observers(&self) -> &OptObserverVec {
        &self.observers
    }

    fn observers_mut(&mut self) -> &mut OptObserverVec {
        &mut self.observers
    }

    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn error(&self, values: &Values) -> dtype {
        self.graph.error(values)
    }

    fn params(&self) -> &BaseOptParams {
        &self.params
    }

    fn init(&mut self, values: &Values) -> Vec<&'static str> {
        const DENSE_NORMAL_EQUATION_MAX_DIM: usize = 512;

        let structure = self.graph.structure(values);
        if self.use_dense_normal_equations && structure.total_cols <= DENSE_NORMAL_EQUATION_MAX_DIM
        {
            let rebuild = self
                .dense_order
                .as_ref()
                .is_none_or(|(hash, _)| *hash != structure.hash);

            if rebuild {
                self.dense_order = Some((structure.hash, structure.order));
                self.graph_order = None;
            }

            return Vec::new();
        }

        self.dense_order = None;
        let rebuild = self
            .graph_order
            .as_ref()
            .is_none_or(|order| order.structure_hash != structure.hash);

        if rebuild {
            self.graph_order = Some(self.graph.sparsity_pattern_from_structure(structure));
            self.solver.reset_symbolic();
        }

        Vec::new()
    }

    fn step(&mut self, mut values: Values, _idx: usize) -> OptResult<(Values, String)> {
        // Solve the linear system
        let delta = if let Some((_, order)) = &self.dense_order {
            let (hessian, rhs) = self.graph.dense_normal_equations(&values, order);
            if let Some(cholesky) = hessian.clone().cholesky() {
                cholesky.solve(&rhs)
            } else {
                hessian.lu().solve(&rhs).ok_or(OptError::InvalidSystem)?
            }
        } else {
            let linear_graph = self.graph.linearize(&values);
            let ordered =
                linear_graph.with_order(self.graph_order.as_ref().expect("Missing graph order"));
            let DiffResult { value: r, diff: j } = ordered.residual_jacobian();

            // Solve Ax = b
            self.solver
                .solve_lst_sq(j.as_ref(), r.as_ref())
                .as_ref()
                .into_nalgebra()
                .column(0)
                .clone_owned()
        };

        // Update the values
        let dx = LinearValues::from_order_and_vector(
            self.dense_order
                .as_ref()
                .map(|(_, order)| order.clone())
                .or_else(|| self.graph_order.as_ref().map(|order| order.order.clone()))
                .expect("Missing graph order"),
            delta,
        );
        values.oplus_mut(&dx);

        Ok((values, String::new()))
    }
}

#[cfg(test)]
mod test {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use faer::{Mat, MatRef, sparse::SparseColMatRef};

    use super::*;
    use crate::{
        assign_symbols,
        containers::{FactorBuilder, Values},
        linear::CholeskySolver,
        residuals::PriorResidual,
        test_optimizer,
        variables::{Variable, VectorVar3},
    };

    test_optimizer!(GaussNewton);

    assign_symbols!(X: VectorVar3);

    struct CountingSolver {
        inner: CholeskySolver,
        resets: Arc<AtomicUsize>,
    }

    impl LinearSolver for CountingSolver {
        fn solve_symmetric(
            &mut self,
            a: SparseColMatRef<usize, dtype>,
            b: MatRef<dtype>,
        ) -> Mat<dtype> {
            self.inner.solve_symmetric(a, b)
        }

        fn solve_lst_sq(
            &mut self,
            a: SparseColMatRef<usize, dtype>,
            b: MatRef<dtype>,
        ) -> Mat<dtype> {
            self.inner.solve_lst_sq(a, b)
        }

        fn reset_symbolic(&mut self) {
            self.resets.fetch_add(1, Ordering::SeqCst);
            self.inner.reset_symbolic();
        }
    }

    #[test]
    fn optimize_reuses_order_until_graph_structure_changes() {
        let resets = Arc::new(AtomicUsize::new(0));

        let mut graph = Graph::new();
        graph.add_factor(
            FactorBuilder::new(PriorResidual::new(VectorVar3::new(1.0, 0.0, 0.0)), X(0)).build(),
        );

        let mut values = Values::new();
        values.insert(X(0), VectorVar3::identity());
        values.insert(X(1), VectorVar3::identity());

        let mut opt = GaussNewton::new_default(graph);
        opt.set_solver(CountingSolver {
            inner: CholeskySolver::default(),
            resets: resets.clone(),
        });

        let values = opt.optimize(values).expect("first optimization succeeds");
        assert_eq!(resets.load(Ordering::SeqCst), 1);

        let values = opt
            .optimize(values)
            .expect("unchanged graph still succeeds");
        assert_eq!(resets.load(Ordering::SeqCst), 1);

        opt.graph_mut().add_factor(
            FactorBuilder::new(PriorResidual::new(VectorVar3::new(0.0, 1.0, 0.0)), X(1)).build(),
        );

        opt.optimize(values)
            .expect("changed graph rebuilds and succeeds");
        assert_eq!(resets.load(Ordering::SeqCst), 2);
    }
}
