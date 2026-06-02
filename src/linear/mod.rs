//! Structs & traits for solving linear factor graphs

mod factor;
pub use factor::LinearFactor;

mod graph;
pub(crate) use graph::accumulate_dense_normal_factor;
pub use graph::{LinearGraph, OrderedLinearGraph};

mod values;
pub use values::LinearValues;

mod solvers;
pub use solvers::{CholeskySolver, LUSolver, LinearSolver, QRSolver};
