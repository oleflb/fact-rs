//! Structs & traits for solving linear factor graphs

mod factor;
pub use factor::LinearFactor;

pub(crate) mod dense_normal;

mod graph;
pub use graph::{LinearGraph, OrderedLinearGraph};

mod values;
pub use values::LinearValues;

mod solvers;
pub use solvers::{CholeskySolver, LUSolver, LinearSolver, QRSolver};
