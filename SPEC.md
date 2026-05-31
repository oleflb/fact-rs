# Residual And VarPack Overhaul

## Goal

Replace the fixed-arity residual API with a pack-based API.

The new API must support:

- one associated residual input type implementing `VarPack`
- typed variable packs, including tuples up to arity 6
- a concrete `DynVarPack` type built from runtime keys
- fixed-output and runtime-output residuals
- residual output dimensions inferred from residual evaluation
- updatable factors whose internal state and output dimension may change between optimizer runs
- dynamic Gaussian noise for runtime-sized residuals
- a single factor construction path

Breaking API changes are acceptable.

## Non-Goals For First Pass

- preserving `Residual1` through `Residual6`
- preserving `FactorBuilder::new1` through `new6`
- allowing factors to change keys during an optimizer run
- allowing factors to change residual row count during an optimizer run
- making dynamic residuals receive concrete typed variables through `DynVarPack`
- adding a large erased manifold API to `VariableSafe`

## Current Roadblocks

- `Residual1` through `Residual6` encode arity, variable types, input dimension, output dimension, and differentiator.
- `ForwardProp`, `NumericalDiff`, `FactorBuilder`, `fac!`, and `#[mark]` mirror the same fixed arity model.
- `Factor` stores `Box<dyn Residual>`, so a typed associated input cannot live directly on the erased storage trait.
- Current optimizers cache graph sparsity in `init()`, so row count and key structure must remain stable inside one `optimize()` call.
- `UnitNoiseDyn` exists, but runtime-sized Gaussian noise does not.

## Design Summary

Use two residual layers:

- `Residual`: typed authoring trait with `type Input: VarPack + DiffInput`
- `ErasedResidual`: object-safe trait stored by `Factor` and `Graph`

Use `VarPack` for residual input shape. Use a separate factor-input/key-pack conversion to produce the ordered keys stored by a factor.

Typed residuals use tuple packs:

```rust
type Input = (SE3, VectorVar3, ImuBias);
```

Dynamic residuals use `DynResidual` with a concrete key-owned pack:

```rust
fn residual(&self, values: &Values, input: &DynVarPack) -> VectorX;
```

`DynVarPack` is not a trait and is not generic over variable types. It is a concrete runtime input specification:

```rust
pub struct DynVarPack {
    keys: Vec<Key>,
}
```

Dynamic residuals receive `Values` and their `DynVarPack`, then index values by keys themselves. This keeps dynamic residuals simple, public, and open-world without adding erased typed-variable access.

## Core API Shape

### `VarPack`

`VarPack` describes the residual input shape. It should not own keys for typed tuple packs, because `type Input = (SE3, ImuBias)` is a type-level variable shape while `(X(0), B(0))` is the runtime key pack supplied during factor construction.

It must support:

- runtime input dimension from `Values`
- typed packing for tuple residuals
- dynamic key packs through `DynVarPack`

Sketch:

```rust
pub trait VarPack: Send + 'static {
    type Packed<T: Numeric>;

    fn dim_in(values: &Values, keys: &[Key]) -> Result<usize, ResidualError>;

    fn input(values: &Values, keys: &[Key]) -> Result<Self, ResidualError>
    where
        Self: Sized;

    fn pack<T: Numeric>(values: &Values, keys: &[Key]) -> Result<Self::Packed<T>, ResidualError>;
}
```

Expected implementations:

- one variable and tuples up to arity 6
- `DynVarPack`

For typed tuples, `VarPack::pack` downcasts variables from `Values` by ordered raw keys. Compile-time key validation happens before this through a separate factor-input trait.

### Factor Input Keys

Factor construction needs a trait that converts user-supplied keys into the ordered `Vec<Key>` stored by the erased factor.

Sketch:

```rust
pub trait FactorInput<P: VarPack> {
    fn into_keys(self) -> Result<Vec<Key>, ResidualError>;
}
```

Typed tuple implementations validate symbol types at compile time:

```rust
impl<K1, V1> FactorInput<V1> for K1
where
    K1: TypedSymbol<V1>,
    V1: VariableDtype;

impl<K1, K2, V1, V2> FactorInput<(V1, V2)> for (K1, K2)
where
    K1: TypedSymbol<V1>,
    K2: TypedSymbol<V2>,
    V1: VariableDtype,
    V2: VariableDtype;
```

Repeat tuple implementations up to arity 6.

`DynVarPack` implements `FactorInput<DynVarPack>` by returning its owned keys.

### `DynVarPack`

`DynVarPack` owns the dynamic keys. It is both the dynamic residual input type and the dynamic factor key pack.

Sketch:

```rust
#[derive(Clone, Debug)]
pub struct DynVarPack {
    keys: Vec<Key>,
}

impl DynVarPack {
    pub fn new<I, K>(keys: I) -> Result<Self, VarPackError>
    where
        I: IntoIterator<Item = K>,
        K: Into<Key>;

    pub fn keys(&self) -> &[Key];
}
```

Rules:

- preserve key order
- reject duplicate keys unless a concrete use case appears
- do not store global offsets
- do not store factor-local offsets in the public API
- do not store `entries` as part of the conceptual model

Dimension is queried through `Values` when needed:

```rust
values.get_raw(key).map(VariableSafe::dim)
```

`DynVarPack` also implements `VarPack` so builders can reuse the same input/key-pack machinery for dynamic factors. Dynamic residual authoring does not rely on generic typed packing; it receives `Values` and `&DynVarPack` directly.

### Typed `Residual`

`Residual` becomes the trait users implement for typed residuals.

Sketch:

```rust
pub trait Residual: Debug + Clone + Send + 'static {
    type Input: VarPack + DiffInput;
    type Differ: Diff<Self::Input>;

    fn residual<T: Numeric>(&self, input: <Self::Input as DiffInput>::Packed<T>) -> VectorX<T>;
}
```

Fixed-output residuals opt in with an additional marker:

```rust
pub trait FixedOutputDim {
    type DimOut: DimName;
}
```

Runtime-output residuals do not implement `FixedOutputDim`. Their output dimension is inferred from `residual(...).len()`.

### Dynamic Residuals

Dynamic residuals should not receive concrete packed variables. They receive values plus the dynamic key pack and index themselves.

Sketch:

```rust
pub trait DynResidual: Debug + Clone + Send + 'static {
    fn residual(&self, values: &Values, input: &DynVarPack) -> VectorX;

    fn residual_jacobian(
        &self,
        values: &Values,
        input: &DynVarPack,
    ) -> DiffResult<VectorX, MatrixX> {
        NumericalDiffDyn::jacobian(self, values, input)
    }
}
```

This intentionally means dynamic residuals are not generic over `T: Numeric` in v1. They can provide their own Jacobian or use dynamic numerical differentiation by default.

This is the key compromise that avoids a large erased manifold/autodiff redesign while still making `DynVarPack` public and usable for arbitrary runtime keys and variable types.

### Erased Residual Storage

`Factor` should store an object-safe erased trait.

Sketch:

```rust
pub trait ErasedResidual: Debug + DynClone + Send {
    fn keys(&self) -> &[Key];

    fn dim_in(&self, values: &Values) -> usize;

    fn dim_out(&self, values: &Values) -> usize;

    fn residual(&self, values: &Values) -> VectorX;

    fn residual_jacobian(&self, values: &Values) -> DiffResult<VectorX, MatrixX>;
}
```

Adapters should erase:

- typed `Residual` plus typed key pack
- `DynResidual` plus `DynVarPack`

## Factor API

Replace fixed-arity builders with one builder entry point.

Typed example:

```rust
let factor = FactorBuilder::new(residual, (X(0), V(0), B(0))).build();
```

Dynamic example:

```rust
let input = DynVarPack::new([X(0), X(1), X(2)])?;
let factor = FactorBuilder::new_dyn(residual, input).build();
```

Prefer one public `new` if trait bounds remain understandable. Use `new_dyn` only if Rust inference becomes unclear.

`Factor` changes:

```rust
pub struct Factor {
    residual: Box<dyn ErasedResidual>,
    noise: Box<dyn NoiseModel>,
    robust: Box<dyn RobustCost>,
}
```

`Factor::keys()` delegates to the erased residual or stores a cached copy.

`Factor::dim_out` becomes value-dependent:

```rust
pub fn dim_out(&self, values: &Values) -> usize;
```

## Noise API

Add `GaussianNoiseDyn`.

Requirements:

- runtime-sized covariance constructor
- runtime-sized information constructor
- runtime-sized sqrt-information constructor
- strict dimension checks during construction and whitening

`UnitNoiseDyn` should become length-agnostic if possible. If it keeps a stored dimension, validate it during whitening.

## Optimizer And Graph Constraints

Graph sparsity and row counts are computed from current `Values` before optimization.

Rules:

- factor keys must not change during one `optimize()` call
- factor output dimensions must not change during one `optimize()` call
- factor internal measurements/state may change between optimizer runs
- if dimensions change between runs, the optimizer must rebuild graph order/sparsity in `init()`

Graph APIs that need row counts must accept `Values`:

```rust
graph.sparsity_pattern(values, order)
factor.dim_out(values)
```

## Error Handling

Introduce explicit errors for build/evaluation paths touched by the overhaul.

Recommended error categories:

- missing key
- wrong variable type for typed pack
- duplicate key in `DynVarPack`
- noise dimension mismatch
- residual/noise dimension mismatch
- jacobian shape mismatch

Use panics only for internal invariants that cannot be caused by user input.

## Proc Macro Changes

Update `#[factrs::mark]`:

- no longer parse `Residual1` through `Residual6`
- support the new typed `Residual` trait
- generate `ErasedResidual` adapter registration for serialization if needed

Update `fac!`:

- call the new builder
- preserve existing ergonomic syntax where practical
- remove hard-coded arity dispatch

## Migration Plan

### Phase 1: Core Types

- Replace `src/residuals/var_pack.rs` with the new `VarPack` and `DynVarPack` definitions.
- Add tuple/key-pack support up to arity 6.
- Add residual errors.
- Add focused tests for `DynVarPack` construction and duplicate rejection.

### Phase 2: Residual Traits And Adapters

- Add typed `Residual` trait.
- Add object-safe `ErasedResidual` trait.
- Add typed-erased adapter.
- Add dynamic residual adapter.
- Keep old residual traits temporarily only if needed to migrate incrementally inside one branch.

### Phase 3: Differentiation

- Add pack-based differentiation for typed tuple packs.
- Add dynamic numerical differentiation for `DynResidual` default Jacobians.
- Refactor `ForwardProp` and `NumericalDiff` behind the pack-based `Diff<I>` API.

### Phase 4: Factor And Graph

- Replace `FactorBuilder::new1` through `new6` with the new builder.
- Make factor/graph row-count APIs value-dependent.
- Update linearization and sparsity pattern assembly.
- Add runtime shape validation around residual, Jacobian, and noise.

### Phase 5: Noise

- Add `GaussianNoiseDyn`.
- Adjust `UnitNoiseDyn` behavior and validation.
- Add shape-focused tests.

### Phase 6: Built-In Residuals

- Port `PriorResidual`.
- Port `BetweenResidual`.
- Port IMU preintegration residual.
- Update helper constructors that build factors.

### Phase 7: Macros And Serialization

- Update `fac!`.
- Update `#[mark]`.
- Update serde tests for boxed residuals.

### Phase 8: Cleanup

- Remove `Residual1` through `Residual6`.
- Remove old fixed-arity builder methods.
- Update docs/examples/README snippets.
- Run full test suite.

## Test Plan

Tests must be small, focused, and orthogonal.

### VarPack Tests

- `dyn_var_pack_preserves_key_order`
- `dyn_var_pack_rejects_duplicate_keys`
- `typed_tuple_pack_reports_keys`
- `typed_tuple_pack_dim_in_sums_value_dims`
- `typed_tuple_pack_rejects_wrong_variable_type`

### Residual Adapter Tests

- `typed_residual_erases_and_evaluates`
- `typed_residual_erases_and_linearizes`
- `runtime_output_dim_is_inferred_from_residual`
- `fixed_output_dim_marker_reports_static_dim`

### Dynamic Residual Tests

- `dyn_residual_receives_values_and_key_pack`
- `dyn_residual_can_index_values_by_keys`
- `dyn_residual_default_numerical_jacobian_has_expected_shape`
- `dyn_residual_runtime_output_dim_changes_between_runs`

### Factor Builder Tests

- `factor_builder_accepts_typed_tuple_keys`
- `factor_builder_rejects_typed_tuple_wrong_key_type`
- `factor_builder_accepts_dyn_var_pack`
- `factor_linearize_validates_jacobian_shape`
- `factor_error_validates_noise_dimension`

### Noise Tests

- `gaussian_noise_dyn_from_cov_whitens_vector`
- `gaussian_noise_dyn_rejects_non_square_matrix`
- `gaussian_noise_dyn_rejects_dimension_mismatch`
- `unit_noise_dyn_accepts_runtime_length`

### Graph And Optimizer Tests

- `graph_sparsity_uses_value_dependent_dim_out`
- `optimizer_rebuilds_sparsity_on_new_optimize_call`
- `factor_dimension_change_between_optimize_calls_is_allowed`
- `factor_dimension_change_during_optimize_call_is_rejected_or_documented`

### Migration Regression Tests

- `prior_residual_optimizes`
- `between_residual_optimizes`
- `imu_preintegration_residual_linearizes`
- `fac_macro_builds_unary_factor`
- `fac_macro_builds_tuple_factor`
- `serde_roundtrip_boxed_residual`

## Open Implementation Choices

These should be settled while coding, based on Rust ergonomics:

- whether dynamic factor construction can share `FactorBuilder::new` or needs `FactorBuilder::new_dyn`
- whether typed tuple keys need a public wrapper trait or can remain internal
- whether old residual traits are removed immediately or after all built-ins are ported in the same branch
- whether `UnitNoiseDyn` should store dimension or be stateless

## Acceptance Criteria

- no public `Residual1` through `Residual6` authoring path remains
- all built-in residuals use `type Input: VarPack + DiffInput`
- `DynVarPack` is a concrete public type with owned `Vec<Key>`
- dynamic residuals receive `Values` and `&DynVarPack`
- output dimension is value-dependent or inferred from residual output
- runtime-sized Gaussian noise exists and is tested
- factor construction has one primary API path
- tests cover typed packs, dynamic packs, fixed output, runtime output, noise, factors, graph sparsity, macros, and migrated built-ins
