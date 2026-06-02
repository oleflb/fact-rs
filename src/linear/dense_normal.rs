use std::ops::AddAssign;

use crate::{
    containers::ValuesOrder,
    linalg::{MatrixX, VectorX},
    linear::LinearFactor,
};

pub(crate) fn accumulate_dense_normal_factor(
    factor: &LinearFactor,
    order: &ValuesOrder,
    hessian: &mut MatrixX,
    rhs: &mut VectorX,
) {
    for (i, key_i) in factor.keys.iter().enumerate() {
        let idx_i = order.get(*key_i).expect("Key missing in values");
        let a_i = factor.a.get_block(i);
        let a_i_t_b = a_i.transpose() * &factor.b;
        rhs.rows_mut(idx_i.idx, idx_i.dim).add_assign(&a_i_t_b);

        for (j, key_j) in factor.keys.iter().take(i + 1).enumerate() {
            let idx_j = order.get(*key_j).expect("Key missing in values");
            let a_j = factor.a.get_block(j);
            let block = a_i.transpose() * a_j;

            hessian
                .view_mut((idx_i.idx, idx_j.idx), (idx_i.dim, idx_j.dim))
                .add_assign(&block);
            if i != j {
                hessian
                    .view_mut((idx_j.idx, idx_i.idx), (idx_j.dim, idx_i.dim))
                    .add_assign(&block.transpose());
            }
        }
    }
}
