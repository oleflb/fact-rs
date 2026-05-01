use factrs::core::{Factor, GaussNewton, Graph};

fn assert_send<T: Send>() {}

#[test]
fn graph_is_send() {
    assert_send::<Graph>();
}

#[test]
fn factor_is_send() {
    assert_send::<Factor>();
}

#[test]
fn factor_is_send() {
    assert_send::<GaussNewton>();
}
