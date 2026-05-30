#[cfg(feature = "serde")]
mod ser_de {
    use factrs::{
        containers::Values, residuals::PriorResidual, symbols::X, traits::ErasedResidual,
        variables::VectorVar1,
    };

    #[test]
    fn test_vector_serialize() {
        let trait_object = &PriorResidual::new(VectorVar1::new(2.3)) as &dyn ErasedResidual;
        let json = serde_json::to_string(trait_object).unwrap();
        let expected = r#"{"tag":"PriorResidual<VectorVar<1>>","prior":[2.3]}"#;
        println!("json: {json}");
        assert_eq!(json, expected);
    }

    #[test]
    fn test_vector() {
        let json = r#"{"tag":"PriorResidual<VectorVar<1>>","prior":[1.2]}"#;
        let trait_object: Box<dyn ErasedResidual> = serde_json::from_str(json).unwrap();

        let mut values = Values::new();
        values.insert_unchecked(X(0), VectorVar1::new(1.2));
        let keys = [X(0).into()];
        let error = trait_object.residual(&values, &keys).unwrap()[0];

        assert_eq!(trait_object.dim_in(&values, &keys).unwrap(), 1);
        assert_eq!(trait_object.dim_out(&values, &keys).unwrap(), 1);
        assert_eq!(error, 0.0);
    }
}
