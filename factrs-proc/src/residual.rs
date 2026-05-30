use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_quote, ItemImpl};

fn residual_trait_name(item: &ItemImpl) -> syn::Result<String> {
    let err = syn::Error::new_spanned(item, "unable to parse residual trait");
    Ok(item
        .trait_
        .clone()
        .ok_or(err.clone())?
        .1
        .segments
        .last()
        .ok_or(err)?
        .ident
        .to_string())
}

pub fn mark(mut item: ItemImpl) -> TokenStream2 {
    let trait_name = match residual_trait_name(&item) {
        Result::Err(e) => return e.to_compile_error(),
        Result::Ok(n) => n,
    };

    let typetag = if cfg!(feature = "serde") {
        let all_type_params: Vec<_> = item.generics.type_params().cloned().collect();
        for type_param in all_type_params {
            let ident = &type_param.ident;
            item.generics.make_where_clause();
            item.generics
                .where_clause
                .as_mut()
                .unwrap()
                .predicates
                .push(parse_quote!(#ident: typetag::Tagged));
        }

        quote!( #[typetag::serde] )
    } else {
        TokenStream2::new()
    };

    let generics = &item.generics;
    let self_ty = &item.self_ty;
    let where_clause = &generics.where_clause;

    match trait_name.as_str() {
        "Residual" => quote! {
            #item

            #typetag
            impl #generics factrs::residuals::ErasedResidual for #self_ty #where_clause {
                fn dim_in(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<usize, factrs::residuals::ResidualError> {
                    <<Self as factrs::residuals::Residual>::Input as factrs::residuals::VarPack>::dim_in(values, keys)
                }

                fn dim_out(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<usize, factrs::residuals::ResidualError> {
                    let input = <<Self as factrs::residuals::Residual>::Input as factrs::residuals::VarPack>::pack::<factrs::dtype>(values, keys)?;
                    Ok(factrs::residuals::Residual::residual::<factrs::dtype>(self, input).len())
                }

                fn residual(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<factrs::linalg::VectorX, factrs::residuals::ResidualError> {
                    let input = <<Self as factrs::residuals::Residual>::Input as factrs::residuals::VarPack>::pack::<factrs::dtype>(values, keys)?;
                    Ok(factrs::residuals::Residual::residual::<factrs::dtype>(self, input))
                }

                fn residual_jacobian(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<factrs::linalg::DiffResult<factrs::linalg::VectorX, factrs::linalg::MatrixX>, factrs::residuals::ResidualError> {
                    <<Self as factrs::residuals::Residual>::Differ as factrs::residuals::DiffPack<<Self as factrs::residuals::Residual>::Input>>::jacobian(self, values, keys)
                }
            }
        },
        "DynResidual" => quote! {
            #item

            #typetag
            impl #generics factrs::residuals::ErasedResidual for #self_ty #where_clause {
                fn dim_in(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<usize, factrs::residuals::ResidualError> {
                    <factrs::residuals::DynVarPack as factrs::residuals::VarPack>::dim_in(values, keys)
                }

                fn dim_out(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<usize, factrs::residuals::ResidualError> {
                    let input = factrs::residuals::DynVarPack::new(keys.to_vec())?;
                    Ok(factrs::residuals::DynResidual::residual(self, values, &input).len())
                }

                fn residual(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<factrs::linalg::VectorX, factrs::residuals::ResidualError> {
                    let input = factrs::residuals::DynVarPack::new(keys.to_vec())?;
                    Ok(factrs::residuals::DynResidual::residual(self, values, &input))
                }

                fn residual_jacobian(
                    &self,
                    values: &factrs::containers::Values,
                    keys: &[factrs::containers::Key],
                ) -> Result<factrs::linalg::DiffResult<factrs::linalg::VectorX, factrs::linalg::MatrixX>, factrs::residuals::ResidualError> {
                    let input = factrs::residuals::DynVarPack::new(keys.to_vec())?;
                    Ok(factrs::residuals::DynResidual::residual_jacobian(self, values, &input))
                }
            }
        },
        _ => syn::Error::new_spanned(item, "expected Residual or DynResidual impl")
            .to_compile_error(),
    }
}
