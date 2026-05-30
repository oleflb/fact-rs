use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{
    parse::Parse, parse_quote, punctuated::Punctuated, spanned::Spanned, Expr, ExprCast, Ident,
    Token,
};

pub struct Factor {
    residual: Expr,
    keys: Expr,
    noise: Option<Expr>,
    robust: Option<Expr>,
}

impl Factor {
    fn noise_call(&self) -> TokenStream2 {
        match &self.noise {
            Some(n) => quote! { .noise(#n) },
            None => TokenStream2::new(),
        }
    }

    fn robust_call(&self) -> TokenStream2 {
        match &self.robust {
            Some(r) => quote! {.robust(#r) },
            None => TokenStream2::new(),
        }
    }

    fn new_call(&self) -> TokenStream2 {
        let res = &self.residual;
        let keys = &self.keys;
        quote! { new(#res, #keys) }
    }
}

impl Parse for Factor {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let input = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;

        // Make sure we go the right number of arguments
        if input.len() < 2 {
            return Err(syn::Error::new_spanned(
                &input[0],
                "Expected at least two items",
            ));
        } else if input.len() > 4 {
            return Err(syn::Error::new_spanned(
                input.last(),
                "Expected at most four items",
            ));
        }

        // Residual is first
        let residual = input[0].clone();

        // Then the keys / input pack
        let keys = input[1].clone();

        // Then the noise
        let noise = if input.len() >= 3 {
            let m = quote!(factrs::noise);
            match &input[2] {
                Expr::Cast(ExprCast { expr, ty, .. }) => {
                    // Make sure it's a cov or std cast
                    let ty = match ty.to_token_stream().to_string().as_str() {
                        "cov" => Ident::new("cov", ty.span()),
                        "std" | "sigma" | "sig" => Ident::new("sigma", ty.span()),
                        _ => return Err(syn::Error::new_spanned(ty, "Unknown cast for noise")),
                    };

                    // Check if it's a tuple or a single variable
                    match expr.as_ref() {
                        Expr::Tuple(t) => {
                            if t.elems.len() != 2 {
                                return Err(syn::Error::new_spanned(
                                    t,
                                    "Expected tuple with two elements for split std/cov",
                                ));
                            }
                            let (a, b) = (&t.elems[0], &t.elems[1]);
                            let func = format_ident!("from_split_{}", ty);
                            Some(parse_quote!(#m::GaussianNoiseDyn::#func(#a, #b)))
                        }
                        _ => {
                            let func = format_ident!("from_scalar_{}", ty);
                            Some(parse_quote!(#m::GaussianNoiseDyn::#func(#expr)))
                        }
                    }
                }
                Expr::Infer(_) => Some(parse_quote!(#m::UnitNoiseDyn::default())),
                _ => Some(input[2].clone()),
            }
        } else {
            None
        };

        // Finally robust cost function
        let robust = if input.len() == 4 {
            Some(input[3].clone())
        } else {
            None
        };

        Ok(Factor {
            residual,
            keys,
            noise,
            robust,
        })
    }
}

pub fn fac(factor: Factor) -> TokenStream2 {
    let call = factor.new_call();
    let noise = factor.noise_call();
    let robust = factor.robust_call();

    let out = quote! {
        factrs::containers::FactorBuilder:: #call #noise #robust.build()
    };

    out
}
