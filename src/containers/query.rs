use std::marker::PhantomData;

use crate::{containers::Factor, noise::NoiseModel, residuals::ErasedResidual, robust::RobustCost};

pub trait FactorQuery<'a>: Iterator<Item = &'a Factor> + Sized {
    fn residual<R>(self) -> ResidualFilter<Self, R>
    where
        R: ErasedResidual + 'static,
    {
        ResidualFilter::new(self)
    }

    fn noise<N>(self) -> NoiseFilter<Self, N>
    where
        N: NoiseModel + 'static,
    {
        NoiseFilter::new(self)
    }

    fn robust<C>(self) -> RobustFilter<Self, C>
    where
        C: RobustCost + 'static,
    {
        RobustFilter::new(self)
    }
}

impl<'a, I> FactorQuery<'a> for I where I: Iterator<Item = &'a Factor> + Sized {}

pub trait FactorQueryMut<'a>: Iterator<Item = &'a mut Factor> + Sized {
    fn residual<R>(self) -> ResidualFilterMut<Self, R>
    where
        R: ErasedResidual + 'static,
    {
        ResidualFilterMut::new(self)
    }

    fn noise<N>(self) -> NoiseFilterMut<Self, N>
    where
        N: NoiseModel + 'static,
    {
        NoiseFilterMut::new(self)
    }

    fn robust<C>(self) -> RobustFilterMut<Self, C>
    where
        C: RobustCost + 'static,
    {
        RobustFilterMut::new(self)
    }
}

impl<'a, I> FactorQueryMut<'a> for I where I: Iterator<Item = &'a mut Factor> + Sized {}

pub struct ResidualFilter<I, R> {
    iter: I,
    _residual: PhantomData<R>,
}

impl<I, R> ResidualFilter<I, R> {
    fn new(iter: I) -> Self {
        Self {
            iter,
            _residual: PhantomData,
        }
    }
}

impl<'a, I, R> Iterator for ResidualFilter<I, R>
where
    I: Iterator<Item = &'a Factor>,
    R: ErasedResidual + 'static,
{
    type Item = &'a Factor;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|factor| factor.is_residual::<R>())
    }
}

pub struct NoiseFilter<I, N> {
    iter: I,
    _noise: PhantomData<N>,
}

impl<I, N> NoiseFilter<I, N> {
    fn new(iter: I) -> Self {
        Self {
            iter,
            _noise: PhantomData,
        }
    }
}

impl<'a, I, N> Iterator for NoiseFilter<I, N>
where
    I: Iterator<Item = &'a Factor>,
    N: NoiseModel + 'static,
{
    type Item = &'a Factor;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|factor| factor.is_noise::<N>())
    }
}

pub struct RobustFilter<I, C> {
    iter: I,
    _robust: PhantomData<C>,
}

impl<I, C> RobustFilter<I, C> {
    fn new(iter: I) -> Self {
        Self {
            iter,
            _robust: PhantomData,
        }
    }
}

impl<'a, I, C> Iterator for RobustFilter<I, C>
where
    I: Iterator<Item = &'a Factor>,
    C: RobustCost + 'static,
{
    type Item = &'a Factor;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|factor| factor.is_robust::<C>())
    }
}

pub struct ResidualFilterMut<I, R> {
    iter: I,
    _residual: PhantomData<R>,
}

impl<I, R> ResidualFilterMut<I, R> {
    fn new(iter: I) -> Self {
        Self {
            iter,
            _residual: PhantomData,
        }
    }
}

impl<'a, I, R> Iterator for ResidualFilterMut<I, R>
where
    I: Iterator<Item = &'a mut Factor>,
    R: ErasedResidual + 'static,
{
    type Item = &'a mut Factor;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|factor| factor.is_residual::<R>())
    }
}

pub struct NoiseFilterMut<I, N> {
    iter: I,
    _noise: PhantomData<N>,
}

impl<I, N> NoiseFilterMut<I, N> {
    fn new(iter: I) -> Self {
        Self {
            iter,
            _noise: PhantomData,
        }
    }
}

impl<'a, I, N> Iterator for NoiseFilterMut<I, N>
where
    I: Iterator<Item = &'a mut Factor>,
    N: NoiseModel + 'static,
{
    type Item = &'a mut Factor;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|factor| factor.is_noise::<N>())
    }
}

pub struct RobustFilterMut<I, C> {
    iter: I,
    _robust: PhantomData<C>,
}

impl<I, C> RobustFilterMut<I, C> {
    fn new(iter: I) -> Self {
        Self {
            iter,
            _robust: PhantomData,
        }
    }
}

impl<'a, I, C> Iterator for RobustFilterMut<I, C>
where
    I: Iterator<Item = &'a mut Factor>,
    C: RobustCost + 'static,
{
    type Item = &'a mut Factor;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find(|factor| factor.is_robust::<C>())
    }
}
