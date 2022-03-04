use std::{fmt::Display, cmp::Ordering};
use std::cmp::PartialOrd;
use num_traits::{Float, AsPrimitive};
use num_complex::Complex;

pub mod polynomial;

pub trait ScientificNotation {
    fn to_scientific_notation(&self, prec: usize) -> String;
}

impl<T> ScientificNotation for T
where 
    T: Float + Display + AsPrimitive<i32>
{
    fn to_scientific_notation(&self, prec: usize) -> String {
        let sig = self.log10().floor().as_();

        let mul = T::from(10usize).unwrap().powi(sig);

        format!("{:.1$}E{2}", *self / mul, prec, sig)
    }
}

pub trait EngineeringNotation {
    fn to_engineering_notation(&self, prec: usize) -> String;
}

impl<T> EngineeringNotation for T
where 
    T: Float + Display + AsPrimitive<i32>
{
    fn to_engineering_notation(&self, prec: usize) -> String {
        // log10(101) ~ 3.001
        let sig = (self.log10().floor().as_()) / 3 * 3;

        let mul = T::from(10usize).unwrap().powi(-sig);

        format!("{0:.1$}E{2}", *self * mul, prec, sig)
    }
}

pub trait CloseEnough {
    fn close_enough(&self, other: &Self, tol: f64) -> bool;
}

impl CloseEnough for f64 {
    fn close_enough(&self, other: &Self, tol: f64) -> bool {
        let diff = (*self - *other).abs();
        diff < tol
    }
}

impl<T> CloseEnough for Complex<T>
where T: Float, f64: Into<T> {
    fn close_enough(&self, other: &Self, tol: f64) -> bool {
        let diff = (self.re - other.re).abs();
        let diff2 = (self.im - other.im).abs();
        diff < tol.into() && diff2 < tol.into()
    }
}

impl<T> CloseEnough for [T]
where T: CloseEnough {
    fn close_enough(&self, other: &Self, tol: f64) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for i in 0..self.len() {
            if !self[i].close_enough(&other[i], tol) {
                return false;
            }
        }

        true
    }
}

macro_rules! assert_close {
    ($a:expr, $b:expr, $tol:expr) => {
        if !($a.close_enough(&$b, $tol)) {
            panic!("assertion failed: `(left == right)` \
                (left: `{:?}`, right: `{:?}`, tolerance: `{:?}`)",
                $a, $b, $tol);
        }
    };
    // Variation where tol is defaulted to 1e-5
    ($a:expr, $b:expr) => {
        assert_close!($a, $b, 1e-5);
    };
}

pub trait Sortable {
    fn comes_before(&self, other: &Self) -> Ordering;
}

impl Sortable for f64 {
    fn comes_before(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl Sortable for f32 {
    fn comes_before(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<T: Float> Sortable for Complex<T> {
    fn comes_before(&self, other: &Self) -> Ordering {
        if self.re.round() == other.re.round() {
            self.im.round().partial_cmp(&other.im.round()).unwrap_or(Ordering::Equal)
        } else {
            self.re.round().partial_cmp(&other.re.round()).unwrap_or(Ordering::Equal)
        }
    }
}

macro_rules! assert_close_no {
    ($a: expr, $b: expr, $tol: expr) => {
        let mut a = $a.clone();
        a.sort_by(Sortable::comes_before);
        let mut b = $b.clone();
        b.sort_by(Sortable::comes_before);
        assert_close!(a, b, $tol);
    };
    // Variation where tol is defaulted to 1e-5
    ($a: expr, $b: expr) => {
        assert_close_no!($a, $b, 1e-5);
    };
}

pub(crate) use assert_close;
pub(crate) use assert_close_no;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scientific_notation_works() {
        let s = (91273.27).to_scientific_notation(3);
        assert_eq!(s, "9.127E4");
    }

    #[test]
    fn engineering_notation_works() {
        let s = (91273.27).to_engineering_notation(3);
        assert_eq!(s, "91.273E3");
    }

    #[test]
    fn close_enough_works() {
        assert!((1.0).close_enough(&0.95, 0.1));
        assert!(
            Complex::new(0.0, 1.0)
                .close_enough(&Complex::new(0.01, 0.99), 0.1)
        );

        assert!(
            vec![1.0, 2.0, 3.0].close_enough(
                &vec![1.05, 2.05, 2.95],
                0.1
            )
        )
    }

    #[test]
    fn assert_close_works() {
        assert_close!(1.0, 0.95, 0.1);
        assert_close!(1.0, 0.9999999);
    }

    #[test]
    #[should_panic]
    fn assert_close_panic_1() {
        assert_close!(1.0, 0.95, 0.01);
    }

    #[test]
    #[should_panic]
    fn assert_close_panic_2() {
        assert_close!(1.0, 0.95);
    }
}