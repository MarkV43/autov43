use std::fmt::Display;
use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign};
use num_complex::Complex;
use num_traits::{Num, Pow, AsPrimitive, pow, NumAssign};
use float_pretty_print::PrettyPrintFloat;
use ndarray::Array2;
use ndarray_linalg::eig::EigVals;

#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    coeffs: Vec<f64>,
}

#[allow(dead_code)]
impl Polynomial {
    fn trimseq(seq: &[f64]) -> Vec<f64> {
        if seq.len() <= 0 {
            seq.to_vec()
        } else {
            let mut i = seq.len() - 1;
            while i > 0 && seq[i] == 0.0 {
                i -= 1;
            }
            seq[0..i+1].to_vec()
        }
    }

    pub fn new<T, U>(coeffs: U) -> Polynomial
    where 
        f64: From<T>,
        T: Copy,
        U: IntoIterator<Item = T>
    {
        let mut coeffs: Vec<_> = coeffs.into_iter().map(|x| f64::from(x)).collect();
        coeffs = Polynomial::trimseq(&coeffs);

        assert!(coeffs.len() > 0);

        assert!(coeffs.iter().all(|x| x.is_finite()));

        Polynomial { 
            coeffs,
        }
    }

    pub fn from_rev_coeffs<T>(coeffs: &[T]) -> Polynomial
    where f64: From<T>, T: Copy {
        Polynomial::new(coeffs.iter().rev().map(|&x| x).collect::<Vec<_>>())
    }

    pub fn from_roots<T>(roots: &[T], gain: f64) -> Polynomial
    where T: Num + Copy + Mul<T> + Neg<Output = T> + Mul<f64> + From<f64> + AddAssign<T> + AsPrimitive<f64> + Copy {
        let mut coeffs = vec![T::from(0.0); roots.len() + 1];
        let mut tmp = vec![T::from(0.0); roots.len() + 1];
        coeffs[0] = T::from(gain);
        for (k, r) in roots.iter().enumerate() {
            tmp.fill(T::from(0.0));
            for i in 0..=k {
                let val = coeffs[i];
                tmp[i] += -((*r) * val);
                tmp[i + 1] += val;
            }
            coeffs.copy_from_slice(&tmp[..]);
        }
        Polynomial::new(coeffs.iter().map(|x| (*x).as_()).collect::<Vec<_>>())
    }

    pub fn from_root<T>(root: T, gain: f64) -> Polynomial
    where T: Num + Copy + Mul<T> + AsPrimitive<f64> {
        let coeffs = vec![-root.as_()*gain, gain];
        Polynomial::new(coeffs)
    }

    pub fn from_complex_root_pair<T>(root: Complex<T>, gain: f64) -> Polynomial
    where
        T: Mul<T, Output = T> + Add<T, Output = T> + Mul<f64, Output = T> + From<f64> + Copy,
        f64: From<T> + Mul<T, Output = T>
    {
        let a = root.re;
        let b = root.im;
        Polynomial::new(vec![(a*a + b*b)*gain, -2.0*a*gain, T::from(gain)])
    }

    pub fn evaluate<T>(&self, x: T) -> T
    where T: Num + Pow<i32, Output=T> + From<f64> + std::ops::AddAssign + Copy {
        let mut result = T::from(0.0);
        for (i, coeff) in self.coeffs.iter().enumerate() {
            result += T::from(*coeff) * x.pow(i as i32);
        }
        result
    }

    pub fn evaluate_complex<T>(&self, x: Complex<T>) -> Complex<T>
    where 
        T: Clone + Num + Neg<Output = T> + NumAssign,
        Complex<T>: From<f64>
    {
        let mut result = Complex::<T>::from(0.0);
        for (i, coeff) in self.coeffs.iter().enumerate() {
            result += Complex::<T>::from(*coeff) * x.powi(i as i32);
        }
        result
    }

    pub fn rev_coeffs(&self) -> Vec<f64> {
        self.coeffs.clone().iter().rev().map(|c| *c).collect()
    }

    #[inline]
    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn derivative(&self) -> Polynomial {
        let mut coeffs = Vec::new();
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if i > 0 {
                coeffs.push(coeff * i as f64);
            }
        }
        Polynomial::new(coeffs)
    }

    pub fn derivative_eval<T>(&self, point: T) -> T
    where
        T: Num + From<f64> + std::ops::AddAssign<T> + Copy
    {
        let mut der = T::zero();
        for (i, coeff) in self.coeffs.iter().enumerate().skip(1) {
            der += T::from(*coeff) * T::from(i as f64) * pow(point, i - 1);
        }
        der
    }

    pub fn is_zero(&self) -> bool {
        println!("{:?}", self.coeffs);
        self.coeffs.len() == 0 || self.coeffs.len() == 1 && self.coeffs[0].abs() < 1e-5
    }

    pub fn to_string<T: Display>(&self, var: T) -> String {
        let mut s = String::with_capacity(200);

        const SUPERSCRIPTS: &str = "⁰¹²³⁴⁵⁶⁷⁸⁹";
        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if *coeff == 0.0 {
                continue;
            }
            if i > 0 {
                if !first {
                    if *coeff > 0.0 {
                        s.push_str(" + ");
                    } else if *coeff < 0.0 {
                        s.push_str(" - ");
                    }
                } else if *coeff < 0.0 {
                    s.push('-');
                }
                if coeff.abs() != 1.0 {
                    // s.push_str(&coeff.abs().to_significant_digits(5));
                    s.push_str(format!("{:.3} ", coeff.abs()).as_str());
                }
                if i == 1 {
                    s.push_str(format!("{}", var).as_str());
                } else {
                    s.push_str(format!("{}{}", var, i.to_string().chars().map(|c| SUPERSCRIPTS.chars().nth((c as usize) - ('0' as usize)).unwrap()).collect::<String>()).as_str());
                }
            } else {
                s.push_str(format!("{}", coeff).as_str());
            }
            first = false;
        }

        s
    }

    fn roots(&self) -> Vec<Complex<f64>> {
        let c = self.rev_coeffs();
        let n = c.len();

        let inz = c
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 0.0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        
        if inz.len() == 0 {
            return vec![];
        }

        let nnz = inz.len();
        let mut c = &c[inz[0]..=inz[nnz - 1]];

        let mut r = vec![Complex::<f64>::from(0.0); n - inz[nnz-1] - 1];

        let mut d = c[1..]
            .iter()
            .map(|&x| x / c[0])
            .collect::<Vec<_>>();

        while d.iter().any(|x| x.is_infinite()) {
            c = &c[1..];
            d = d[1..]
                .iter()
                .map(|&x| x / c[0])
                .collect();
        }

        let n = c.len();

        if n > 1 {
            let mut a: Array2<Complex<f64>> = 
                Array2::from_diag_elem(n-1, Complex::<f64>::from(0.0));
            
            a[[0, 0]] = Complex::<f64>::from(-d[0]);
            for i in 0..n-2 {
                a[[0, i+1]] = Complex::<f64>::from(-d[i+1]);
                a[[i+1, i]] = Complex::<f64>::from(1.0);
            }

            let eig = a.eigvals().unwrap().to_vec();
            r.extend_from_slice(&eig);
        }

        r
    }
}

impl Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        const SUPERSCRIPTS: &str = "⁰¹²³⁴⁵⁶⁷⁸⁹";
        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if *coeff == 0.0 {
                continue;
            }
            if !first {
                if *coeff > 0.0 {
                    write!(f, " + ")?;
                } else if *coeff < 0.0 {
                    write!(f, " - ")?;
                }
            } else if *coeff < 0.0 {
                write!(f, "-")?;
            }
            if coeff.abs() != 1.0 {
                // write!(f, "{}", coeff.abs().to_significant_digits(5))?;
                write!(f, "{:.3}", PrettyPrintFloat(coeff.abs()))?;
                if i > 0 {
                    write!(f, " ")?;
                }
            }
            if i == 1 {
                write!(f, "x")?;
            } else if i != 0 {
                write!(f, "x{}", i.to_string().chars().map(|c| SUPERSCRIPTS.chars().nth((c as usize) - ('0' as usize)).unwrap()).collect::<String>())?;
            }
            first = false;
        }
        Ok(())
    }
}

impl<T> Add<T> for Polynomial
where T: Into<Polynomial> {
    type Output = Polynomial;

    fn add(self, rhs: T) -> Polynomial {
        let rhs = rhs.into();
        if self.degree() > rhs.degree() {
            rhs + self
        } else {
            let coeffs: Vec<_> = self.coeffs
                .iter()
                .chain(std::iter::repeat(&0.0))
                .zip(rhs.coeffs.iter())
                .map(|(a, b)| a + b)
                .collect();
            Polynomial::new(coeffs)
        }
    }
}

impl<T> Sub<T> for Polynomial
where T: Into<Polynomial> {
    type Output = Polynomial;

    fn sub(self, rhs: T) -> Polynomial {
        self + (-rhs.into())
    }
}

impl Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Polynomial {
        let coeffs: Vec<_> = self.coeffs.iter().map(|c| -*c).collect();
        Polynomial::new(coeffs)
    }
}

impl<T> Mul<T> for Polynomial
where T: Into<Polynomial> {
    type Output = Polynomial;

    fn mul(self, rhs: T) -> Polynomial {
        let rhs = rhs.into();
        let mut coeffs = vec![0.0; self.degree() + rhs.degree() + 1];
        for i in 0..self.degree() + 1 {
            for j in 0..rhs.degree() + 1 {
                coeffs[i + j] += self.coeffs[i] * rhs.coeffs[j];
            }
        }
        Polynomial::new(coeffs)
    }
}

impl<T> Div<T> for Polynomial
where T: Into<Polynomial> {
    type Output = (Polynomial, Polynomial);

    fn div(self, rhs: T) -> (Polynomial, Polynomial) {
        let rhs = rhs.into();
        if self.degree() < rhs.degree() {
            (Polynomial::new(vec![0.0]), rhs.clone())
        } else if rhs.degree() == 0 {
            (
                Polynomial::new(self.coeffs.iter().map(|c| c / rhs.coeffs[0]).collect::<Vec<f64>>()),
                Polynomial::new(vec![0.0])
            )
        } else {
            let sc1 = rhs.coeffs[rhs.degree()];
            let mut c1  = self.coeffs.clone();
            let c2 = rhs.coeffs[..rhs.degree()].iter().map(|c| c / sc1).collect::<Vec<f64>>();
            let mut i = (self.degree() - rhs.degree()) as i64;
            let mut j: i64 = self.degree() as i64;

            while i >= 0 {
                for (ind, k) in (i..j).enumerate() {
                    c1[k as usize] -= c2[ind] * c1[j as usize];
                }
                i -= 1;
                j -= 1;
            }
            
            (
                Polynomial::new(c1[(j+1) as usize..].iter().map(|c| c / sc1).collect::<Vec<_>>()),
                Polynomial::new(c1[..(j+1) as usize].to_vec())
            )
        }
    }
}

impl Into<Polynomial> for &Polynomial {
    fn into(self) -> Polynomial {
        self.clone()
    }
}

impl<T> From<T> for Polynomial
where T: Into<f64> {
    fn from(t: T) -> Polynomial {
        Polynomial::new([t.into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;

    #[test]
    fn from_roots_works() {
        let p = Polynomial::from_roots(&vec![2.0, -3.0, 0.0], 2.0);
        // 2(x - 2)(x + 3)(x + 0)
        assert_eq!(p.coeffs, vec![0.0, -12.0, 2.0, 2.0], "Real roots don't work");

        let p = Polynomial::from_roots(&vec![Complex::<f64>::new(1.0, 1.0), Complex::<f64>::new(1.0, -1.0)], 1.0);
        assert_eq!(p.coeffs, vec![2.0, -2.0, 1.0], "Complex roots don't work");

        let p = Polynomial::from_complex_root_pair(Complex::<f64>::new(1.0, 1.0), 1.0);
        assert_eq!(p.coeffs, vec![2.0, -2.0, 1.0], "Complex root pair doesn't work");
    }

    #[test]
    fn evaluate_works() {
        let p = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        // 3 + x + 2x² + 4x³
        assert_eq!(p.evaluate(2.0), 45.0);
        assert_eq!(p.evaluate_complex(Complex::<f64>::new(2.0, 2.0)), Complex::<f64>::new(-59.0, 82.0));
    }

    #[test]
    fn derivative_works() {
        let p = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        // 3 + x + 2x² + 4x³
        // 1 + 4x + 12x²
        assert_eq!(p.derivative().coeffs, vec![1.0, 4.0, 12.0]);
    }

    #[test]
    fn derivative_eval_works() {
        let p = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        // 3 + x + 2x² + 4x³
        // 1 + 4x + 12x²
        assert_eq!(p.derivative_eval(Complex::new(3.0, 1.0)), Complex::new(109.0, 76.0));
    }

    #[test]
    fn rev_coeffs_works() {
        let p = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        assert_eq!(p.rev_coeffs(), vec![4.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn degree_works() {
        let p = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        assert_eq!(p.degree(), 3);
    }

    #[test]
    fn display_works() {
        let p1 = Polynomial::new(vec![-3.0, 1.0, -2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(format!("{}", p1), "x¹¹ + 4.0 x³ - 2.0 x² + x - 3.0");
        let p2 = Polynomial::new(vec![0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(format!("{}", p2), "x⁷ - x");
    }

    #[test]
    fn trimseq_works() {
        let v = vec![0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0];
        assert_eq!(Polynomial::trimseq(&v), vec![0.0, 0.0, 1.0, 2.0, 0.0, 1.0]);

        let v2 = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let trimmed = Polynomial::trimseq(&v2);
        assert_eq!(trimmed, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn add_works() {
        let p1 = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        let p2 = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!((p1 + p2).coeffs, vec![4.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn sub_works() {
        let p1 = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        let p2 = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!((p1.clone() - &p2).coeffs, vec![2.0, -1.0, -1.0, 4.0]);
        assert_eq!((p2 - &p1).coeffs, vec![-2.0, 1.0, 1.0, -4.0]);
        assert_eq!((p1 - 3).coeffs, vec![0.0, 1.0, 2.0, 4.0]);
    }

    #[test]
    fn multiply_works() {
        let p1 = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        let p2 = Polynomial::new(vec![1.0, 2.0, 3.0]);
        // 3 + x + 2x² + 4x³
        // 1 + 2x + 3x²
        // 3 + 2x + 6x² + 12x³
        assert_eq!((p1 * &p2).coeffs, vec![3.0, 7.0, 13.0, 11.0, 14.0, 12.0]);

        assert_eq!(p2 * 2, Polynomial::new(vec![2.0, 4.0, 6.0]));
    }

    #[test]
    fn divide_works() {
        let p1 = Polynomial::new(vec![3.0, 7.0, 13.0, 11.0, 14.0, 12.0]);
        let p2 = Polynomial::new(vec![3.0, 1.0, 2.0, 4.0]);
        let (q1, r1) = p1 / &p2;
        assert_eq!(q1.coeffs, vec![1.0, 2.0, 3.0]);
        assert_eq!(r1.coeffs, vec![0.0]);
        let p3 = Polynomial::new(vec![5.0, 8.0, 12.0, 11.0, 14.0, 12.0]);
        let (q2, r2) = p3 / &p2;
        assert_eq!(q2.coeffs, vec![1.0, 2.0, 3.0]);
        assert_eq!(r2.coeffs, vec![2.0, 1.0, -1.0]);

        assert_eq!((p2 / 2).0, Polynomial::new(vec![1.5, 0.5, 1.0, 2.0]));
    }

    #[test]
    fn roots_works() {
        let p = Polynomial::new(vec![1.0, 1.0, 1.0]);

        let roots = p.roots();

        let expected = vec![
            Complex::new(-0.5f64, -0.8660254037844386),
            Complex::new(-0.5,  0.8660254037844386),
        ];

        assert_eq!(roots.len(), 2);
        assert_close_no!(roots, expected);
    }

    #[test]
    fn roots_works2() {
        let p = Polynomial::new(vec![1, -5, 0, 1, -1]);

        let roots = p.roots();

        let expected = vec![
            Complex::new(-1.50407869,  0.0),
            Complex::new( 0.20130306,  0.0),
            Complex::new( 1.15138782, -1.40608738),
            Complex::new( 1.15138782,  1.40608738),
        ];

        assert_eq!(roots.len(), 4);
        assert_close_no!(roots, expected);
    }
}