use ndarray_linalg::eig::EigVals;
use ndarray::prelude::*;
use rand::prelude::*;

pub mod math;

fn main() {
    // Create a 3x3 matrix
    let mut rng = thread_rng();
    let matrix = array![[rng.gen::<f64>(), rng.gen(), rng.gen()], [rng.gen(), rng.gen(), rng.gen()], [rng.gen(), rng.gen(), rng.gen()]];

    let eig = matrix.eigvals().unwrap();

    let sl = eig.as_slice().unwrap();

    println!("{:?}\n\n{:?}", matrix, sl);
}

#[cfg(test)]
mod tests {
    use super::main;

    #[test]
    fn test_main() {
        main();
        assert!(true);
    }
}