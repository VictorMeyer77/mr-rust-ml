use crate::activation::activation::Activation;
use ndarray::Array2;

#[derive(Debug)]
pub struct Tanh;

impl Activation for Tanh {
    fn function(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|x| x.tanh())
    }

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|x| 1.0 - x.tanh().powf(2.0))
    }

    fn get_name(&self) -> String {
        "Tanh".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn activation_tanh_function() -> () {
        let tanh: Tanh = Tanh;
        let input: Array2<f64> = arr2(&[[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]);
        let output: Array2<f64> = tanh.function(&input);
        let target: Array2<f64> = arr2(&[
            [0.37994896, 0.46211716, 0.53704957],
            [0.60436778, 0.66403677, 0.71629787],
        ]);
        assert_eq!(output.shape(), target.shape());
        let output_vec: Vec<f64> = output.into_raw_vec();
        let target_vec: Vec<f64> = target.into_raw_vec();
        for i in 0..6 {
            assert!((output_vec[i] - target_vec[i]).powf(2.0) < 0.00001)
        }
    }

    #[test]
    fn activation_tanh_derivative() -> () {
        let tanh: Tanh = Tanh;
        let input: Array2<f64> = arr2(&[[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]);
        let output: Array2<f64> = tanh.derivative(&input);
        let target: Array2<f64> = arr2(&[
            [0.85563879, 0.78644773, 0.71157776],
            [0.63473959, 0.55905517, 0.48691736],
        ]);
        assert_eq!(output.shape(), target.shape());
        let output_vec: Vec<f64> = output.into_raw_vec();
        let target_vec: Vec<f64> = target.into_raw_vec();
        for i in 0..6 {
            assert!((output_vec[i] - target_vec[i]).powf(2.0) < 0.00001)
        }
    }
}
