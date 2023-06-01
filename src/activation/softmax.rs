use crate::activation::activation::Activation;
use ndarray::Array2;

#[derive(Debug)]
pub struct Softmax;

impl Activation for Softmax {
    fn function(&self, x: &Array2<f64>) -> Array2<f64> {
        if x.shape()[0] != 1 {
            panic!(
                "array must have only one row to apply softmax, actually {}",
                x.shape()[0]
            )
        }
        let exp_sum: f64 = x.map(|&x| x.exp()).sum();
        x.map(|&x| x.exp() / exp_sum)
    }

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        let eye: Array2<f64> = Array2::eye(x.shape()[0]);
        let softmax: Array2<f64> = self.function(x);
        &softmax * (eye - softmax.t())
    }

    fn get_name(&self) -> String {
        "Softmax".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn activation_softmax_function() -> () {
        let softmax: Softmax = Softmax;
        let input: Array2<f64> = arr2(&[[1.0, 2.0, 3.0, 6.0]]);
        let output: Array2<f64> = softmax.function(&input);
        let target: Array2<f64> = arr2(&[[0.00626879, 0.01704033, 0.04632042, 0.93037047]]);
        assert_eq!(output.shape(), target.shape());
        let output_vec: Vec<f64> = output.into_raw_vec();
        let target_vec: Vec<f64> = target.into_raw_vec();
        for i in 0..4 {
            assert!((output_vec[i] - target_vec[i]).powf(2.0) < 0.00001)
        }
    }

    #[test]
    #[should_panic(expected = "array must have only one row to apply softmax, actually 3")]
    fn activation_softmax_function_should_panic_when_array_has_not_one_row() -> () {
        let softmax: Softmax = Softmax;
        let input: Array2<f64> = arr2(&[
            [1.0, 2.0, 3.0, 6.0],
            [1.0, 2.0, 3.0, 6.0],
            [1.0, 2.0, 3.0, 6.0],
        ]);
        softmax.function(&input);
    }

    #[test]
    fn activation_softmax_derivative() -> () {
        let softmax: Softmax = Softmax;
        let input: Array2<f64> = arr2(&[[0.00626879, 0.01704033, 0.04632042, 0.93037047]]);
        let output: Array2<f64> = softmax.derivative(&input);
        let target: Array2<f64> = arr2(&[
            [0.14727423, 0.14886917, 0.15329252, 0.3710727],
            [0.14692532, 0.14851649, 0.15292935, 0.3701936],
            [0.14595769, 0.14753837, 0.15192218, 0.36775555],
            [0.09831691, 0.09938166, 0.10233458, 0.24771967],
        ]);
        assert_eq!(output.shape(), target.shape());
        let output_vec: Vec<f64> = output.into_raw_vec();
        let target_vec: Vec<f64> = target.into_raw_vec();
        for i in 0..16 {
            assert!((output_vec[i] - target_vec[i]).powf(2.0) < 0.00001)
        }
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        assert_eq!(Softmax.get_name(), "Softmax");
    }
}
