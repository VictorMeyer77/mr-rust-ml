use crate::activation::activation::Activation;
use ndarray::Array2;

#[derive(Debug)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn function(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|&x| 1.0 / (1.0 + (x * -1.0).exp()))
    }

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        let sigmoid: Array2<f64> = self.function(x);
        &sigmoid * (1.0 - &sigmoid)
    }

    fn get_name(&self) -> String {
        "Sigmoid".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn activation_sigmoid_function() -> () {
        let sigmoid: Sigmoid = Sigmoid;
        let input: Array2<f64> = arr2(&[[1.0, -10.0, 0.0], [15.0, -2.0, 0.0]]);
        let output: Array2<f64> = sigmoid.function(&input);
        let target: Array2<f64> = arr2(&[
            [0.7310585786300049, 0.0, 0.5],
            [1.0, 0.11920292202211755, 0.5],
        ]);
        assert_eq!(output.shape(), target.shape());
        let output_vec: Vec<f64> = output.into_raw_vec();
        let target_vec: Vec<f64> = target.into_raw_vec();
        for i in 0..6 {
            assert!((output_vec[i] - target_vec[i]).powf(2.0) < 0.00001)
        }
    }

    #[test]
    fn activation_sigmoid_derivative() -> () {
        let sigmoid: Sigmoid = Sigmoid;
        let input: Array2<f64> = arr2(&[[1.0, -10.0, 0.0], [15.0, -2.0, 0.0]]);
        let output: Array2<f64> = sigmoid.derivative(&input);
        let target: Array2<f64> = arr2(&[
            [0.19661193324148185, 0.0, 0.25],
            [0.0, 0.1049935854035065, 0.25],
        ]);
        assert_eq!(output.shape(), target.shape());
        let output_vec: Vec<f64> = output.into_raw_vec();
        let target_vec: Vec<f64> = target.into_raw_vec();
        for i in 0..6 {
            assert!((output_vec[i] - target_vec[i]).powf(2.0) < 0.00001)
        }
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        assert_eq!(Sigmoid.get_name(), "Sigmoid");
    }
}
