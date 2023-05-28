use crate::activation::activation::Activation;
use ndarray::Array2;

#[derive(Debug)]
pub struct Relu;

impl Activation for Relu {
    fn function(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn get_name(&self) -> String {
        "Relu".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn activation_relu_function() -> () {
        let relu: Relu = Relu;
        let input: Array2<f64> = arr2(&[[-0.4, 0.5, -0.6], [0.7, -0.8, 0.9]]);
        let output: Array2<f64> = relu.function(&input);
        let target: Array2<f64> = arr2(&[[0.0, 0.5, 0.0], [0.7, 0.0, 0.9]]);
        assert_eq!(output, target);
    }

    #[test]
    fn activation_relu_derivative() -> () {
        let relu: Relu = Relu;
        let input: Array2<f64> = arr2(&[[-0.4, 0.5, -0.6], [0.7, -0.8, 0.9]]);
        let output: Array2<f64> = relu.derivative(&input);
        let target: Array2<f64> = arr2(&[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]);
        assert_eq!(output, target);
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        assert_eq!(Relu.get_name(), "Relu");
    }
}
