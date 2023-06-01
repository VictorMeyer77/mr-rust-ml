use crate::loss::loss::Loss;
use ndarray::Array2;

pub struct CategoricalCrossEntropy;

impl Loss for CategorialCrossEntropy {
    fn function(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        let pred_log: Array2<f64> = (y_pred + 1e-10).log();
        (y_true * pred_log).sum() * -1.0
    }

    fn derivative(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64> {
        (2.0 * (y_pred - y_true)) / y_true.len() as f64
    }

    fn get_name(&self) -> String {
        "MSE".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn activation_mse_function() -> () {
        let mse: CategoricalCrossEntropy = CategoricalCrossEntropy;
        let y_pred: Array2<f64> = arr2(&[[1.0, 2.0, 1.0, 0.0]]);
        let y_true: Array2<f64> = arr2(&[[2.0, 1.0, 1.0, 0.0]]);
        let output: f64 = mse.function(&y_pred, &y_true);
        println!("{}", output);
        assert!((output - 0.5).sqrt() < 0.0001)
    }

    #[test]
    fn activation_mse_derivative() -> () {
        let mse: CategoricalCrossEntropy = CategoricalCrossEntropy;
        let y_pred: Array2<f64> = arr2(&[[1.0, 2.0, 1.0, 0.0]]);
        let y_true: Array2<f64> = arr2(&[[2.0, 1.0, 1.0, 0.0]]);
        let output: Array2<f64> = mse.derivative(&y_pred, &y_true);
        let target: Array2<f64> = arr2(&[[0.5, -0.5, 0.0, 0.0]]);
        assert_eq!(output.shape(), target.shape());
        let output_vec: Vec<f64> = output.into_raw_vec();
        let target_vec: Vec<f64> = target.into_raw_vec();
        for i in 0..4 {
            assert!((output_vec[i] - target_vec[i]).powf(2.0) < 0.00001)
        }
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        assert_eq!(Categoric  alCrossEntropy.get_name(), "MSE");
    }
}
