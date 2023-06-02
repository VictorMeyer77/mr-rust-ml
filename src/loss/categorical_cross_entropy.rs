use crate::loss::loss::Loss;
use ndarray::Array2;
use ndarray_stats::EntropyExt;

pub struct CategoricalCrossEntropy;

impl Loss for CategoricalCrossEntropy {
    fn function(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        y_true.cross_entropy(y_pred).unwrap()
    }

    fn derivative(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64> {
        (y_true / (y_pred + 1e-10)) * -1.0
    }

    fn get_name(&self) -> String {
        "categorical_cross_entropy".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn loss_categorical_cross_entropy_function() -> () {
        let categorical_cross_entropy: CategoricalCrossEntropy = CategoricalCrossEntropy;
        let y_true: Array2<f64> = arr2(&[[0.0, 1.0, 0.0, 0.0]]);
        let y_pred: Array2<f64> = arr2(&[[0.05, 0.85, 0.10, 0.0]]);
        let output: f64 = categorical_cross_entropy.function(&y_true, &y_pred);
        assert!((output - 0.16251892949777494).sqrt() < 0.0001)
    }

    #[test]
    fn loss_categorical_cross_entropy_derivative() -> () {
        let categorical_cross_entropy: CategoricalCrossEntropy = CategoricalCrossEntropy;
        let y_true: Array2<f64> = arr2(&[[0.0, 1.0, 0.0, 0.0]]);
        let y_pred: Array2<f64> = arr2(&[[0.05, 0.85, 0.10, 0.0]]);
        let output: Array2<f64> = categorical_cross_entropy.derivative(&y_true, &y_pred);
        assert_eq!(output.shape(), &[1, 4]);
        let output_vec: Vec<f64> = output.into_raw_vec();
        assert!(output_vec[0].powf(2.0) < 0.0000001);
        assert!(output_vec[2].powf(2.0) < 0.0000001);
        assert!(output_vec[3].powf(2.0) < 0.0000001);
        assert!((output_vec[1] - -1.1764705880968858).powf(2.0) < 0.0000001);
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        assert_eq!(
            CategoricalCrossEntropy.get_name(),
            "categorical_cross_entropy"
        );
    }
}
