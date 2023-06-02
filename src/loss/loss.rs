use crate::loss::categorical_cross_entropy::CategoricalCrossEntropy;
use crate::loss::mse::Mse;
use ndarray::Array2;
use std::io::{Error, ErrorKind};

pub trait Loss {
    fn function(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64;

    fn derivative(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Array2<f64>;

    fn get_name(&self) -> String;
}

pub fn from_string(name: String) -> Result<Box<dyn Loss>, Error> {
    match name.to_uppercase().as_str() {
        "MSE" => Ok(Box::new(Mse)),
        "CATEGORICAL_CROSS_ENTROPY" => Ok(Box::new(CategoricalCrossEntropy)),
        _ => Err(Error::new(
            ErrorKind::InvalidInput,
            format!("unknown loss '{}'", name),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_string_should_return_loss() -> () {
        assert_eq!(
            from_string("MSE".to_string()).unwrap().get_name(),
            "MSE".to_string()
        );
        assert_eq!(
            from_string("categorical_cross_entropy".to_string())
                .unwrap()
                .get_name(),
            "categorical_cross_entropy".to_string()
        );
    }

    #[test]
    #[should_panic(expected = "unknown loss 'Unknown'")]
    fn from_string_should_raise_error_when_name_is_unknown() -> () {
        from_string("Unknown".to_string()).unwrap();
    }
}
