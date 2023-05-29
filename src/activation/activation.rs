use crate::activation::relu::Relu;
use crate::activation::sigmoid::Sigmoid;
use crate::activation::tanh::Tanh;
use ndarray::Array2;
use std::io::{Error, ErrorKind};

pub trait Activation {
    fn function(&self, x: &Array2<f64>) -> Array2<f64>;

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64>;

    fn get_name(&self) -> String;
}

pub fn from_string(name: String) -> Result<Box<dyn Activation>, Error> {
    match name.to_uppercase().as_str() {
        "TANH" => Ok(Box::new(Tanh)),
        "RELU" => Ok(Box::new(Relu)),
        "SIGMOID" => Ok(Box::new(Sigmoid)),
        _ => Err(Error::new(
            ErrorKind::InvalidInput,
            format!("unknown activation '{}'", name),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_string_should_return_activation() -> () {
        assert_eq!(
            from_string("Tanh".to_string()).unwrap().get_name(),
            "Tanh".to_string()
        );
        assert_eq!(
            from_string("Relu".to_string()).unwrap().get_name(),
            "Relu".to_string()
        );
        assert_eq!(
            from_string("Sigmoid".to_string()).unwrap().get_name(),
            "Sigmoid".to_string()
        );
    }

    #[test]
    #[should_panic(expected = "unknown activation 'Unknown'")]
    fn from_string_should_raise_error_when_name_is_unknown() -> () {
        from_string("Unknown".to_string()).unwrap();
    }
}
