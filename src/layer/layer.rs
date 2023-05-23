use crate::layer::activation_layer::ActivationLayer;
use crate::layer::fc_layer::FCLayer;
use ndarray::Array2;
use std::error::Error;
use std::io;

pub trait Layer {
    fn forward_propagation(&mut self, x: &Array2<f64>) -> Array2<f64>;

    fn backward_propagation(&mut self, y: &Array2<f64>, learning_rate: f64) -> Array2<f64>;

    fn get_shape(&self) -> (usize, usize);

    fn get_name(&self) -> String;

    fn to_json(&self) -> Result<String, Box<dyn Error>>;
}

pub fn from_string(name: String, json_str: &str) -> Result<Box<dyn Layer>, Box<dyn Error>> {
    match name.to_uppercase().as_str() {
        "FCLAYER" => Ok(Box::new(FCLayer::from_json(json_str)?)),
        "ACTIVATIONLAYER" => Ok(Box::new(ActivationLayer::from_json(json_str)?)),
        _ => Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown layer '{}'", name),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_string_should_return_layer() -> () {
        let fc_layer_str: &str =
            "{\"weights\":[0.0,1.0,0.0,0.5,1.0,0.5],\"bias\":[1.0,1.0,0.25],\"shape\":[2,3]}";
        let activation_layer_str: &str = "{\"activation\":\"Tanh\",\"shape\":[2,3]}";
        assert_eq!(
            from_string("FCLayer".to_string(), fc_layer_str)
                .unwrap()
                .get_name(),
            "FCLayer".to_string()
        );
        assert_eq!(
            from_string("ActivationLayer".to_string(), activation_layer_str)
                .unwrap()
                .get_name(),
            "ActivationLayer".to_string()
        );
    }

    #[test]
    #[should_panic(expected = "unknown layer 'Unknown'")]
    fn from_string_should_raise_error_when_name_is_unknown() -> () {
        from_string("Unknown".to_string(), "").unwrap();
    }
}
