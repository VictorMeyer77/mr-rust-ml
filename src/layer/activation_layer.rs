use crate::activation;
use crate::activation::activation::Activation;
use crate::layer::layer::Layer;
use ndarray::{Array, Array2};
use serde::{Deserialize, Serialize};
use std::error::Error;

pub struct ActivationLayer {
    input: Array2<f64>,
    activation: Box<dyn Activation>,
    shape: (usize, usize),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationLayerModel {
    activation: String,
    shape: (usize, usize),
}

impl ActivationLayer {
    pub fn build(
        activation: Box<dyn Activation>,
        input_size: usize,
        output_size: usize,
    ) -> ActivationLayer {
        ActivationLayer {
            input: Array::zeros((0, 0)),
            activation,
            shape: (input_size, output_size),
        }
    }

    pub fn from_json(json_str: &str) -> Result<ActivationLayer, Box<dyn Error>> {
        let model: ActivationLayerModel = serde_json::from_str(json_str)?;
        let layer: ActivationLayer = ActivationLayer {
            input: Array::zeros((0, 0)),
            activation: activation::activation::from_string(model.activation)?,
            shape: model.shape,
        };
        Ok(layer)
    }
}

impl Layer for ActivationLayer {
    fn forward_propagation(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.input = (*x).clone();
        self.activation.function(x)
    }

    fn backward_propagation(&mut self, y: &Array2<f64>, _learning_rate: f64) -> Array2<f64> {
        self.activation.derivative(&self.input) * y
    }

    fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    fn get_name(&self) -> String {
        "ActivationLayer".to_string()
    }

    fn to_json(&self) -> Result<String, Box<dyn Error>> {
        let model: ActivationLayerModel = ActivationLayerModel {
            activation: self.activation.get_name(),
            shape: self.shape,
        };
        Ok(serde_json::to_string(&model)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::tanh::Tanh;
    use ndarray::arr2;

    fn generate_test_activation_layer() -> ActivationLayer {
        ActivationLayer {
            input: arr2(&[[1.0, 0.5, 0.5]]),
            activation: Box::new(Tanh),
            shape: (2, 3),
        }
    }

    #[test]
    fn build_should_initialize_layer() -> () {
        let layer: ActivationLayer = ActivationLayer::build(Box::new(Tanh), 3, 2);
        assert_eq!(layer.input.len(), 0);
        assert_eq!(layer.shape, (3, 2));
    }

    #[test]
    fn forward_propagation_should_apply_weights_and_bias() -> () {
        let mut layer: ActivationLayer = generate_test_activation_layer();
        let result: Array2<f64> = layer.forward_propagation(&arr2(&[[0.5, 1.0]]));
        assert_eq!(layer.input, arr2(&[[0.5, 1.0]]));
        let result_vec: Vec<f64> = result.into_raw_vec();
        assert!((result_vec[0] - 0.46211715726000974).powf(2.0) < 0.00001);
        assert!((result_vec[1] - 0.7615941559557649).powf(2.0) < 0.00001);
    }

    #[test]
    fn backward_propagation_should_return_input_error() -> () {
        let mut layer: ActivationLayer = generate_test_activation_layer();
        layer.forward_propagation(&arr2(&[[0.9, 0.5]]));
        let result: Array2<f64> = layer.backward_propagation(&arr2(&[[1.0, 1.0]]), 0.0);
        assert_eq!(result.shape(), &[1, 2]);
        let result_vec: Vec<f64> = result.into_raw_vec();
        assert!((result_vec[0] - 0.4869173611483415).powf(2.0) < 0.00001);
        assert!((result_vec[1] - 0.7864477329659274).powf(2.0) < 0.00001);
    }

    #[test]
    fn get_shape_should_return_layer_dim() -> () {
        let layer: ActivationLayer = generate_test_activation_layer();
        assert_eq!(layer.shape, (2, 3))
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        let layer: ActivationLayer = generate_test_activation_layer();
        assert_eq!(layer.get_name(), "ActivationLayer");
    }

    #[test]
    fn to_json_should_serialize_layer() -> () {
        let layer: ActivationLayer = generate_test_activation_layer();
        assert_eq!(
            "{\"activation\":\"Tanh\",\"shape\":[2,3]}",
            layer.to_json().unwrap()
        );
    }

    #[test]
    fn from_json_should_deserialize_layer() -> () {
        let target_layer: ActivationLayer = generate_test_activation_layer();
        let json_str: &str = "{\"activation\":\"Tanh\",\"shape\":[2,3]}";
        let output_layer = ActivationLayer::from_json(json_str).unwrap();
        assert_eq!(target_layer.shape, output_layer.shape);
        assert_eq!(target_layer.activation.get_name(), "Tanh");
        assert_eq!(&[0, 0], output_layer.input.shape());
    }
}
