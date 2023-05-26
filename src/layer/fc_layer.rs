use crate::layer::layer::Layer;
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug)]
pub struct FCLayer {
    input: Array2<f64>,
    weights: Array2<f64>,
    bias: Array2<f64>,
    shape: (usize, usize),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FCLayerModel {
    weights: Vec<f64>,
    bias: Vec<f64>,
    shape: (usize, usize),
}

impl FCLayer {
    pub fn build(input_size: usize, output_size: usize) -> FCLayer {
        FCLayer {
            input: Array::zeros((1, input_size)),
            weights: Array::random((input_size, output_size), Uniform::new(0.0, 1.0)) - 0.5,
            bias: Array::random((1, output_size), Uniform::new(0.0, 1.0)) - 0.5,
            shape: (input_size, output_size),
        }
    }

    pub fn from_json(json_str: &str) -> Result<FCLayer, Box<dyn Error>> {
        let model: FCLayerModel = serde_json::from_str(json_str)?;
        let weights: Array2<f64> = Array2::from_shape_vec(model.shape, model.weights)?;
        let bias: Array2<f64> = Array2::from_shape_vec([1, model.shape.1], model.bias)?;
        let layer: FCLayer = FCLayer {
            input: Array::zeros((1, model.shape.0)),
            weights,
            bias,
            shape: model.shape,
        };
        Ok(layer)
    }
}

impl Layer for FCLayer {
    fn forward_propagation(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.input = (*x).clone();
        self.input.dot(&self.weights) + &self.bias
    }

    fn backward_propagation(&mut self, y: &Array2<f64>, learning_rate: f64) -> Array2<f64> {
        let input_error: Array2<f64> = y.dot(&self.weights.t());
        let weight_error: Array2<f64> = self.input.t().dot(y);
        self.weights = &self.weights - (learning_rate * weight_error);
        self.bias = &self.bias - (learning_rate * y);
        input_error
    }

    fn get_shape(&self) -> (usize, usize) {
        self.shape
    }

    fn get_name(&self) -> String {
        "FCLayer".to_string()
    }

    fn to_json(&self) -> Result<String, Box<dyn Error>> {
        let model: FCLayerModel = FCLayerModel {
            weights: self.weights.clone().into_raw_vec(),
            bias: self.bias.clone().into_raw_vec(),
            shape: self.shape,
        };
        Ok(serde_json::to_string(&model)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn generate_test_fc_layer() -> FCLayer {
        FCLayer {
            input: arr2(&[[1.0, 0.5]]),
            weights: arr2(&[[0.0, 1.0, 0.0], [0.5, 1.0, 0.5]]),
            bias: arr2(&[[1.0, 1.0, 0.25]]),
            shape: (2, 3),
        }
    }

    #[test]
    fn build_should_initialize_layer() -> () {
        let layer: FCLayer = FCLayer::build(2, 3);
        assert_eq!(layer.input, arr2(&[[0.0, 0.0]]));
        assert_eq!(layer.weights.shape(), [2, 3]);
        assert_eq!(layer.bias.shape(), [1, 3]);
        assert_eq!(layer.shape, (2, 3));
    }

    #[test]
    fn forward_propagation_should_apply_weights_and_bias() -> () {
        let mut layer: FCLayer = generate_test_fc_layer();
        let result: Array2<f64> = layer.forward_propagation(&arr2(&[[1.0, 1.0]]));
        assert_eq!(layer.input, arr2(&[[1.0, 1.0]]));
        assert_eq!(result, arr2(&[[1.5, 3.0, 0.75]]));
    }

    #[test]
    fn backward_propagation_should_correct_weights_and_bias() -> () {
        let mut layer: FCLayer = generate_test_fc_layer();
        let result: Array2<f64> = layer.backward_propagation(&arr2(&[[1.0, 0.0, 0.0]]), 0.5);
        assert_eq!(layer.bias, arr2(&[[0.5, 1.0, 0.25]]));
        assert_eq!(layer.weights, arr2(&[[-0.5, 1.0, 0.0], [0.25, 1.0, 0.5]]));
        assert_eq!(result, arr2(&[[0.0, 0.5]]));
    }

    #[test]
    fn get_shape_should_return_layer_dim() -> () {
        let layer: FCLayer = generate_test_fc_layer();
        assert_eq!(layer.get_shape(), (2, 3));
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        let layer: FCLayer = generate_test_fc_layer();
        assert_eq!(layer.get_name(), "FCLayer");
    }

    #[test]
    fn to_json_should_serialize_layer() -> () {
        let layer: FCLayer = generate_test_fc_layer();
        assert_eq!(
            "{\"weights\":[0.0,1.0,0.0,0.5,1.0,0.5],\"bias\":[1.0,1.0,0.25],\"shape\":[2,3]}",
            layer.to_json().unwrap()
        );
    }

    #[test]
    fn from_json_should_deserialize_layer() -> () {
        let target_layer: FCLayer = generate_test_fc_layer();
        let json_str: &str =
            "{\"weights\":[0.0,1.0,0.0,0.5,1.0,0.5],\"bias\":[1.0,1.0,0.25],\"shape\":[2,3]}";
        let output_layer = FCLayer::from_json(json_str).unwrap();
        assert_eq!(target_layer.shape, output_layer.shape);
        assert_eq!(target_layer.bias.shape(), output_layer.bias.shape());
        assert_eq!(target_layer.weights.shape(), output_layer.weights.shape());
        let target_weights: Vec<f64> = target_layer.weights.into_raw_vec();
        let output_weights: Vec<f64> = output_layer.weights.into_raw_vec();
        for i in 0..target_weights.len() {
            assert!((target_weights[i] - output_weights[i]).powf(2.0) < 0.00001)
        }
        let target_bias: Vec<f64> = target_layer.bias.into_raw_vec();
        let output_bias: Vec<f64> = output_layer.bias.into_raw_vec();
        for i in 0..target_bias.len() {
            assert!((target_bias[i] - output_bias[i]).powf(2.0) < 0.00001)
        }
    }
}
