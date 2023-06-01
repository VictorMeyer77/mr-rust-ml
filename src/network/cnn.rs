use ndarray::{Array2, Array3, Axis};
use crate::activation::relu::Relu;
use crate::activation::softmax::Softmax;
use crate::layer::activation_layer::ActivationLayer;
use crate::layer::conv_layer::ConvLayer;
use crate::layer::flatten_layer::FlattenLayer;
use crate::layer::layer::Layer;
use crate::layer::max_pooling_layer::MaxPoolingLayer;
use crate::loss::categorical_cross_entropy::CategoricalCrossEntropy;

pub struct Cnn {}

impl Cnn {
    pub fn fit(
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        x_test: Option<&Array2<f64>>,
        y_test: Option<&Array2<f64>>,
        epochs: usize,
        learning_rate: f64,
        accuracy_function: &str,
    ) -> () {

        let mut conv_layer: ConvLayer = ConvLayer::build(16, 3);
        let mut max_pooling_layer: MaxPoolingLayer = MaxPoolingLayer::build(2);
        let mut flatten_layer: FlattenLayer = FlattenLayer::new();
        let mut relu_layer: ActivationLayer = ActivationLayer::build(Box::new(Relu), 10, 10);
        let mut softmax_layer: ActivationLayer = ActivationLayer::build(Box::new(Softmax), 10, 10);

        let loss: CategoricalCrossEntropy = CategoricalCrossEntropy;

        for i in 0..epochs {

            let mut error: f64 = 0.0;

            for r in 0..(x_train.shape()[0]) {

                let mut row: Array2<f64> = x_train.select(Axis(0), &[r]);

                let conv_output: Array3<f64> = conv_layer.forward_propagation(&row);
                let max_pooling_output: Array3<f64> = max_pooling_layer.forward_propagation(&conv_output);
                let flatten_output: Array2<f64> = flatten_layer.forward_propagation(&max_pooling_output);
                let relu_output: Array2<f64> = relu_layer.forward_propagation(&flatten_output);
                let softmax_output: Array2<f64> = softmax_layer.forward_propagation(&relu_output);

                error += loss.func

            }

        }

    }
}
