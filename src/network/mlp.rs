use crate::accuracy::accuracy::accuracy;
use crate::layer::layer;
use crate::layer::layer::Layer;
use crate::loss::loss;
use crate::loss::loss::Loss;
use crate::network::network::Network;
use crate::report::report::Report;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::read_to_string;
use std::time::Instant;

pub struct Mlp {
    layers: Vec<Box<dyn Layer>>,
    loss: Box<dyn Loss>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MlpModel {
    layers: Vec<(String, String)>,
    loss: String,
}

impl Mlp {
    pub fn build(loss: Box<dyn Loss>) -> Mlp {
        Mlp {
            layers: vec![],
            loss,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) -> () {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut result: Array2<f64> = Array2::zeros((0, self.layers.last().unwrap().get_shape().1));
        for r in 0..(x.shape()[0]) {
            let mut output: Array2<f64> = x.select(Axis(0), &[r]);
            self.layers
                .iter_mut()
                .for_each(|l| output = l.forward_propagation(&output));
            result.push_row(output.row(0)).unwrap();
        }
        result
    }

    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        x_test: Option<&Array2<f64>>,
        y_test: Option<&Array2<f64>>,
        epochs: usize,
        learning_rate: f64,
        accuracy_function: &str,
        report: &mut Report,
        report_step: usize,
    ) -> () {
        let start: Instant = Instant::now();
        let (x_test_shape, y_test_shape): (Option<&[usize]>, Option<&[usize]>) =
            if x_test.is_some() && y_test.is_some() {
                (Some(x_test.unwrap().shape()), Some(y_test.unwrap().shape()))
            } else {
                (None, None)
            };

        for i in 0..epochs {
            let mut error: f64 = 0.0;

            for r in 0..(x_train.shape()[0]) {
                let mut output: Array2<f64> = x_train.select(Axis(0), &[r]);
                self.layers
                    .iter_mut()
                    .for_each(|l| output = l.forward_propagation(&output));

                error += self.loss.function(&y_train.select(Axis(0), &[r]), &output);
                let mut error_buffer: Array2<f64> = self
                    .loss
                    .derivative(&y_train.select(Axis(0), &[r]), &output);

                self.layers.iter_mut().rev().for_each(|layer| {
                    error_buffer = layer.backward_propagation(&error_buffer, learning_rate)
                })
            }

            let train_accuracy: f64 =
                accuracy(accuracy_function, &self.predict(&x_train), &y_train);
            let train_loss: f64 = error / x_train.shape()[0] as f64;
            let test_accuracy: Option<f64> = if x_test.is_some() && y_test.is_some() {
                Some(accuracy(
                    accuracy_function,
                    &self.predict(&x_test.unwrap()),
                    &y_test.unwrap(),
                ))
            } else {
                None
            };
            report.add_data(i, train_accuracy, train_loss, test_accuracy);

            println!(
                "epochs {}/{} train loss {} train accuracy {}",
                i, epochs, train_loss, train_accuracy
            );
            if i > 0 && (i % report_step == 0 || i == epochs - 1) {
                report.generate(
                    self.get_name().as_str(),
                    start,
                    epochs,
                    x_train.shape(),
                    y_train.shape(),
                    x_test_shape,
                    y_test_shape,
                    accuracy_function,
                    self.loss.get_name().as_str(),
                );
            }
        }
    }
}

impl Network<Mlp> for Mlp {
    fn get_name(&self) -> String {
        "Mlp".to_string()
    }

    fn from_json(json_str: &str) -> Result<Mlp, Box<dyn Error>> {
        let model: MlpModel = serde_json::from_str(json_str)?;
        let mut layers: Vec<Box<dyn Layer>> = vec![];
        for layer in model.layers {
            layers.push(layer::from_string(layer.0, layer.1.as_str())?)
        }
        let mlp: Mlp = Mlp {
            layers,
            loss: loss::from_string(model.loss)?,
        };
        Ok(mlp)
    }

    fn to_json(&self) -> Result<String, Box<dyn Error>> {
        let mut layers: Vec<(String, String)> = vec![];
        for layer in &self.layers {
            layers.push((layer.get_name(), layer.to_json()?));
        }
        let model: MlpModel = MlpModel {
            layers,
            loss: self.loss.get_name(),
        };
        Ok(serde_json::to_string(&model)?)
    }

    fn load(path: &str) -> Result<Mlp, Box<dyn Error>> {
        let content: String = read_to_string(path)?;
        Ok(Mlp::from_json(content.as_str())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::tanh::Tanh;
    use crate::layer::activation_layer::ActivationLayer;
    use crate::layer::fc_layer::FCLayer;
    use crate::loss::mse::Mse;
    use ndarray::{arr2, Array1};
    use ndarray_stats::QuantileExt;
    use std::fs;
    use std::fs::remove_file;
    use std::path::Path;

    #[test]
    fn mlp_should_build_train_and_predict() -> () {
        let x_train: Array2<f64> = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y_train: Array2<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]);
        let x_test: Array2<f64> = arr2(&[[-0.05, -0.05], [0.0, 0.95], [0.95, 0.0], [1.05, 1.05]]);
        let y_test: Array2<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]);

        let mut mlp: Mlp = Mlp::build(Box::new(Mse));
        mlp.add_layer(Box::new(FCLayer::build(2, 3)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 3)));
        mlp.add_layer(Box::new(FCLayer::build(3, 2)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 2)));

        mlp.fit(
            &x_train,
            &y_train,
            Some(&x_test),
            Some(&y_test),
            1000,
            0.1,
            "categorical_accuracy",
            &mut Report::build("./test_report_mlp_1"),
            500,
        );

        let result: Array2<f64> = mlp.predict(&x_test);
        let result_argmax: Array1<usize> = result.map_axis(Axis(1), |row| row.argmax().unwrap());
        assert_eq!(mlp.layers.len(), 4);
        assert_eq!(result_argmax, Array1::from_vec(vec![0, 1, 1, 0]));
        assert!(Path::new("./test_report_mlp_1/500").exists());
        assert!(Path::new("./test_report_mlp_1/999").exists());

        fs::remove_dir_all("./test_report_mlp_1").unwrap();
    }

    #[test]
    fn to_json_should_serialize_mlp() -> () {
        let mut mlp: Mlp = Mlp::build(Box::new(Mse));
        mlp.add_layer(Box::new(FCLayer::build(2, 3)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 3)));
        mlp.add_layer(Box::new(FCLayer::build(3, 1)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 1)));

        let result: String = mlp.to_json().unwrap();

        assert_eq!(result.matches("FCLayer").count(), 2);
        assert_eq!(result.matches("ActivationLayer").count(), 2);
        assert_eq!(result.matches("Tanh").count(), 2);
        assert_eq!(result.matches("MSE").count(), 1);
        assert_eq!(result.matches("[2,3]").count(), 1);
        assert_eq!(result.matches("[3,3]").count(), 1);
        assert_eq!(result.matches("[3,1]").count(), 2);
    }

    #[test]
    fn from_json_should_deserialize_mlp() -> () {
        let x_train: Array2<f64> = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y_train: Array2<f64> = arr2(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]);
        let x_test: Array2<f64> = arr2(&[[-0.05, -0.05], [0.0, 0.95], [0.95, 0.0], [1.05, 1.05]]);

        let mut mlp: Mlp = Mlp::build(Box::new(Mse));
        mlp.add_layer(Box::new(FCLayer::build(2, 3)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 3)));
        mlp.add_layer(Box::new(FCLayer::build(3, 2)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 2)));

        mlp.fit(
            &x_train,
            &y_train,
            None,
            None,
            1000,
            0.1,
            "categorical_accuracy",
            &mut Report::build("./test_report_mlp_2"),
            500,
        );

        let network_str: String = mlp.to_json().unwrap();
        let mut mlp: Mlp = Mlp::from_json(network_str.as_str()).unwrap();

        let result: Array2<f64> = mlp.predict(&x_test);
        let result_argmax: Array1<usize> = result.map_axis(Axis(1), |row| row.argmax().unwrap());
        assert_eq!(mlp.layers.len(), 4);
        assert_eq!(result_argmax, Array1::from_vec(vec![0, 1, 1, 0]));
        fs::remove_dir_all("./test_report_mlp_2").unwrap();
    }

    #[test]
    fn get_name_should_return_struct_name() -> () {
        assert_eq!(Mlp::build(Box::new(Mse)).get_name(), "Mlp");
    }

    #[test]
    fn save_should_write_mlp_to_file() -> () {
        let mut mlp: Mlp = Mlp::build(Box::new(Mse));
        mlp.add_layer(Box::new(FCLayer::build(2, 3)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 3)));
        mlp.add_layer(Box::new(FCLayer::build(3, 1)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 1)));
        mlp.save("", "UnitTestMLP1".to_string()).unwrap();

        let result: String = read_to_string("UnitTestMLP1.json").unwrap();
        remove_file("UnitTestMLP1.json").unwrap();

        assert_eq!(result.matches("FCLayer").count(), 2);
        assert_eq!(result.matches("ActivationLayer").count(), 2);
        assert_eq!(result.matches("Tanh").count(), 2);
        assert_eq!(result.matches("MSE").count(), 1);
        assert_eq!(result.matches("[2,3]").count(), 1);
        assert_eq!(result.matches("[3,3]").count(), 1);
        assert_eq!(result.matches("[3,1]").count(), 2);
    }

    #[test]
    fn load_should_read_mlp() -> () {
        let mut mlp: Mlp = Mlp::build(Box::new(Mse));
        mlp.add_layer(Box::new(FCLayer::build(2, 3)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 3)));
        mlp.add_layer(Box::new(FCLayer::build(3, 1)));
        mlp.add_layer(Box::new(ActivationLayer::build(Box::new(Tanh), 3, 1)));
        mlp.save("", "UnitTestMLP2".to_string()).unwrap();

        let mlp: Mlp = Mlp::load("UnitTestMLP2.json").unwrap();
        let result: String = mlp.to_json().unwrap();
        remove_file("UnitTestMLP2.json").unwrap();

        assert_eq!(result.matches("FCLayer").count(), 2);
        assert_eq!(result.matches("ActivationLayer").count(), 2);
        assert_eq!(result.matches("Tanh").count(), 2);
        assert_eq!(result.matches("MSE").count(), 1);
        assert_eq!(result.matches("[2,3]").count(), 1);
        assert_eq!(result.matches("[3,3]").count(), 1);
        assert_eq!(result.matches("[3,1]").count(), 2);
    }
}
