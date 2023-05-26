use ndarray::{Array1, Array2, Axis};
use ndarray_stats::QuantileExt;

pub fn accuracy(function: &str, y_pred: Array2<f64>, y_true: Array2<f64>) -> f64 {
    if y_pred.shape() != y_true.shape() {
        panic!(
            "shapes are not equals {:?} != {:?}",
            y_pred.shape(),
            y_true.shape()
        );
    }
    match function {
        "categorical_accuracy" => categorical_accuracy(y_pred, y_true),
        _ => panic!("unknown accuracy function '{}'", function),
    }
}

fn categorical_accuracy(y_pred: Array2<f64>, y_true: Array2<f64>) -> f64 {
    if y_pred.shape()[1] < 2 || y_true.shape()[1] < 2 {
        panic!("array must be one hot encoding");
    }
    let pred_argmax: Array1<usize> = y_pred.map_axis(Axis(1), |row| row.argmax().unwrap());
    let true_argmax: Array1<usize> = y_true.map_axis(Axis(1), |row| row.argmax().unwrap());
    let mut correct: i32 = 0;
    for i in 0..pred_argmax.len() {
        if pred_argmax[i] == true_argmax[i] {
            correct += 1;
        }
    }
    correct as f64 / pred_argmax.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn categorical_accuracy_should_compute_accuracy() -> () {
        let y_pred: Array2<f64> = arr2(&[
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        let y_true: Array2<f64> = arr2(&[
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        assert_eq!(accuracy("categorical_accuracy", y_pred, y_true), 0.75)
    }

    #[test]
    #[should_panic(expected = "array must be one hot encoding")]
    fn categorical_accuracy_should_panic_when_arrays_are_not_one_hot() -> () {
        let y_pred: Array2<f64> = arr2(&[[0.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]);
        let y_true: Array2<f64> = arr2(&[[0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0]]);
        accuracy("categorical_accuracy", y_pred, y_true);
    }

    #[test]
    #[should_panic(expected = "unknown accuracy function 'Unknown'")]
    fn accuracy_should_panic_when_function_is_unknown() -> () {
        accuracy("Unknown", Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    #[test]
    #[should_panic(expected = "shapes are not equals [2, 3] != [4, 3]")]
    fn accuracy_should_panic_when_array_shapes_are_not_equal() -> () {
        let y_pred: Array2<f64> = arr2(&[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]);
        let y_true: Array2<f64> = arr2(&[
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]);
        accuracy("categorical_accuracy", y_pred, y_true);
    }
}
