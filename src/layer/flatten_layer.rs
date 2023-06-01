use ndarray::{Array2, Array3};

#[derive(Debug)]
pub struct FlattenLayer {
    input_shape: (usize, usize, usize),
}

impl FlattenLayer {
    pub fn new() -> FlattenLayer {
        FlattenLayer {
            input_shape: (0, 0, 0),
        }
    }

    pub fn forward_propagation(&mut self, x: &Array3<f64>) -> Array2<f64> {
        let x_clone: Array3<f64> = (*x).clone();
        let shape: &[usize] = x_clone.shape();
        self.input_shape = (shape[0], shape[1], shape[2]);
        Array2::from_shape_vec((1, x_clone.len()), x_clone.into_raw_vec()).unwrap()
    }

    pub fn backward_propagation(&self, y: Array2<f64>) -> Array3<f64> {
        Array3::from_shape_vec(self.input_shape, y.into_raw_vec()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_should_initialize_layer() -> () {
        let layer: FlattenLayer = FlattenLayer::new();
        assert_eq!(layer.input_shape, (0, 0, 0))
    }

    #[test]
    fn forward_propagation_should_flatten_input() -> () {
        let mut layer: FlattenLayer = FlattenLayer::new();
        let input: Array3<f64> = Array3::from_shape_vec(
            (3, 3, 2),
            vec![
                30.0, 49.0, 36.0, 62.0, 45.5, 83.5, 22.5, 38.5, 34.5, 58.0, 44.5, 76.5, 28.5, 51.5,
                35.0, 61.5, 44.5, 76.5,
            ],
        )
        .unwrap();
        let flatten: Array2<f64> = layer.forward_propagation(&input);
        assert_eq!(layer.input_shape, (3, 3, 2));
        assert_eq!(
            flatten,
            Array2::from_shape_vec(
                (1, 18),
                vec![
                    30.0, 49.0, 36.0, 62.0, 45.5, 83.5, 22.5, 38.5, 34.5, 58.0, 44.5, 76.5, 28.5,
                    51.5, 35.0, 61.5, 44.5, 76.5,
                ]
            )
            .unwrap()
        )
    }

    #[test]
    fn backward_propagation_should_reshape_input() -> () {
        let mut layer: FlattenLayer = FlattenLayer::new();
        let input: Array3<f64> = Array3::from_shape_vec(
            (3, 3, 2),
            vec![
                30.0, 49.0, 36.0, 62.0, 45.5, 83.5, 22.5, 38.5, 34.5, 58.0, 44.5, 76.5, 28.5, 51.5,
                35.0, 61.5, 44.5, 76.5,
            ],
        )
        .unwrap();
        layer.forward_propagation(&input);
        let error_input: Array2<f64> = Array2::from_shape_vec(
            (1, 18),
            vec![
                30.0, 49.0, 36.0, 62.0, 45.5, 83.5, 22.5, 38.5, 34.5, 58.0, 44.5, 76.5, 28.5, 51.5,
                35.0, 61.5, 44.5, 76.5,
            ],
        )
        .unwrap();
        let output: Array3<f64> = layer.backward_propagation(error_input);
        assert_eq!(output, input);
    }
}
