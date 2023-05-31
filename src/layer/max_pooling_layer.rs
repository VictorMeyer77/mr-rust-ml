use ndarray::{s, Array3, ArrayView3};
use ndarray_stats::QuantileExt;

#[derive(Debug)]
pub struct MaxPoolingLayer<'a> {
    input: Option<&'a Array3<f64>>,
    kernel_size: usize,
}

impl<'a> MaxPoolingLayer<'a> {
    fn build(kernel_size: usize) -> MaxPoolingLayer<'a> {
        MaxPoolingLayer {
            input: None,
            kernel_size,
        }
    }

    fn image_to_patches(&mut self, x: &'a Array3<f64>) -> Vec<(ArrayView3<f64>, usize, usize)> {
        self.input = Some(&x);
        let shape: &[usize] = x.shape();
        let mut patches_buffer: Vec<(ArrayView3<f64>, usize, usize)> = vec![];
        for h in 0..(shape[0] / self.kernel_size) {
            for w in 0..(shape[1] / self.kernel_size) {
                patches_buffer.push((
                    x.slice(s![
                        (h * self.kernel_size)..(h * self.kernel_size + self.kernel_size),
                        (w * self.kernel_size)..(w * self.kernel_size + self.kernel_size),
                        ..
                    ]),
                    h,
                    w,
                ));
            }
        }
        patches_buffer
    }

    fn forward_propagation(&mut self, x: &'a Array3<f64>) -> Array3<f64> {
        let x_shape: &[usize] = x.shape();
        let mut max_pooling_output: Array3<f64> = Array3::zeros((
            x_shape[0] / self.kernel_size,
            x_shape[1] / self.kernel_size,
            x_shape[2],
        ));
        for (patch, h, w) in self.image_to_patches(x) {
            let argmax: (usize, usize, usize) = patch.argmax().unwrap();
            for i in 0..x_shape[2] {
                max_pooling_output[[h, w, i]] = patch[[argmax.0, argmax.1, i]]
            }
        }
        max_pooling_output
    }

    fn backward_propagation(&mut self, y: &Array3<f64>, _learning_rate: f64) -> Array3<f64> {
        let input_shape: &[usize] = self.input.unwrap().shape();
        let mut kernel_error: Array3<f64> =
            Array3::zeros((input_shape[0], input_shape[1], input_shape[2]));
        let kernel_size: usize = self.kernel_size;
        let patches: Vec<(ArrayView3<f64>, usize, usize)> =
            self.image_to_patches(self.input.unwrap());
        for (patch, h, w) in &patches {
            let patch_shape: &[usize] = patch.shape();
            let argmax: (usize, usize, usize) = patch.argmax().unwrap();
            for ph in 0..patch_shape[0] {
                for pw in 0..patch_shape[1] {
                    for pk in 0..patch_shape[2] {
                        if (patch[[ph, pw, pk]] - patch[[argmax.0, argmax.1, pk]]).powf(2.0)
                            < 0.00000001
                        {
                            kernel_error[[*h * kernel_size + ph, *w * kernel_size + pw, pk]] =
                                y[[*h, *w, pk]];
                        }
                    }
                }
            }
        }
        kernel_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    fn generate_test_max_pooling_layer<'a>() -> MaxPoolingLayer<'a> {
        MaxPoolingLayer {
            input: None,
            kernel_size: 2,
        }
    }

    #[test]
    fn build_should_initialize_layer() -> () {
        let layer: MaxPoolingLayer = MaxPoolingLayer::build(2);
        assert!(layer.input.is_none());
        assert_eq!(layer.kernel_size, 2);
    }

    #[test]
    fn image_to_patches_should_return_vec_of_pattern() -> () {
        let mut layer: MaxPoolingLayer = generate_test_max_pooling_layer();
        let x: Array3<f64> = arr3(&[
            [[30.0, 49.0], [36.0, 62.0], [45.5, 83.5]],
            [[22.5, 38.5], [34.5, 58.0], [44.5, 76.5]],
            [[28.5, 51.5], [35.0, 61.5], [44.5, 76.5]],
        ]);
        let output: Vec<(ArrayView3<f64>, usize, usize)> = layer.image_to_patches(&x);
        let target: Vec<(Array3<f64>, usize, usize)> = vec![(
            Array3::from_shape_vec(
                (2, 2, 2),
                vec![30.0, 49.0, 36.0, 62.0, 22.5, 38.5, 34.5, 58.0],
            )
            .unwrap(),
            0,
            0,
        )];
        assert_eq!(output.len(), target.len());
        for i in 0..target.len() {
            assert_eq!(output[i].0, target[i].0.view());
            assert_eq!(output[i].1, target[i].1);
            assert_eq!(output[i].2, target[i].2);
        }
    }

    #[test]
    fn forward_propagation_apply_kernel() -> () {
        let mut layer: MaxPoolingLayer = generate_test_max_pooling_layer();
        let x: Array3<f64> = arr3(&[
            [[30.0, 49.0], [36.0, 62.0], [45.5, 83.5]],
            [[22.5, 38.5], [34.5, 58.0], [44.5, 76.5]],
            [[28.5, 51.5], [35.0, 61.5], [44.5, 76.5]],
        ]);
        let output: Array3<f64> = layer.forward_propagation(&x);
        assert_eq!(
            output,
            Array3::from_shape_vec((1, 1, 2), vec![36.0, 62.0]).unwrap()
        );
    }

    #[test]
    fn backward_propagation_should_correct_kernel() -> () {
        let mut layer: MaxPoolingLayer = generate_test_max_pooling_layer();
        let x: Array3<f64> = arr3(&[
            [[30.0, 49.0], [36.0, 62.0], [45.5, 83.5]],
            [[22.5, 38.5], [34.5, 58.0], [44.5, 76.5]],
            [[28.5, 51.5], [35.0, 61.5], [44.5, 76.5]],
        ]);
        layer.forward_propagation(&x);
        let output: Array3<f64> = layer.backward_propagation(&x, 0.0);
        let target: Array3<f64> = Array3::from_shape_vec(
            (3, 3, 2),
            vec![
                0.0, 0.0, 30.0, 49.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ],
        )
        .unwrap();
        assert_eq!(output, target);
    }
}
