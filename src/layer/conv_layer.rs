use ndarray::{s, Array, Array1, Array2, Array3, ArrayView2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct ConvLayer<'a> {
    input: Option<&'a Array2<f64>>,
    kernel_size: usize,
    kernel_num: usize,
    kernels: Array3<f64>,
}

impl<'a> ConvLayer<'a> {
    pub fn build(kernel_size: usize, kernel_num: usize) -> ConvLayer<'a> {
        ConvLayer {
            input: None,
            kernel_size,
            kernel_num,
            kernels: Array::random(
                (kernel_num, kernel_size, kernel_size),
                Uniform::new(0.0, 1.0),
            ) / kernel_size.pow(2) as f64,
        }
    }

    fn image_to_patches(&mut self, x: &'a Array2<f64>) -> Vec<(ArrayView2<f64>, usize, usize)> {
        self.input = Some(&x);
        let shape: &[usize] = x.shape();
        let mut patches_buffer: Vec<(ArrayView2<f64>, usize, usize)> = vec![];
        for h in 0..(shape[0] - self.kernel_size + 1) {
            for w in 0..(shape[1] - self.kernel_size + 1) {
                patches_buffer.push((
                    x.slice(s![h..(h + self.kernel_size), w..(w + self.kernel_size)]),
                    h,
                    w,
                ));
            }
        }
        patches_buffer
    }

    pub fn forward_propagation(&mut self, x: &'a Array2<f64>) -> Array3<f64> {
        let shape: &[usize] = x.shape();
        let mut convolution_buffer: Vec<f64> = vec![];
        let kernel_buffer: Array3<f64> = self.kernels.clone();
        for (patch, _, _) in self.image_to_patches(x) {
            let product: Array3<f64> = &patch * &kernel_buffer;
            let sum: Array1<f64> = product.sum_axis(Axis(2)).sum_axis(Axis(1));
            convolution_buffer.append(&mut sum.into_raw_vec());
        }
        Array3::from_shape_vec(
            (
                shape[0] - self.kernel_size + 1,
                shape[1] - self.kernel_size + 1,
                self.kernel_num,
            ),
            convolution_buffer,
        )
        .unwrap()
    }

    pub fn backward_propagation(&mut self, y: &Array3<f64>, learning_rate: f64) -> Array3<f64> {
        let kernel_num: usize = self.kernel_num.clone();
        let mut kernel_error: Array3<f64> =
            Array3::zeros((self.kernel_num, self.kernel_size, self.kernel_size));
        for (patch, h, w) in self.image_to_patches(self.input.unwrap()) {
            for k in 0..kernel_num {
                let product: Array2<f64> = &patch * y[[h, w, k]];
                for i in 0..product.shape()[0] {
                    for j in 0..product.shape()[1] {
                        kernel_error[[k, i, j]] += product[[i, j]];
                    }
                }
            }
        }
        self.kernels = &self.kernels - (learning_rate * &kernel_error);
        kernel_error
    }
}
/*
impl Layer for ConvLayer {
    fn forward_propagation(&mut self, x: &Array2<f64>) -> Array2<f64> {
        todo!()
    }

    fn backward_propagation(&mut self, y: &Array2<f64>, learning_rate: f64) -> Array2<f64> {
        todo!()
    }

    fn get_shape(&self) -> (usize, usize) {
        todo!()
    }

    fn get_name(&self) -> String {
        todo!()
    }

    fn to_json(&self) -> Result<String, Box<dyn Error>> {
        todo!()
    }
}*/

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn generate_test_conv_layer<'a>() -> ConvLayer<'a> {
        ConvLayer {
            input: None,
            kernel_size: 2,
            kernel_num: 2,
            kernels: Array3::from_shape_vec(
                (2, 2, 2),
                vec![1.0, 0.5, 1.5, 2.5, 0.5, 3.5, 4.0, 1.5],
            )
            .unwrap(),
        }
    }

    #[test]
    fn build_should_initialize_layer() -> () {
        let layer: ConvLayer = ConvLayer::build(2, 2);
        assert!(layer.input.is_none());
        assert_eq!(layer.kernel_size, 2);
        assert_eq!(layer.kernel_num, 2);
        assert_eq!(layer.kernels.shape(), &[2, 2, 2]);
    }

    #[test]
    fn image_to_patches_should_return_vec_of_pattern() -> () {
        let mut layer: ConvLayer = generate_test_conv_layer();
        let x: Array2<f64> = arr2(&[
            [6.0, 6.0, 7.0, 11.0],
            [4.0, 6.0, 7.0, 9.0],
            [2.0, 5.0, 7.0, 9.0],
            [6.0, 6.0, 7.0, 9.0],
        ]);
        let output: Vec<(ArrayView2<f64>, usize, usize)> = layer.image_to_patches(&x);
        let target: Vec<(Array2<f64>, usize, usize)> = vec![
            (
                Array2::from_shape_vec((2, 2), vec![6.0, 6.0, 4.0, 6.0]).unwrap(),
                0,
                0,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![6.0, 7.0, 6.0, 7.0]).unwrap(),
                0,
                1,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![7.0, 11.0, 7.0, 9.0]).unwrap(),
                0,
                2,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![4.0, 6.0, 2.0, 5.0]).unwrap(),
                1,
                0,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![6.0, 7.0, 5.0, 7.0]).unwrap(),
                1,
                1,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![7.0, 9.0, 7.0, 9.0]).unwrap(),
                1,
                2,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![2.0, 5.0, 6.0, 6.0]).unwrap(),
                2,
                0,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![5.0, 7.0, 6.0, 7.0]).unwrap(),
                2,
                1,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![7.0, 9.0, 7.0, 9.0]).unwrap(),
                2,
                2,
            ),
        ];
        assert_eq!(output.len(), target.len());
        for i in 0..target.len() {
            assert_eq!(output[i].0, target[i].0.view());
            assert_eq!(output[i].1, target[i].1);
            assert_eq!(output[i].2, target[i].2);
        }
    }

    #[test]
    fn forward_propagation_apply_kernel() -> () {
        let mut layer: ConvLayer = generate_test_conv_layer();
        let x: Array2<f64> = arr2(&[
            [6.0, 6.0, 7.0, 11.0],
            [4.0, 6.0, 7.0, 9.0],
            [2.0, 5.0, 7.0, 9.0],
            [6.0, 6.0, 7.0, 9.0],
        ]);
        let output: Array3<f64> = layer.forward_propagation(&x);
        let target: Array3<f64> = Array3::from_shape_vec(
            (3, 3, 2),
            vec![
                30.0, 49.0, 36.0, 62.0, 45.5, 83.5, 22.5, 38.5, 34.5, 58.0, 44.5, 76.5, 28.5, 51.5,
                35.0, 61.5, 44.5, 76.5,
            ],
        )
        .unwrap();
        assert_eq!(output, target);
    }

    #[test]
    fn backward_propagation_should_correct_kernel() -> () {
        let mut layer: ConvLayer = generate_test_conv_layer();
        let x: Array2<f64> = arr2(&[
            [6.0, 6.0, 7.0, 11.0],
            [4.0, 6.0, 7.0, 9.0],
            [2.0, 5.0, 7.0, 9.0],
            [6.0, 6.0, 7.0, 9.0],
        ]);
        layer.forward_propagation(&x);
        let error: Array3<f64> = Array3::from_shape_vec(
            (3, 3, 2),
            vec![
                0.1, 0.2, 0.3, 0.4, 0.9, 1.0, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.5, 1.6, 1.7, 1.8,
                1.9, 2.0,
            ],
        )
        .unwrap();
        let target: Array3<f64> = Array3::from_shape_vec(
            (2, 2, 2),
            vec![47.4, 66.9, 53.2, 66.1, 52.4, 73.6, 58.2, 72.6],
        )
        .unwrap();
        let output: Array3<f64> = layer.backward_propagation(&error, 0.1);
        assert_eq!(target.shape(), output.shape());
        let target_vec: Vec<f64> = target.into_raw_vec();
        let output_vec: Vec<f64> = output.into_raw_vec();
        for i in 0..8 {
            assert!((target_vec[i] - output_vec[i]).powf(2.0) < 0.00001);
        }
        let target_kernels: Array3<f64> = Array3::from_shape_vec(
            (2, 2, 2),
            vec![-3.74, -6.19, -3.82, -4.11, -4.74, -3.86, -1.82, -5.76],
        )
        .unwrap();
        assert_eq!(target_kernels.shape(), layer.kernels.shape());
        let target_kernels_vec: Vec<f64> = target_kernels.into_raw_vec();
        let output_kernels_vec: Vec<f64> = layer.kernels.into_raw_vec();
        for i in 0..8 {
            assert!((target_kernels_vec[i] - output_kernels_vec[i]).powf(2.0) < 0.00001);
        }
    }
}
