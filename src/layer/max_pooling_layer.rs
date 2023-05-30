use ndarray::{s, Array2, ArrayView2, Array3};

pub struct MaxPoolingLayer<'a> {
    input: Option<&'a Array3<f64>>,
    kernel_size: usize,
}

impl<'a> MaxPoolingLayer<'a> {
    fn image_to_patches(&mut self, x: &'a Array3<f64>) -> Vec<(ArrayView2<f64>, usize, usize)> {
        self.input = Some(&x);
        let shape: &[usize] = x.shape();
        let mut patches_buffer: Vec<(ArrayView2<f64>, usize, usize)> = vec![];
        for h in 0..(shape[0] / self.kernel_size) {
            for w in 0..(shape[1] / self.kernel_size) {
                patches_buffer.push((
                    x.slice(s![
                        (h * self.kernel_size)..(h * self.kernel_size + self.kernel_size),
                        (w * self.kernel_size)..(w * self.kernel_size + self.kernel_size)
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
        let max_pooling_output: Array3<f64> = Array3::zeros((x_shape[0] / self.kernel_size, x_shape[1] / self.kernel_size, x_shape[2]));

    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    fn generate_test_maw_pooling_layer<'a>() -> MaxPoolingLayer<'a> {
        MaxPoolingLayer {
            input: None,
            kernel_size: 2,
        }
    }

    #[test]
    fn build_should_initialize_layer() -> () {

    }


    #[test]
    fn image_to_patches_should_return_vec_of_pattern() -> () {
        let mut layer: MaxPoolingLayer = generate_test_maw_pooling_layer();
        let x: Array3<f64> = arr3(&[
            [6.0, 6.0, 7.0, 11.0],
            [4.0, 6.0, 7.0, 9.0],
            [2.0, 5.0, 7.0, 9.0],
            [6.0, 6.0, 7.0, 9.0],
        ]);
        println!("{:?}", x);
        /*let output: Vec<(ArrayView2<f64>, usize, usize)> = layer.image_to_patches(&x);
        let target: Vec<(Array2<f64>, usize, usize)> = vec![
            (
                Array2::from_shape_vec((2, 2), vec![6.0, 6.0, 4.0, 6.0]).unwrap(),
                0,
                0,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![7.0, 11.0, 7.0, 9.0]).unwrap(),
                0,
                1,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![2.0, 5.0, 6.0, 6.0]).unwrap(),
                1,
                0,
            ),
            (
                Array2::from_shape_vec((2, 2), vec![7.0, 9.0, 7.0, 9.0]).unwrap(),
                1,
                1,
            )
        ];
        assert_eq!(output.len(), target.len());
        for i in 0..target.len() {
            assert_eq!(output[i].0, target[i].0.view());
            assert_eq!(output[i].1, target[i].1);
            assert_eq!(output[i].2, target[i].2);
        }*/
    }
}