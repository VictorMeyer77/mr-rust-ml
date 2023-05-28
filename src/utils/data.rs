use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn one_hot_encoding(x: &Array2<f64>) -> Array2<f64> {
    if x.shape()[1] != 1 {
        panic!(
            "array must have only one column, actually: {}",
            x.shape()[1]
        );
    }
    let max: usize = *x.max().unwrap() as usize;
    let mut one_hot_vec_buffer: Vec<f64> = vec![];
    x.iter().for_each(|&value| {
        let mut one_hot_row_vec: Vec<f64> = vec![0.0; max + 1];
        one_hot_row_vec[value as usize] = 1.0;
        one_hot_vec_buffer.append(&mut one_hot_row_vec);
    });
    Array2::from_shape_vec((x.len(), max + 1), one_hot_vec_buffer).unwrap()
}

pub fn shuffle_arrays(arrays: Vec<&Array2<f64>>) -> Vec<Array2<f64>> {
    let array_lens: Vec<usize> = arrays.iter().map(|array| array.shape()[0]).collect();
    if array_lens
        .iter()
        .filter(|len| **len != array_lens[0])
        .count()
        > 0
    {
        panic!("arrays must have the same column length")
    }
    let mut random_indexes: Vec<usize> = (0..arrays[0].shape()[0]).collect();
    random_indexes.shuffle(&mut thread_rng());
    arrays
        .iter()
        .map(|array| array.select(Axis(0), random_indexes.as_slice()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn one_hot_encoding_should_encode_array() -> () {
        let x: Array2<f64> = arr2(&[[0.0], [1.0], [2.0], [3.0]]);
        let x_one_hot: Array2<f64> = one_hot_encoding(&x);
        assert_eq!(
            x_one_hot,
            arr2(&[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
        )
    }

    #[test]
    #[should_panic(expected = "array must have only one column, actually: 2")]
    fn one_hot_encoding_should_panic_when_array_has_not_one_column() -> () {
        let x: Array2<f64> = arr2(&[[0.0, 3.0]]);
        one_hot_encoding(&x);
    }

    #[test]
    fn shuffle_arrays_should_same_shuffle_vec_of_arrays() -> () {
        let x: Array2<f64> = arr2(&[[0.0], [1.0], [2.0], [3.0]]);
        let y: Array2<f64> = x.clone();
        let shuffle_arrays: Vec<Array2<f64>> = shuffle_arrays(vec![&x, &y]);
        assert_eq!(shuffle_arrays[0], shuffle_arrays[1]);
        assert!(shuffle_arrays[0] != arr2(&[[0.0], [1.0], [2.0], [3.0]]));
    }

    #[test]
    #[should_panic(expected = "arrays must have the same column length")]
    fn shuffle_arrays_should_should_panic_when_arrays_have_not_same_size() -> () {
        let x: Array2<f64> = arr2(&[[0.0], [1.0], [2.0], [3.0]]);
        let y: Array2<f64> = arr2(&[[0.0], [1.0], [2.0]]);
        shuffle_arrays(vec![&x, &y]);
    }
}
