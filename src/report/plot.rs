use plotters::prelude::*;
use std::path;

pub fn generate_2d_plot(
    output_directory: &str,
    x: &Vec<f64>,
    y: &Vec<f64>,
    x_label: &str,
    y_label: &str,
    title: &str,
) -> () {
    if x.len() != y.len() {
        panic!(
            "vectors must have the same length: {} != {}",
            x.len(),
            y.len()
        )
    }

    let x_min: f64 = x.iter().map(|&x| x as i32).min().unwrap() as f64;
    let x_max: f64 = x.iter().map(|&x| x as i32).max().unwrap() as f64;
    let y_min: f64 = y.iter().map(|&y| y as i32).min().unwrap() as f64;
    let y_max: f64 = y.iter().map(|&y| y as i32).max().unwrap() as f64;

    let path: String = path::Path::new(output_directory)
        .join(title.to_string() + ".png")
        .to_str()
        .unwrap()
        .to_string();

    let root_area = BitMapBackend::new(&path, (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d((x_min - 1.0)..(x_max + 1.0), (y_min - 1.0)..(y_max + 1.0))
        .unwrap();

    ctx.configure_mesh()
        .y_desc(y_label)
        .x_desc(x_label)
        .draw()
        .unwrap();

    ctx.draw_series(LineSeries::new((0..x.len()).map(|i| (x[i], y[i])), &BLUE))
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn generate_2d_plot_should_create_png() -> () {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y: Vec<f64> = vec![2.0, 4.0, 6.0];
        generate_2d_plot(".", &x, &y, "x_label", "y_label", "test");
        assert!(path::Path::new("./test.png").exists());
        fs::remove_file("./test.png").unwrap();
    }

    #[test]
    #[should_panic(expected = "vectors must have the same length: 3 != 4")]
    fn generate_2d_plot_should_panic_when_vectors_have_not_same_length() -> () {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
        generate_2d_plot(".", &x, &y, "x_label", "y_label", "test");
    }
}
