use crate::report::html::{generate_full_html, generate_images_html, generate_resume_html};
use crate::report::plot::generate_2d_plot;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

pub struct Report {
    steps: Vec<usize>,
    train_accuracies: Vec<f64>,
    train_losses: Vec<f64>,
    test_accuracies: Vec<f64>,
    output_directory: String,
}

impl Report {
    pub fn build(output_directory: &str) -> Report {
        Report {
            steps: vec![],
            train_accuracies: vec![],
            train_losses: vec![],
            test_accuracies: vec![],
            output_directory: output_directory.to_string(),
        }
    }

    pub fn add_data(
        &mut self,
        step: usize,
        train_accuracy: f64,
        train_loss: f64,
        test_accuracy: Option<f64>,
    ) -> () {
        self.steps.push(step);
        self.train_accuracies.push(train_accuracy);
        self.train_losses.push(train_loss);
        if test_accuracy.is_some() {
            self.test_accuracies.push(test_accuracy.unwrap());
        }
    }

    pub fn generate(
        &self,
        network_name: &str,
        start_time: Instant,
        epochs: usize,
        x_train_shape: &[usize],
        y_train_shape: &[usize],
        x_test_shape: Option<&[usize]>,
        y_test_shape: Option<&[usize]>,
        accuracy_function: &str,
        loss_function: &str,
    ) -> () {
        let image_directory: String = Path::new(self.output_directory.as_str())
            .join((self.train_accuracies.len() - 1).to_string())
            .join("static")
            .to_str()
            .unwrap()
            .to_string();
        fs::create_dir_all(image_directory.as_str()).unwrap();
        self.generate_plots(image_directory.as_str(), accuracy_function, loss_function);
        let mut images: Vec<&str> = vec!["static/train accuracy.png", "static/train loss.png"];
        if !self.test_accuracies.is_empty() {
            images.push("static/test accuracy.png");
        }
        let image_html: String = generate_images_html(images);

        let test_accuracy: Option<f64> = if self.test_accuracies.len() > 0 {
            Some(*self.test_accuracies.last().unwrap())
        } else {
            None
        };
        let resume_html: String = generate_resume_html(
            network_name,
            start_time.elapsed().as_secs(),
            self.steps.len() - 1,
            epochs,
            x_train_shape,
            y_train_shape,
            x_test_shape,
            y_test_shape,
            accuracy_function,
            loss_function,
            *self.train_accuracies.last().unwrap(),
            *self.train_losses.last().unwrap(),
            test_accuracy,
        );

        let report_html: String = generate_full_html(resume_html, image_html);
        let mut report_file: File = File::create(
            Path::new(self.output_directory.as_str())
                .join((self.train_accuracies.len() - 1).to_string())
                .join("report.html"),
        )
        .unwrap();
        report_file.write_all(report_html.as_bytes()).unwrap();
    }

    fn generate_plots(
        &self,
        output_directory: &str,
        accuracy_function: &str,
        loss_function: &str,
    ) -> () {
        let step_float: Vec<f64> = self.steps.iter().map(|&step| step as f64).collect();
        generate_2d_plot(
            output_directory,
            &step_float,
            &self.train_accuracies,
            "epochs",
            accuracy_function,
            "train accuracy",
        );
        generate_2d_plot(
            output_directory,
            &step_float,
            &self.train_losses,
            "epochs",
            loss_function,
            "train loss",
        );
        if !self.test_accuracies.is_empty() {
            generate_2d_plot(
                output_directory,
                &step_float,
                &self.test_accuracies,
                "epochs",
                accuracy_function,
                "test accuracy",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_data_should_push_in_vectors() -> () {
        let mut report_with_test: Report = Report::build("test_report");
        let mut report_without_test: Report = Report::build("test_report");

        for i in 0..5 {
            report_with_test.add_data(i, i as f64 * 1.5, i as f64 * 2.0, Some(i as f64 * 5.0));
            report_without_test.add_data(i, i as f64 * 1.5, i as f64 * 2.0, None);
        }

        assert_eq!(report_with_test.steps, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            report_with_test.train_accuracies,
            vec![0.0, 1.5, 3.0, 4.5, 6.0]
        );
        assert_eq!(report_with_test.train_losses, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(
            report_with_test.test_accuracies,
            vec![0.0, 5.0, 10.0, 15.0, 20.0]
        );
        assert_eq!(report_without_test.steps, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            report_without_test.train_accuracies,
            vec![0.0, 1.5, 3.0, 4.5, 6.0]
        );
        assert_eq!(
            report_without_test.train_losses,
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert!(report_without_test.test_accuracies.is_empty());
    }

    #[test]
    fn generate_should_create_full_report() -> () {
        let mut report: Report = Report::build("test_report");
        let now: Instant = Instant::now();

        for i in 0..5 {
            report.add_data(i, i as f64 * 1.5, i as f64 * 2.0, Some(i as f64 * 5.0));
        }

        report.generate(
            "MLP",
            now,
            1000,
            &[5, 5],
            &[8, 8],
            Some(&[6, 8]),
            Some(&[8, 5]),
            "Categorical Accuracy",
            "MSE",
        );

        assert!(Path::new("test_report").exists());
        fs::remove_dir_all("test_report").unwrap();
    }

    #[test]
    fn generate_plots_should_create_png() -> () {
        let mut report: Report = Report::build("test_report");

        for i in 0..5 {
            report.add_data(i, i as f64 * 1.5, i as f64 * 2.0, Some(i as f64 * 5.0));
        }

        report.generate_plots(".", "Categorical Accuracy", "MSE");

        assert!(Path::new("./train loss.png").exists());
        assert!(Path::new("./train accuracy.png").exists());
        assert!(Path::new("./test accuracy.png").exists());

        fs::remove_file("./train loss.png").unwrap();
        fs::remove_file("./train accuracy.png").unwrap();
        fs::remove_file("./test accuracy.png").unwrap();
    }
}
