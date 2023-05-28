pub fn generate_images_html(images: Vec<&str>) -> String {
    let mut html: String = String::new();
    for image in images {
        html.push_str(
            format!(
                "<div class=\"imgBlock\">
  <figure>
    <img src=\"{0}\" width=\"700\" height=\"400\" />
  </figure>
</div>\n",
                image,
            )
            .as_str(),
        );
    }
    html
}

pub fn generate_resume_html(
    network_name: &str,
    duration: u64,
    step: usize,
    epochs: usize,
    x_train_shape: &[usize],
    y_train_shape: &[usize],
    x_test_shape: Option<&[usize]>,
    y_test_shape: Option<&[usize]>,
    accuracy_function: &str,
    loss_function: &str,
    train_accuracy: f64,
    train_loss: f64,
    test_accuracy: Option<f64>,
) -> String {
    let mut html: String = "<div class=\"tableBlock\">\n\t<table>".to_string();
    html.push_str(&format!(
        "\n\t\t<tr><td>Type</td><td>{}</td></tr>",
        network_name
    ));
    html.push_str(&format!(
        "\n\t\t<tr><td>Duration (s)</td><td>{:?}</td></tr>",
        duration
    ));
    html.push_str(&format!(
        "\n\t\t<tr><td>Epochs</td><td>{}/{}</td></tr>",
        step, epochs
    ));
    html.push_str(&format!(
        "\n\t\t<tr><td>X Train</td><td>{:?}</td></tr>",
        x_train_shape
    ));
    html.push_str(&format!(
        "\n\t\t<tr><td>Y Train</td><td>{:?}</td></tr>",
        y_train_shape
    ));
    if x_test_shape.is_some() {
        html.push_str(&format!(
            "\n\t\t<tr><td>X Test</td><td>{:?}</td></tr>",
            x_test_shape.unwrap()
        ));
    }
    if y_test_shape.is_some() {
        html.push_str(&format!(
            "\n\t\t<tr><td>Y Test</td><td>{:?}</td></tr>",
            y_test_shape.unwrap()
        ));
    }
    html.push_str(&format!(
        "\n\t\t<tr><td>Accuracy Function</td><td>{}</td></tr>",
        accuracy_function
    ));
    html.push_str(&format!(
        "\n\t\t<tr><td>Loss Function</td><td>{}</td></tr>",
        loss_function
    ));
    html.push_str(&format!(
        "\n\t\t<tr><td>Train Accuracy</td><td>{:?}</td></tr>",
        train_accuracy
    ));
    html.push_str(&format!(
        "\n\t\t<tr><td>Train Loss</td><td>{:?}</td></tr>",
        train_loss
    ));
    if test_accuracy.is_some() {
        html.push_str(&format!(
            "\n\t\t<tr><td>Test Accuracy</td><td>{:?}</td></tr>",
            test_accuracy.unwrap()
        ));
    }
    html.push_str(&format!("\n\t</table>\n</div>\n"));
    html
}

pub fn generate_full_html(resume_html: String, image_html: String) -> String {
    format!(
        "<!DOCTYPE html>
<html>
<head>
<title>Report</title>
<style>
table, th, td {{
  border: 1px solid;
}}
</style>
</head>
<body>
<h1>Report</h1>
{}
{}
</body>
</html>",
        resume_html, image_html
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_images_html_should_format_list_of_image() -> () {
        let images_html: String = generate_images_html(vec!["image1.png", "image2.png"]);
        assert_eq!(
            images_html,
            "<div class=\"imgBlock\">
  <figure>
    <img src=\"image1.png\" width=\"700\" height=\"400\" />
  </figure>
</div>
<div class=\"imgBlock\">
  <figure>
    <img src=\"image2.png\" width=\"700\" height=\"400\" />
  </figure>
</div>
"
        )
    }

    #[test]
    fn generate_resume_html_should_create_resume_with_test() -> () {
        let resume_html: String = generate_resume_html(
            "MLP",
            56,
            5,
            100,
            &[10, 1],
            &[5, 1],
            Some(&[6, 1]),
            Some(&[4, 1]),
            "Categorical Accuracy",
            "MSE",
            0.66,
            0.33,
            Some(0.8),
        );
        assert_eq!(
            resume_html,
            "<div class=\"tableBlock\">
	<table>
		<tr><td>Type</td><td>MLP</td></tr>
		<tr><td>Duration (s)</td><td>56</td></tr>
		<tr><td>Epochs</td><td>5/100</td></tr>
		<tr><td>X Train</td><td>[10, 1]</td></tr>
		<tr><td>Y Train</td><td>[5, 1]</td></tr>
		<tr><td>X Test</td><td>[6, 1]</td></tr>
		<tr><td>Y Test</td><td>[4, 1]</td></tr>
		<tr><td>Accuracy Function</td><td>Categorical Accuracy</td></tr>
		<tr><td>Loss Function</td><td>MSE</td></tr>
		<tr><td>Train Accuracy</td><td>0.66</td></tr>
		<tr><td>Train Loss</td><td>0.33</td></tr>
		<tr><td>Test Accuracy</td><td>0.8</td></tr>
	</table>
</div>
"
        )
    }

    #[test]
    fn generate_resume_html_should_create_resume_without_test() -> () {
        let resume_html: String = generate_resume_html(
            "MLP",
            56,
            5,
            100,
            &[10, 1],
            &[5, 1],
            None,
            None,
            "Categorical Accuracy",
            "MSE",
            0.66,
            0.33,
            None,
        );
        assert_eq!(
            resume_html,
            "<div class=\"tableBlock\">
	<table>
		<tr><td>Type</td><td>MLP</td></tr>
		<tr><td>Duration (s)</td><td>56</td></tr>
		<tr><td>Epochs</td><td>5/100</td></tr>
		<tr><td>X Train</td><td>[10, 1]</td></tr>
		<tr><td>Y Train</td><td>[5, 1]</td></tr>
		<tr><td>Accuracy Function</td><td>Categorical Accuracy</td></tr>
		<tr><td>Loss Function</td><td>MSE</td></tr>
		<tr><td>Train Accuracy</td><td>0.66</td></tr>
		<tr><td>Train Loss</td><td>0.33</td></tr>
	</table>
</div>
"
        )
    }

    #[test]
    fn generate_full_html_should_create_full_report() -> () {
        let resume_html: String = generate_resume_html(
            "MLP",
            56,
            5,
            100,
            &[10, 1],
            &[5, 1],
            Some(&[6, 1]),
            Some(&[4, 1]),
            "Categorical Accuracy",
            "MSE",
            0.66,
            0.33,
            Some(0.8),
        );
        let images_html: String = generate_images_html(vec!["image1.png", "image2.png"]);
        let report_html: String = generate_full_html(resume_html, images_html);
        assert_eq!(
            report_html,
            "<!DOCTYPE html>
<html>
<head>
<title>Report</title>
<style>
table, th, td {
  border: 1px solid;
}
</style>
</head>
<body>
<h1>Report</h1>
<div class=\"tableBlock\">
	<table>
		<tr><td>Type</td><td>MLP</td></tr>
		<tr><td>Duration (s)</td><td>56</td></tr>
		<tr><td>Epochs</td><td>5/100</td></tr>
		<tr><td>X Train</td><td>[10, 1]</td></tr>
		<tr><td>Y Train</td><td>[5, 1]</td></tr>
		<tr><td>X Test</td><td>[6, 1]</td></tr>
		<tr><td>Y Test</td><td>[4, 1]</td></tr>
		<tr><td>Accuracy Function</td><td>Categorical Accuracy</td></tr>
		<tr><td>Loss Function</td><td>MSE</td></tr>
		<tr><td>Train Accuracy</td><td>0.66</td></tr>
		<tr><td>Train Loss</td><td>0.33</td></tr>
		<tr><td>Test Accuracy</td><td>0.8</td></tr>
	</table>
</div>

<div class=\"imgBlock\">
  <figure>
    <img src=\"image1.png\" width=\"700\" height=\"400\" />
  </figure>
</div>
<div class=\"imgBlock\">
  <figure>
    <img src=\"image2.png\" width=\"700\" height=\"400\" />
  </figure>
</div>

</body>
</html>"
        );
    }
}
