use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub trait Network<T> {
    fn get_name(&self) -> String;

    fn from_json(json_str: &str) -> Result<T, Box<dyn Error>>;

    fn to_json(&self) -> Result<String, Box<dyn Error>>;

    fn save(&self, output_directory: &str, model_name: String) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(Path::new(output_directory).join(model_name + ".json"))?;
        file.write_all(self.to_json()?.as_bytes())?;
        Ok(())
    }

    fn load(path: &str) -> Result<T, Box<dyn Error>>;
}
