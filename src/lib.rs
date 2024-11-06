#![warn(clippy::cast_lossless)]

pub mod models;

use std::fmt::Display;

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device, Result,
};
use clap::Subcommand;
use openai::pipelines::{pipeline::DefaultLoader, ModelLoader};

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the llama3 model (default llama3.1-8b).
    Llama3 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the qwen model (default 1.8b).
    Qwen2 {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        top_p: Option<f64>,

        #[arg(long)]
        top_k: Option<usize>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the gemma model (default 2b).
    Gemma {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },

    /// Select the mistral model (default 7b).
    Mistral {
        /// Control the application of repeat penalty for the last n tokens
        #[arg(long)]
        repeat_last_n: Option<usize>,

        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        penalty: Option<f32>,

        #[arg(long)]
        max_gen_tokens: Option<usize>,

        #[arg(long)]
        quant: Option<String>,
    },
}

impl Display for ModelSelected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSelected::Llama3 { .. } => write!(f, "llama3"),
            ModelSelected::Qwen2 { .. } => write!(f, "qwen2"),
            ModelSelected::Gemma { .. } => write!(f, "gemma"),
            ModelSelected::Mistral { .. } => write!(f, "mistral"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecificConfig {
    repeat_last_n: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    penalty: Option<f32>,
    max_gen_tokens: Option<usize>,
    quant: Option<String>,
}

impl SpecificConfig {
    pub fn new(
        repeat_last_n: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        penalty: Option<f32>,
        max_gen_tokens: Option<usize>,
        quant: Option<String>,
    ) -> Self {
        Self {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
        }
    }
}

pub fn get_model_loader(
    selected_model: ModelSelected,
    model_id: Option<String>,
) -> (Box<dyn ModelLoader>, String) {
    if model_id.is_none() {
        tracing::info!(
            "No model id specified, using the default model or specified in the weight_path!"
        );
    }
    match selected_model {
        ModelSelected::Llama3 {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "llama3".to_string(),
            )),
            model_id.unwrap_or("meta-llama/Meta-Llama-3.1-8B-Instruct".to_string()),
        ),
        ModelSelected::Qwen2 {
            repeat_last_n,
            temperature,
            top_k,
            top_p,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    top_k,
                    top_p,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "qwen2".to_string(),
            )),
            model_id.unwrap_or("Qwen/Qwen2.5-1.5B-Instruct".to_string()),
        ),
        ModelSelected::Gemma {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "gemma".to_string(),
            )),
            model_id.unwrap_or("google/gemma-2b-it".to_string()),
        ),
        ModelSelected::Mistral {
            repeat_last_n,
            temperature,
            penalty,
            max_gen_tokens,
            quant,
        } => (
            Box::new(DefaultLoader::new(
                SpecificConfig::new(
                    repeat_last_n,
                    temperature,
                    None,
                    None,
                    penalty,
                    max_gen_tokens,
                    quant,
                ),
                "mistral".to_string(),
            )),
            model_id.unwrap_or("mistralai/Mistral-7B-Instruct-v0.3".to_string()),
        ),
    }
}

pub fn hub_load_local_safetensors(
    path: &String,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    println!("{:}", path.to_owned() + json_file);
    let jsfile = std::fs::File::open(path.to_owned() + json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&jsfile).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = Vec::<std::path::PathBuf>::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(0, (path.to_owned() + file).into());
        }
    }
    Ok(safetensors_files)
}

pub mod backend;
pub mod engine;
pub mod openai;
pub mod paged_attention;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
