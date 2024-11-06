use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device, Result,
};
use candle_vllm::{
    engine::{pipeline::DefaultLoader, ModelLoader},
    SpecificConfig,
};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Huggingface token environment variable (optional). If not specified, load using hf_token_path.
    #[arg(long)]
    pub hf_token: Option<String>,

    /// Huggingface token file (optional). If neither `hf_token` or `hf_token_path` are specified this is used with the value
    /// of `~/.cache/huggingface/token`
    #[arg(long)]
    pub hf_token_path: Option<String>,

    /// Port to serve on (localhost:port)
    #[arg(long)]
    pub port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long)]
    pub verbose: bool,

    #[clap(subcommand)]
    pub command: ModelSelected,

    /// Maximum number of sequences to allow
    #[arg(long, default_value_t = 256)]
    pub max_num_seqs: usize,

    /// Size of a block
    #[arg(long, default_value_t = 32)]
    pub block_size: usize,

    /// if weight_path is passed, it will ignore the model_id
    #[arg(long)]
    pub model_id: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online), path must include last "/"
    #[arg(long)]
    pub weight_path: Option<String>,

    #[arg(long)]
    pub dtype: Option<String>,

    #[arg(long, default_value_t = false)]
    pub cpu: bool,

    /// Available GPU memory for kvcache (MB)
    #[arg(long, default_value_t = 4096)]
    pub kvcache_mem_gpu: usize,

    /// Available CPU memory for kvcache (MB)
    #[arg(long, default_value_t = 4096)]
    pub kvcache_mem_cpu: usize,

    /// Record conversation (default false, the client need to record chat history)
    #[arg(long)]
    pub record_conversation: bool,
}

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

impl std::fmt::Display for ModelSelected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSelected::Llama3 { .. } => write!(f, "llama3"),
            ModelSelected::Qwen2 { .. } => write!(f, "qwen2"),
            ModelSelected::Gemma { .. } => write!(f, "gemma"),
            ModelSelected::Mistral { .. } => write!(f, "mistral"),
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
            model_id.unwrap_or("Qwen/Qwen2.5-0.5B-Instruct".to_string()),
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
    tracing::info!("{:}", path.to_owned() + json_file);
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

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(1)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            tracing::info!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            tracing::info!(
                "Running on CPU, to run on GPU, build this example with `--features cuda`"
            );
        }
        Ok(Device::Cpu)
    }
}
