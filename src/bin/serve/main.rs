mod command;

use command::{device, get_model_loader, hub_load_local_safetensors, Args};

use std::path::Path;
use std::sync::Arc;

use axum::{
    http::{self, Method},
    routing::post,
    Router,
};
use candle_core::{DType, Device};
use candle_vllm::{
    engine::{
        cache_engine::CacheConfig, llm_engine::LLMEngine, pipeline::DefaultModelPaths,
        SchedulerConfig,
    },
    error::Result,
    models::Config,
    openai::{handlers::chat_completions, responses::APIError, OpenAIServerData},
};
use clap::Parser;
use tokio::sync::Notify;
use tower_http::cors::{AllowOrigin, CorsLayer};

const SIZE_IN_MB: usize = 1024 * 1024;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let (loader, model_id) = get_model_loader(args.command, args.model_id.clone());

    let paths = match &args.weight_path {
        Some(path) => Box::new(DefaultModelPaths {
            tokenizer_filename: (path.to_owned() + "tokenizer.json").into(),
            config_filename: (path.to_owned() + "config.json").into(),
            filenames: if Path::new(&(path.to_owned() + "model.safetensors.index.json")).exists() {
                hub_load_local_safetensors(path, "model.safetensors.index.json").unwrap()
            } else {
                //a single weight file case
                let mut safetensors_files = Vec::<std::path::PathBuf>::new();
                safetensors_files.insert(0, (path.to_owned() + "model.safetensors").into());
                safetensors_files
            },
        }),
        _ => {
            if args.hf_token.is_none() && args.hf_token_path.is_none() {
                //no token provided
                let token_path = format!(
                    "{}/.cache/huggingface/token",
                    dirs::home_dir()
                        .ok_or(APIError::new_str("No home directory"))?
                        .display()
                );
                if !Path::new(&token_path).exists() {
                    //also no token cache
                    use std::io::Write;
                    let mut input_token = String::new();
                    tracing::info!("Please provide your huggingface token to download model:\n");
                    std::io::stdin()
                        .read_line(&mut input_token)
                        .expect("Failed to read token!");
                    std::fs::create_dir_all(Path::new(&token_path).parent().unwrap()).unwrap();
                    let mut output = std::fs::File::create(token_path).unwrap();
                    write!(output, "{}", input_token.trim()).expect("Failed to save token!");
                }
            }
            loader.download_model(model_id, None, args.hf_token, args.hf_token_path)?
        }
    };

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => panic!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    let device = device(args.cpu).unwrap();
    let model = loader.load_model(paths, dtype, device)?;
    let config: Config = model.0.get_model_config();
    let dsize = config.kv_cache_dtype.size_in_bytes();
    let num_gpu_blocks = args.kvcache_mem_gpu * SIZE_IN_MB
        / dsize
        / args.block_size
        / config.num_key_value_heads
        / config.get_head_size()
        / config.num_hidden_layers
        / 2;
    let num_cpu_blocks = args.kvcache_mem_cpu * SIZE_IN_MB
        / dsize
        / args.block_size
        / config.num_key_value_heads
        / config.get_head_size()
        / config.num_hidden_layers
        / 2;
    let cache_config = CacheConfig {
        block_size: args.block_size,
        num_gpu_blocks: Some(num_gpu_blocks),
        num_cpu_blocks: Some(num_cpu_blocks),
        fully_init: true,
        dtype: config.kv_cache_dtype,
    };
    tracing::info!("Cache config {:?}", cache_config);
    let finish_notify = Arc::new(Notify::new());
    let llm_engine = LLMEngine::new(
        model.0,
        SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
        },
        cache_config,
        Arc::new(Notify::new()),
        finish_notify.clone(),
    )?;

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
        model: llm_engine,
        record_conversation: args.record_conversation,
        device: Device::Cpu,
        finish_notify: finish_notify.clone(),
    };

    // start openai server
    tracing::info!("Server started at http://127.0.0.1:{}.", args.port);
    let allow_origin = AllowOrigin::any();
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    let app = Router::new()
        .layer(cors_layer)
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(Arc::new(server_data));

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port))
        .await
        .map_err(|e| APIError::new(e.to_string()))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| APIError::new(e.to_string()))?;

    Ok(())
}
