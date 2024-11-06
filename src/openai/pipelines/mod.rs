use crate::openai::sampling_params::Logprobs;
use candle_core::{DType, Device, Tensor, WithDType};
use dirs;
use either::Either;
use std::{env, fs, path::PathBuf, sync::Arc};

use crate::{paged_attention::input_metadata::InputMetadata, try_api};

use super::{conversation::Conversation, responses::APIError, PipelineConfig};
use crate::models::Config;
/// The LLMEngine is effectively a wrapper around a ModulePipeline. It contains a Scheduler and a CacheEngine
/// which are used to scheduler and manage the cache during generation requests, respectively.
pub mod llm_engine;
pub mod pipeline;
use crate::engine::sequence::SequenceGroup;
type TokenOrFinishReason = Either<Logprobs, String>;
use std::collections::VecDeque;
pub trait ModulePipeline: Send + Sync {
    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: &[Vec<usize>],
        kv_cache: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: InputMetadata,
    ) -> Result<Tensor, APIError>;

    fn sample(
        &mut self,
        logits: Tensor,
        groups: &VecDeque<Arc<SequenceGroup>>,
    ) -> Result<Vec<TokenOrFinishReason>, APIError>;

    fn name(&self) -> &str;

    fn tokenizer(&self) -> &TokenOutputStream;

    fn get_conversation(&mut self, with_history: bool) -> &mut dyn Conversation;

    fn get_model_config(&self) -> Config;

    fn get_dtype(&self) -> DType;

    fn device(&self) -> &Device;

    fn reset_decoder(&mut self) -> Option<String>;
}

// TODO(EricLBuehler): Ensure the padding token matches tokenizer
fn _make_tensor_with_pad<D: WithDType>(
    x: Vec<Vec<D>>,
    max_len: usize,
    pad: D,
    device: &Device,
) -> Result<Tensor, APIError> {
    let mut padded_x = Vec::new();
    for mut x_i in x {
        assert!(x_i.len() <= max_len);
        x_i.extend([pad].repeat(max_len - x_i.len()));
        let shape = (1, x_i.len());
        padded_x.push(try_api!(Tensor::from_vec(x_i, shape, device)));
    }
    Tensor::cat(&padded_x[..], 0).map_err(APIError::from)
}

pub(crate) fn get_token(
    hf_token: Option<String>,
    hf_token_path: Option<String>,
) -> Result<String, APIError> {
    Ok(match (hf_token, hf_token_path) {
        (Some(envvar), None) => try_api!(env::var(envvar)).trim().to_string(),
        (None, Some(path)) => try_api!(fs::read_to_string(path)).trim().to_string(),
        (None, None) => try_api!(fs::read_to_string(format!(
            "{}/.cache/huggingface/token",
            dirs::home_dir()
                .ok_or(APIError::new_str("No home directory"))?
                .display()
        )))
        .trim()
        .to_string(),
        _ => {
            return Err(APIError::new_str(
                "Do not specify `hf_token` and `hf_token_path` at the same time.",
            ))
        }
    })
}

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &Vec<PathBuf>;
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
}

pub trait ModelLoader {
    fn download_model(
        &self,
        model_id: String,
        revision: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
    ) -> Result<Box<dyn ModelPaths>, APIError>;

    fn load_model(
        &self,
        paths: Box<dyn ModelPaths>,
        dtype: DType,
        device: Device,
    ) -> Result<(Box<dyn ModulePipeline>, PipelineConfig), APIError>;
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> candle_core::Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> candle_core::Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> candle_core::Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> candle_core::Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
