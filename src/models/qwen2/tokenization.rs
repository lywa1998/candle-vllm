#![allow(unused)]

use crate::error::Result;
use crate::models::utils::Conversation;

use tokenizers::AddedToken;

const VOCAB_FILES_NAMES: [&str; 3] = ["vocab.json", "merges.txt", "tokenizer.json"];

const MAX_MODEL_INPUT_SIZES: usize = 32768;

/// Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
/// Byte-Pair-Encoding.
///
/// Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
/// be encoded differently whether it is at the beginning of the sentence (without space) or not:
pub struct Qwen2Tokenizer {}

impl Qwen2Tokenizer {
    pub fn new() -> Result<Self> {
        let eos_token = AddedToken::from("<|endoftext|>", true);
        let unk_token = AddedToken::from("<|endoftext|>", true);
        let pad_token = AddedToken::from("<|endoftext|>", true);
        todo!()
    }

    pub fn apply_chat_template(&self, conversation: &Conversation) {
        todo!()
    }
}
