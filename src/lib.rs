pub mod backend;
pub mod engine;
pub mod error;
pub mod models;
pub mod openai;
pub mod paged_attention;

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
