use super::{Config, QuantConfig};
use crate::openai::models::linear::{
    linear_no_bias_x as linear_no_bias, linear_x as linear, LinearX as Linear,
};
use crate::paged_attention::input_metadata::InputMetadata;
use crate::paged_attention::PagedAttention;
use crate::SpecificConfig;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, LayerNorm, VarBuilder};
use either::Either;
use std::iter::zip;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct StableLMConfig {
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub use_cache: bool,
    pub use_qkv_bias: Option<bool>, // Used in StableLM-2
    pub partial_rotary_factor: Option<f32>,
    pub rope_pct: Option<f32>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub sliding_window: Option<usize>,
    pub quantization_config: Option<QuantConfig>,
}

impl StableLMConfig {
    pub fn into_config(
        self,
        use_flash_attn: bool,
        kv_cache_dtype: DType,
        scfg: &SpecificConfig,
    ) -> Config {
        Config {
            hidden_size: self.hidden_size,
            head_dim: Some(self.hidden_size / self.num_attention_heads),
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: super::TokenID(Either::Left(Some(self.bos_token_id as u32))),
            eos_token_id: super::TokenID(Either::Left(Some(self.bos_token_id as u32))),
            max_seq_len: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            hidden_act: Some(self.hidden_act),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            rope_scaling: None,
            original_max_position_embeddings: None,
            attention_bias: false,
            partial_rotary_factor: Some(
                self.partial_rotary_factor
                    .unwrap_or(self.rope_pct.unwrap_or(0.25)),
            ),
            qk_layer_rms_norm: None,
            kv_cache_dtype,
            use_qkv_bias: Some(self.use_qkv_bias.unwrap_or(false)),
            custom_stop_tokens: None,
            specific_config: scfg.clone(),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: self.quantization_config,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    pub(crate) fn new(_dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let dim = (cfg.partial_rotary_factor.unwrap() * head_dim as f32) as usize;
        let max_seq_len = cfg.max_seq_len;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            dim,
        })
    }

    fn apply_rotary_emb(&self, xs: &Tensor, input_positions: &[Vec<usize>]) -> Result<Tensor> {
        let (b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let mut embeds = Vec::new();
        for (b, seqlen_offset) in zip(0..b_size, input_positions) {
            let xs_rot = xs.narrow(3, 0, self.dim)?.contiguous()?;
            let xs_pass = xs.narrow(3, self.dim, _headdim - self.dim)?;
            let c = self.cos.narrow(0, seqlen_offset[0], seq_len)?;
            let s = self.sin.narrow(0, seqlen_offset[0], seq_len)?;
            let xs_rot = xs_rot.narrow(0, b, 1)?;
            let xs_pass = xs_pass.narrow(0, b, 1)?;

            let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, &c, &s)?;
            let embed = Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()?;
            embeds.push(embed);
        }
        Tensor::cat(&embeds, 0)
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
    span: tracing::Span,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("gate_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let up_proj = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("up_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let down_proj = linear_no_bias(
            intermediate_sz,
            hidden_sz,
            vb.pp("down_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act.unwrap_or(Activation::Silu),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    attn: PagedAttention,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;

        let linear_layer = if cfg.use_qkv_bias.unwrap_or(false) {
            linear
        } else {
            linear_no_bias
        };

        let q_proj = linear_layer(
            hidden_sz,
            num_heads * head_dim,
            vb.pp("q_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let k_proj = linear_layer(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let v_proj = linear_layer(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        let o_proj = linear_no_bias(
            num_heads * head_dim,
            hidden_sz,
            vb.pp("o_proj"),
            &cfg.specific_config.quant,
            &cfg.quantization_config,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            attn: PagedAttention::new(
                cfg.num_attention_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(cfg.num_key_value_heads),
                None,
                vb.device().clone(),
                None,
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let (q, k, v) = if seq_len == 1 {
            //no need transpose for seq_len == 1, change reshape dim
            let q = query_states.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = key_states.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = value_states.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        } else {
            let q = query_states
                .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = key_states
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = value_states
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v.contiguous()?)
        };

        let q = self
            .rotary_emb
            .apply_rotary_emb(&q.to_dtype(DType::F32)?, input_positions)?;

        let k = self
            .rotary_emb
            .apply_rotary_emb(&k.to_dtype(DType::F32)?, input_positions)?;

        let q = q.to_dtype(v.dtype())?;
        let k = k.to_dtype(v.dtype())?;

        let y = self.attn.forward(
            &q,
            &k,
            &v,
            attention_mask,
            cache.map(|(k_, _)| k_.clone()),
            cache.map(|(_, v_)| v_.clone()),
            input_metadata,
            None,
        )?;

        let y = if attention_mask.is_some() {
            y.transpose(1, 2)?
                .reshape(&[b_sz, seq_len, self.hidden_size])?
        } else {
            y.reshape(&[b_sz, seq_len, self.hidden_size])?
        };
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &[Vec<usize>],
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct StableLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    cfg: Config,
}

impl StableLM {
    pub fn new(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("lm_head"),
            &None, //no quant for lm_head
            &None,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
    }

    fn prepare_decoder_attention_mask(&self, b_size: usize, tgt_len: usize) -> Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        mask.expand((b_size, 1, tgt_len, tgt_len))?
            .to_dtype(self.dtype)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        input_positions: &[Vec<usize>],
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter_mut()) {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?
            }
        } else {
            for layer in self.layers.iter_mut() {
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?
            }
        }
        xs.i((.., seq_len - 1, ..))?
            .apply(&self.norm)?
            .apply(&self.lm_head)?
            .to_dtype(DType::F32)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
