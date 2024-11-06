//! The Scheduler uses a BlockEngine to schedule and automatically batch sequences. The
//! primary method `schedule` returns the batched sequences as inputs, as well as the
//! operations to be executed on the cache by the CacheEngine.

/// The LLMEngine is effectively a wrapper around a ModulePipeline. It contains a Scheduler and a CacheEngine
/// which are used to scheduler and manage the cache during generation requests, respectively.
// pub mod llm_engine;

/// The higher-level manager of the blocks allocated. Operations performed by the block engine do
/// not directly change memory.
pub mod block_engine;
/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
pub mod cache_engine;
pub mod scheduler;
pub mod sequence;

pub use scheduler::{Scheduler, SchedulerConfig, SchedulerOutput};
