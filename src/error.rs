use thiserror::Error;

use crate::openai::responses::APIError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("API error: {0}")]
    Other(String),
}

impl From<APIError> for Error {
    fn from(e: APIError) -> Self {
        Self::Other(e.to_string())
    }
}
