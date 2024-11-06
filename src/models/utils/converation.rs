/// A struct for managing prompt templates and conversation history.
#[allow(dead_code)]
pub struct Conversation {
    name: String,
    system_message: String,
    system_template: String,
    messages: Vec<Message>,
    offset: usize,
    sep_style: SeparatorStyle,
    stop_criteria: String,
    stop_token_ids: Vec<u32>,
    roles: (String, String),
    sep: String,
    sep2: Option<String>,
}

/// A message in a conversation
pub struct Message((String, Option<String>));

impl Message {
    pub fn new(message: (String, String)) -> Message {
        Message((message.0, Some(message.1)))
    }
}

/// Separator style for default conversation.
#[derive(Default)]
pub enum SeparatorStyle {
    #[default]
    AddColonSingle,
    AddColonTwo,
    AddColonSpaceSingle,
    NoColonSingle,
    NoColonTwo,
    AddNewLineSingle,
    Llama,
    Llama3,
    Phi,
    Qwen2,
    Gemma,
    Mistral,
    Yi,
    StableLM,
    ChatGLM,
    ChatML,
    ChatIntern,
    Dolly,
    RWKV,
    Phoenix,
    Robin,
    FalconChat,
}
