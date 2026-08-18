use llama_cpp_4::{
    model::LlamaChatMessage, AddBos, LlamaBackend, LlamaBatch, LlamaContext, LlamaModel,
    LlamaSampler, Special,
};

use crate::types::{ChatMessage, ChatSettings, ErrorPayload};

pub fn generate(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    _backend: &LlamaBackend,
    messages: Vec<ChatMessage>,
    settings: ChatSettings,
) -> Result<String, ErrorPayload> {
    let chat_messages = build_chat_messages(&messages, &settings)?;

    let prompt = match model.apply_chat_template(None, &chat_messages, true) {
        Ok(prompt) => prompt,
        Err(_) => build_prompt_fallback(&messages, &settings),
    };

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|error| ErrorPayload::new("tokenization_failed", error.to_string(), None))?;

    if tokens.is_empty() {
        return Ok(String::new());
    }

    let sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(settings.temperature as f32),
        LlamaSampler::dist(0),
    ]);

    let mut n_past = 0usize;

    let batch_size = 512;
    for chunk in tokens.chunks(batch_size) {
        let mut batch = LlamaBatch::new(chunk.len(), 1);

        for (i, &token) in chunk.iter().enumerate() {
            let position = n_past + i;
            let logits = position + 1 == tokens.len();
            batch
                .add(token, position as i32, &[0], logits)
                .map_err(|error| ErrorPayload::new("batch_add_failed", error.to_string(), None))?;
        }

        context
            .decode(&mut batch)
            .map_err(|error| ErrorPayload::new("context_decode_failed", error.to_string(), None))?;

        n_past += chunk.len();
    }

    let mut generated = String::new();
    let eos_token = model.token_eos();
    let max_new_tokens = settings.max_tokens as usize;
    let mut num_generated = 0;

    while num_generated < max_new_tokens {
        let next_token = sampler.sample(context, (n_past - 1) as i32);

        if next_token == eos_token {
            break;
        }

        if let Ok(bytes) = model.token_to_bytes(next_token, Special::Plaintext) {
            generated.push_str(&String::from_utf8_lossy(&bytes));
        }

        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(next_token, n_past as i32, &[0], true)
            .map_err(|error| ErrorPayload::new("batch_add_failed", error.to_string(), None))?;

        context
            .decode(&mut batch)
            .map_err(|error| ErrorPayload::new("context_decode_failed", error.to_string(), None))?;

        n_past += 1;
        num_generated += 1;
    }

    Ok(generated)
}

fn build_chat_messages(
    messages: &[ChatMessage],
    settings: &ChatSettings,
) -> Result<Vec<LlamaChatMessage>, ErrorPayload> {
    let mut chat = Vec::new();

    if !settings.system_prompt.is_empty() {
        chat.push(
            LlamaChatMessage::new("system".into(), settings.system_prompt.clone()).map_err(
                |error| ErrorPayload::new("chat_message_new_failed", error.to_string(), None),
            )?,
        );
    }

    for message in messages {
        chat.push(
            LlamaChatMessage::new(message.role.clone(), message.content.clone()).map_err(
                |error| ErrorPayload::new("chat_message_new_failed", error.to_string(), None),
            )?,
        );
    }

    Ok(chat)
}

fn build_prompt_fallback(messages: &[ChatMessage], settings: &ChatSettings) -> String {
    let mut prompt = String::new();

    if !settings.system_prompt.is_empty() {
        prompt.push_str("System: ");
        prompt.push_str(&settings.system_prompt);
        prompt.push_str("\n\n");
    }

    for message in messages {
        prompt.push_str(&format!("{}: {}\n", message.role, message.content));
    }

    prompt.push_str("assistant: ");
    prompt
}
