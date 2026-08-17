use std::num::NonZeroU32;

use llama_cpp_4::prelude::*;

use crate::types::{ChatMessage, ChatSettings, ErrorPayload};

pub fn generate(
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    backend: &LlamaBackend,
    messages: Vec<ChatMessage>,
    settings: ChatSettings,
) -> Result<String, ErrorPayload> {
    // Get the model's chat template if available
    let chat_template = model.chat_template();
    
    // Build the prompt from messages using the template
    let prompt = if let Some(template) = chat_template {
        build_prompt_with_template(template, &messages, &settings)?
    } else {
        build_prompt_fallback(&messages, &settings)
    };

    // Tokenize the prompt
    let tokens = model
        .tokenize(&prompt, false)
        .map_err(|error| {
            ErrorPayload::new(
                "tokenization_failed",
                error.to_string(),
                None,
            )
        })?;

    // Create a sampler for generation
    let mut sampler = model.sampler_default()
        .temperature(settings.temperature as f32);

    // Prepare the context
    context.clear();

    // Process tokens in batches (prefill)
    let batch_size = 512;
    for chunk in tokens.chunks(batch_size) {
        let mut batch = LlamaBatch::new(chunk.len(), 1);
        
        for (i, &token) in chunk.iter().enumerate() {
            batch.add(token, i as i32, &[0], false);
        }
        
        context
            .decode(backend, batch)
            .map_err(|error| {
                ErrorPayload::new(
                    "context_decode_failed",
                    error.to_string(),
                    None,
                )
            })?;
    }

    // Generate tokens until EOS or max_tokens
    let mut generated = String::new();
    let eos_token = model.token_eos();
    let max_new_tokens = settings.max_tokens as usize;
    let mut num_generated = 0;

    loop {
        if num_generated >= max_new_tokens {
            break;
        }

        let next_token = context
            .sample(&sampler)
            .map_err(|error| {
                ErrorPayload::new(
                    "sampling_failed",
                    error.to_string(),
                    None,
                )
            })?;

        if next_token == eos_token {
            break;
        }

        // Decode the token to string
        if let Ok(text) = model.token_to_piece(next_token) {
            generated.push_str(&text);
        }

        // Add token to context for next iteration
        let mut batch = LlamaBatch::new(1, 1);
        batch.add(next_token, 0, &[0], true);
        
        context
            .decode(backend, batch)
            .map_err(|error| {
                ErrorPayload::new(
                    "context_decode_failed",
                    error.to_string(),
                    None,
                )
            })?;

        num_generated += 1;
    }

    Ok(generated)
}

fn build_prompt_with_template(
    template: &str,
    messages: &[ChatMessage],
    settings: &ChatSettings,
) -> Result<String, ErrorPayload> {
    // For now, use a simple fallback
    // In a complete implementation, this would parse the template
    // and apply it correctly for models like Qwen, Mistral, etc.
    Ok(build_prompt_fallback(messages, settings))
}

fn build_prompt_fallback(
    messages: &[ChatMessage],
    settings: &ChatSettings,
) -> String {
    let mut prompt = String::new();

    // Add system prompt if present
    if !settings.system_prompt.is_empty() {
        prompt.push_str("System: ");
        prompt.push_str(&settings.system_prompt);
        prompt.push_str("\n\n");
    }

    // Add messages
    for message in messages {
        prompt.push_str(&format!("{}: {}\n", message.role, message.content));
    }

    prompt.push_str("assistant: ");
    prompt
}
