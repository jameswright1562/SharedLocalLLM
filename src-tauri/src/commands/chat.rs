use serde_json::{json, Value};
use tauri::{Emitter, State};

use crate::{
    state::AppState,
    types::{ChatMessage, ChatResponse, ChatSettings, ErrorPayload},
};

#[tauri::command]
pub async fn send_chat_message(
    app: tauri::AppHandle,
    messages: Vec<ChatMessage>,
    settings: ChatSettings,
    images: Vec<String>,
    state: State<'_, AppState>,
) -> Result<ChatResponse, ErrorPayload> {
    let (api_key, api_port, local_running, peer_running) = {
        let inner = state.lock()?;
        let peer_running = inner.peers.iter().any(|peer| {
            peer.capabilities.as_ref().is_some_and(|capabilities| {
                capabilities
                    .cluster_status
                    .as_deref()
                    .is_some_and(|status| status == "running")
            })
        });
        (
            inner.api_key.clone(),
            inner.api_port,
            inner.cluster.status == "running",
            peer_running,
        )
    };
    let messages = json!(messages
        .into_iter()
        .filter(|message| message.role != "error")
        .map(|message| json!({"role": message.role, "content": message.content}))
        .collect::<Vec<_>>());
    let settings_value = json!({
        "systemPrompt": settings.system_prompt,
        "temperature": settings.temperature,
        "maxTokens": settings.max_tokens
    });
    if !local_running && peer_running {
        let client = state.peer_client().await?;
        let content = client.proxy_chat(messages, settings_value, images).await?;
        let _ = app.emit("chat-token", json!({"content": content}));
        return Ok(ChatResponse { content });
    }
    if !local_running {
        return Err(ErrorPayload::new(
            "cluster_not_running",
            "Start a model before sending a chat message.",
            Some("Choose a model and start the cluster.".into()),
        ));
    }
    let (cancel, cancelled) = tokio::sync::oneshot::channel();
    *state.chat_cancel.lock().map_err(|_| {
        ErrorPayload::new(
            "chat_state",
            "Chat cancellation state is unavailable.",
            None,
        )
    })? = Some(cancel);
    let result = tokio::select! {
        result = stream_local(app.clone(), api_port, &api_key, messages, settings, images) => result,
        _ = cancelled => {
            let _ = app.emit("chat-cancelled", json!({"cancelled": true}));
            Err(ErrorPayload::new("generation_cancelled", "Generation was cancelled.", None))
        }
    };
    let _ = state.chat_cancel.lock().map(|mut slot| slot.take());
    result
}

pub(crate) async fn complete_local(
    api_port: u16,
    api_key: &str,
    messages: Value,
    settings: Value,
    images: Vec<String>,
) -> Result<String, ErrorPayload> {
    let chat_settings = ChatSettings {
        system_prompt: settings
            .get("systemPrompt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .into(),
        temperature: settings
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(0.7),
        max_tokens: settings
            .get("maxTokens")
            .and_then(Value::as_u64)
            .unwrap_or(1024) as u32,
    };
    let body = request_body(messages, &chat_settings, images, false)?;
    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(proxy_error)?
        .post(format!("http://127.0.0.1:{api_port}/v1/chat/completions"))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(proxy_error)?
        .error_for_status()
        .map_err(proxy_error)?;
    let value: Value = response.json().await.map_err(proxy_error)?;
    value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            ErrorPayload::new(
                "inference_response_invalid",
                "llama-server returned no assistant content.",
                None,
            )
        })
}

async fn stream_local(
    app: tauri::AppHandle,
    api_port: u16,
    api_key: &str,
    messages: Value,
    settings: ChatSettings,
    images: Vec<String>,
) -> Result<ChatResponse, ErrorPayload> {
    use futures_util::StreamExt;
    let body = request_body(messages, &settings, images, true)?;
    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(proxy_error)?
        .post(format!("http://127.0.0.1:{api_port}/v1/chat/completions"))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(proxy_error)?
        .error_for_status()
        .map_err(proxy_error)?;
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        buffer.push_str(&String::from_utf8_lossy(&chunk.map_err(proxy_error)?));
        while let Some(index) = buffer.find("\n\n") {
            let frame = buffer[..index].to_owned();
            buffer = buffer[index + 2..].to_owned();
            if let Some(delta) = parse_sse_delta(&frame) {
                content.push_str(&delta);
                let _ = app.emit("chat-token", json!({"content": delta}));
            }
        }
    }
    if content.is_empty() {
        return Err(ErrorPayload::new(
            "inference_response_invalid",
            "llama-server returned no assistant content.",
            None,
        ));
    }
    Ok(ChatResponse { content })
}

fn request_body(
    mut messages: Value,
    settings: &ChatSettings,
    images: Vec<String>,
    stream: bool,
) -> Result<Value, ErrorPayload> {
    let mut wire = messages.as_array().cloned().unwrap_or_default();
    if !settings.system_prompt.trim().is_empty() {
        wire.insert(
            0,
            json!({"role":"system", "content": settings.system_prompt}),
        );
    }
    attach_images(&mut wire, images)?;
    messages = Value::Array(wire);
    Ok(json!({
        "model":"active",
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": stream
    }))
}

fn parse_sse_delta(frame: &str) -> Option<String> {
    let data = frame
        .lines()
        .find_map(|line| line.strip_prefix("data:"))?
        .trim();
    if data == "[DONE]" {
        return None;
    }
    let value: Value = serde_json::from_str(data).ok()?;
    value
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

#[tauri::command]
pub fn cancel_generation(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), ErrorPayload> {
    if let Some(cancel) = state
        .chat_cancel
        .lock()
        .map_err(|_| {
            ErrorPayload::new(
                "chat_state",
                "Chat cancellation state is unavailable.",
                None,
            )
        })?
        .take()
    {
        let _ = cancel.send(());
        let _ = app.emit("chat-cancelled", json!({"cancelled": true}));
    }
    Ok(())
}

fn attach_images(messages: &mut [Value], images: Vec<String>) -> Result<(), ErrorPayload> {
    if images.is_empty() {
        return Ok(());
    }
    if let Some(last) = messages.last_mut() {
        let text = last
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let mut content = vec![json!({"type":"text", "text": text})];
        for image in images {
            let valid_type = [
                "data:image/png;base64,",
                "data:image/jpeg;base64,",
                "data:image/webp;base64,",
            ]
            .iter()
            .any(|prefix| image.starts_with(prefix));
            if !valid_type {
                return Err(ErrorPayload::new(
                    "image_source_rejected",
                    "Images must be picker-loaded PNG, JPEG, or WebP data URLs.",
                    None,
                ));
            }
            if image.len() > 14 * 1024 * 1024 {
                return Err(ErrorPayload::new(
                    "image_too_large",
                    "An image exceeds the 10 MiB attachment limit.",
                    None,
                ));
            }
            content.push(json!({"type":"image_url", "image_url":{"url": image}}));
        }
        last["content"] = Value::Array(content);
    }
    Ok(())
}

fn proxy_error(error: reqwest::Error) -> ErrorPayload {
    ErrorPayload::new(
        "inference_unavailable",
        error.to_string(),
        Some("Confirm the cluster is running and inspect the runtime logs.".into()),
    )
}
