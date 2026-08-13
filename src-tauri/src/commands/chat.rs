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
    let (api_key, api_port) = {
        let inner = state.lock()?;
        if inner.cluster.status != "running" {
            return Err(ErrorPayload::new(
                "cluster_not_running",
                "Start a model before sending a chat message.",
                Some("Choose a model and start the cluster.".into()),
            ));
        }
        (inner.api_key.clone(), inner.api_port)
    };
    let mut wire_messages: Vec<Value> = messages
        .into_iter()
        .map(|message| json!({"role": message.role, "content": message.content}))
        .collect();
    if !settings.system_prompt.trim().is_empty() {
        wire_messages.insert(
            0,
            json!({"role":"system", "content": settings.system_prompt}),
        );
    }
    attach_images(&mut wire_messages, images)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(proxy_error)?;
    let request = client
        .post(format!("http://127.0.0.1:{api_port}/v1/chat/completions"))
        .bearer_auth(api_key)
        .json(&json!({
            "model":"active", "messages":wire_messages,
            "temperature":settings.temperature, "max_tokens":settings.max_tokens, "stream":false
        }))
        .send();
    let (cancel, cancelled) = tokio::sync::oneshot::channel();
    *state.chat_cancel.lock().map_err(|_| {
        ErrorPayload::new(
            "chat_state",
            "Chat cancellation state is unavailable.",
            None,
        )
    })? = Some(cancel);
    let response = tokio::select! {
        response = request => response.map_err(proxy_error)?,
        _ = cancelled => {
            let _ = app.emit("chat-cancelled", json!({"cancelled": true}));
            return Err(ErrorPayload::new("generation_cancelled", "Generation was cancelled.", None));
        }
    }
    .error_for_status()
    .map_err(proxy_error)?;
    let _ = state.chat_cancel.lock().map(|mut slot| slot.take());
    let value: Value = response.json().await.map_err(proxy_error)?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ErrorPayload::new(
                "inference_response_invalid",
                "llama-server returned no assistant content.",
                None,
            )
        })?
        .to_owned();
    let _ = app.emit("chat-token", json!({"content":content}));
    Ok(ChatResponse { content })
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
