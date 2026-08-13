use tauri::State;

use crate::{
    state::{directory_for, AppState},
    types::{ErrorPayload, ModelDirectory, ModelRecord},
};

#[tauri::command]
pub fn discover_models(state: State<'_, AppState>) -> Result<Vec<ModelRecord>, ErrorPayload> {
    state.refresh_models_shared()
}

#[tauri::command]
pub fn add_model_directory(
    state: State<'_, AppState>,
) -> Result<Option<ModelDirectory>, ErrorPayload> {
    let selected = std::env::var_os("SHARED_LOCAL_LLM_MODEL_DIRECTORY")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            rfd::FileDialog::new()
                .set_title("Choose a GGUF model directory")
                .pick_folder()
        });
    let Some(path) = selected else {
        return Ok(None);
    };
    if !path.is_dir() {
        return Err(ErrorPayload::new(
            "model_directory_invalid",
            "The selected model directory does not exist.",
            None,
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| ErrorPayload::new("model_directory_invalid", error.to_string(), None))?;
    let local_id = state.lock()?.local.id.clone();
    let directory = directory_for(&canonical, "custom", &local_id);
    {
        let mut inner = state.lock()?;
        if !inner.directories.iter().any(|item| item.id == directory.id) {
            inner.directories.push(directory.clone());
        }
    }
    state.persist()?;
    state.refresh_models_shared()?;
    Ok(Some(directory))
}

#[tauri::command]
pub fn remove_model_directory(id: String, state: State<'_, AppState>) -> Result<(), ErrorPayload> {
    state
        .lock()?
        .directories
        .retain(|directory| directory.id != id || directory.source != "custom");
    state.persist()?;
    state.refresh_models_shared()?;
    Ok(())
}
