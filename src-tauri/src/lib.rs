pub mod autostart;
pub mod capacity;
pub mod commands;
pub mod gguf;
pub mod hardware;
pub mod models;
pub mod network;
pub mod pairing;
pub mod peer;
pub mod runtime;
pub mod secrets;
pub mod state;
pub mod types;

use state::AppState;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};

pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                commands::pairing::lifecycle::start_persistent_peer_service(handle).await;
            });
            install_tray(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let running = window
                    .state::<AppState>()
                    .lock()
                    .map(|inner| inner.cluster.status == "running")
                    .unwrap_or(false);
                if running {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::app::get_app_snapshot,
            commands::app::install_runtime,
            commands::app::refresh_hardware,
            commands::app::complete_setup,
            commands::app::update_settings,
            commands::models::discover_models,
            commands::models::add_model_directory,
            commands::models::remove_model_directory,
            commands::network::run_network_test,
            commands::pairing::generate_pairing_code,
            commands::pairing::pair_with_peer,
            commands::pairing::reset::reset_pairing,
            commands::cluster::split::estimate_model_split,
            commands::cluster::start_cluster,
            commands::cluster::stop_cluster,
            commands::cluster::benchmark::run_inference_benchmark,
            commands::cluster::benchmark::cancel_inference_benchmark,
            commands::chat::send_chat_message,
            commands::chat::cancel_generation,
            commands::api::get_api_config,
            commands::api::regenerate_api_key,
            commands::app::open_network_settings,
            commands::app::open_logs_folder
        ])
        .run(tauri::generate_context!())
        .expect("error while running SharedLocalLLM");
}

fn install_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let show = MenuItem::with_id(app, "show", "Show SharedLocalLLM", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;
    TrayIconBuilder::new()
        .icon(
            app.default_window_icon()
                .cloned()
                .ok_or("missing tray icon")?,
        )
        .menu(&menu)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .build(app)?;
    Ok(())
}
