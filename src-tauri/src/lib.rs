pub mod autostart;
mod backend;
pub mod firewall;
pub mod types;

use backend::BackendProcess;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};

pub fn run() {
    crate::firewall::ensure_firewall_elevation();
    tauri::Builder::default()
        .manage(BackendProcess::default())
        .setup(|app| {
            app.state::<BackendProcess>().start(app.handle())?;
            if let Ok(executable) = std::env::current_exe() {
                tauri::async_runtime::spawn(async move {
                    let _ = crate::firewall::ensure_peer_firewall_rules(
                        &executable,
                        49_158,
                        49_157,
                    )
                    .await;
                });
            }
            install_tray(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.state::<BackendProcess>().is_cluster_running() {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            backend::backend_request,
            backend::pick_model_directory,
            backend::open_network_settings,
            backend::open_logs_folder,
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
