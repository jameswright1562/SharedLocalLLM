pub mod capacity;
pub mod commands;
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

pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
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
            commands::cluster::start_cluster,
            commands::cluster::stop_cluster,
            commands::cluster::run_inference_benchmark,
            commands::cluster::cancel_inference_benchmark,
            commands::chat::send_chat_message,
            commands::chat::cancel_generation,
            commands::api::get_api_config,
            commands::api::regenerate_api_key,
            commands::app::open_logs_folder
        ])
        .run(tauri::generate_context!())
        .expect("error while running SharedLocalLLM");
}
