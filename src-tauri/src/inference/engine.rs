use std::num::NonZeroU32;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};

use llama_cpp_4::{LlamaBackend, LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams};

use crate::{
    inference::{generation, rpc},
    types::{ChatMessage, ChatSettings, ErrorPayload, GpuLayerAllocation},
};

pub enum InferenceCommand {
    Load {
        model_path: String,
        context_size: u32,
        allocations: Vec<GpuLayerAllocation>,
        rpc_endpoint: Option<String>,
        response: tokio::sync::oneshot::Sender<Result<(), ErrorPayload>>,
    },

    Generate {
        messages: Vec<ChatMessage>,
        settings: ChatSettings,
        response: tokio::sync::oneshot::Sender<Result<String, ErrorPayload>>,
    },

    Unload,
    Shutdown,
}

pub struct InferenceEngine {
    sender: Sender<InferenceCommand>,
}

impl InferenceEngine {
    pub fn start() -> Result<Self, ErrorPayload> {
        let (sender, receiver) = mpsc::channel::<InferenceCommand>();

        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), ErrorPayload>>(1);

        std::thread::Builder::new()
            .name("llama-inference".into())
            .spawn(move || {
                inference_thread(receiver, ready_tx);
            })
            .map_err(|error| {
                ErrorPayload::new("inference_thread_failed", error.to_string(), None)
            })?;

        ready_rx.recv().map_err(|error| {
            ErrorPayload::new("inference_thread_failed", error.to_string(), None)
        })??;

        Ok(Self { sender })
    }

    pub async fn load(
        &self,
        model_path: String,
        context_size: u32,
        allocations: Vec<GpuLayerAllocation>,
        rpc_endpoint: Option<String>,
    ) -> Result<(), ErrorPayload> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sender
            .send(InferenceCommand::Load {
                model_path,
                context_size,
                allocations,
                rpc_endpoint,
                response: tx,
            })
            .map_err(|_| {
                ErrorPayload::new(
                    "inference_thread_stopped",
                    "The inference thread is no longer running.",
                    None,
                )
            })?;

        rx.await.map_err(|_| {
            ErrorPayload::new(
                "inference_response_lost",
                "The inference thread stopped while loading the model.",
                None,
            )
        })?
    }

    pub async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        settings: ChatSettings,
    ) -> Result<String, ErrorPayload> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.sender
            .send(InferenceCommand::Generate {
                messages,
                settings,
                response: tx,
            })
            .map_err(|_| {
                ErrorPayload::new(
                    "inference_thread_stopped",
                    "The inference thread is no longer running.",
                    None,
                )
            })?;

        rx.await.map_err(|_| {
            ErrorPayload::new(
                "inference_response_lost",
                "The inference thread stopped while generating.",
                None,
            )
        })?
    }

    pub async fn unload(&self) -> Result<(), ErrorPayload> {
        self.sender.send(InferenceCommand::Unload).map_err(|_| {
            ErrorPayload::new(
                "inference_thread_stopped",
                "The inference thread is no longer running.",
                None,
            )
        })?;

        // Give the thread a moment to unload
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<(), ErrorPayload> {
        self.sender.send(InferenceCommand::Shutdown).map_err(|_| {
            ErrorPayload::new(
                "inference_thread_stopped",
                "The inference thread is no longer running.",
                None,
            )
        })?;

        Ok(())
    }
}

fn inference_thread(
    receiver: Receiver<InferenceCommand>,
    ready: SyncSender<Result<(), ErrorPayload>>,
) {
    let backend = match LlamaBackend::init() {
        Ok(backend) => {
            let _ = ready.send(Ok(()));
            backend
        }

        Err(error) => {
            let _ = ready.send(Err(ErrorPayload::new(
                "llama_backend_init_failed",
                error.to_string(),
                None,
            )));
            return;
        }
    };

    while let Ok(command) = receiver.recv() {
        match command {
            InferenceCommand::Load {
                model_path,
                context_size,
                allocations,
                rpc_endpoint,
                response,
            } => {
                let result = load_model(
                    &backend,
                    &receiver,
                    model_path,
                    context_size,
                    allocations,
                    rpc_endpoint,
                );

                let _ = response.send(result);
            }

            InferenceCommand::Shutdown => {
                break;
            }

            InferenceCommand::Generate { response, .. } => {
                let _ = response.send(Err(ErrorPayload::new(
                    "model_not_loaded",
                    "No model is currently loaded.",
                    None,
                )));
            }

            InferenceCommand::Unload => {
                // Nothing loaded yet.
            }
        }
    }
}

fn load_model(
    backend: &LlamaBackend,
    receiver: &Receiver<InferenceCommand>,
    model_path: String,
    context_size: u32,
    allocations: Vec<GpuLayerAllocation>,
    rpc_endpoint: Option<String>,
) -> Result<(), ErrorPayload> {
    // 1. Register the remote RPC device BEFORE model loading.
    if let Some(endpoint) = rpc_endpoint.as_deref() {
        rpc::register_remote_server(endpoint)?;
    }

    // 2. Work out how many layers are being GPU-offloaded.
    let gpu_layers: u32 = allocations.iter().map(|allocation| allocation.layers).sum();

    // For now use all GPU layers if the UI didn't specify anything.
    let gpu_layers = if gpu_layers == 0 {
        u32::MAX
    } else {
        gpu_layers
    };

    // 3. Create llama.cpp model parameters.
    let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);

    // 4. Load model
    let model =
        LlamaModel::load_from_file(backend, &model_path, &model_params).map_err(|error| {
            ErrorPayload::new(
                "model_load_failed",
                error.to_string(),
                Some("Check the GGUF file and available GPU memory.".into()),
            )
        })?;

    // 5. Create the context.
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(context_size));

    let mut context = model.new_context(backend, ctx_params).map_err(|error| {
        ErrorPayload::new(
            "context_create_failed",
            error.to_string(),
            Some("Try reducing the context size.".into()),
        )
    })?;

    // 6. The model + context now stay alive in this function.
    run_loaded_model(receiver, &model, &mut context, backend)
}

fn run_loaded_model(
    receiver: &Receiver<InferenceCommand>,
    model: &LlamaModel,
    context: &mut LlamaContext<'_>,
    backend: &LlamaBackend,
) -> Result<(), ErrorPayload> {
    loop {
        let command = receiver.recv().map_err(|_| {
            ErrorPayload::new(
                "inference_thread_channel_closed",
                "The inference command channel was closed.",
                None,
            )
        })?;

        match command {
            InferenceCommand::Generate {
                messages,
                settings,
                response,
            } => {
                let result = generation::generate(model, context, backend, messages, settings);

                let _ = response.send(result);
            }

            InferenceCommand::Unload => {
                return Ok(());
            }

            InferenceCommand::Shutdown => {
                return Ok(());
            }

            InferenceCommand::Load { response, .. } => {
                let _ = response.send(Err(ErrorPayload::new(
                    "model_already_loaded",
                    "Unload the current model before loading another one.",
                    None,
                )));
            }
        }
    }
}
