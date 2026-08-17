use std::{
    ffi::CString,
    ptr,
    sync::OnceLock,
};

use llama_cpp_sys_4 as sys;

use crate::types::ErrorPayload;

static RPC_WORKER_STARTED: OnceLock<()> = OnceLock::new();

pub fn start_rpc_worker() -> Result<(), ErrorPayload> {
    // Only ever start one RPC worker per SharedLocalLLM process.
    if RPC_WORKER_STARTED.get().is_some() {
        return Ok(());
    }

    std::thread::Builder::new()
        .name("llama-rpc-worker".into())
        .spawn(|| {
            if let Err(error) = run_rpc_worker() {
                eprintln!("RPC worker failed: {error}");
            }
        })
        .map_err(|error| {
            ErrorPayload::new(
                "rpc_worker_thread_failed",
                error.to_string(),
                None,
            )
        })?;

    let _ = RPC_WORKER_STARTED.set(());

    Ok(())
}

fn run_rpc_worker() -> Result<(), ErrorPayload> {
    unsafe {
        let device_count =
            sys::ggml_backend_dev_count();

        let mut devices = Vec::new();

        for index in 0..device_count {
            let device =
                sys::ggml_backend_dev_get(index);

            if device.is_null() {
                continue;
            }

            let device_type =
                sys::ggml_backend_dev_type(device);

            if device_type ==
                sys::GGML_BACKEND_DEVICE_TYPE_GPU
            {
                devices.push(device);
            }
        }

        if devices.is_empty() {
            return Err(ErrorPayload::new(
                "rpc_worker_no_gpu",
                "No GPU backend was available for the RPC worker.",
                None,
            ));
        }

        let endpoint =
            CString::new("127.0.0.1:50052")
                .expect("static RPC endpoint");

        sys::ggml_backend_rpc_start_server(
            endpoint.as_ptr(),
            ptr::null(),
            4,
            devices.len(),
            devices.as_mut_ptr(),
        );
    }

    Ok(())
}
