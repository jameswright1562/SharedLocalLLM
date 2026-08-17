use std::ffi::CString;

use llama_cpp_sys_4 as sys;

use crate::types::ErrorPayload;

pub fn register_remote_server(
    endpoint: &str,
) -> Result<(), ErrorPayload> {
    let endpoint = CString::new(endpoint)
        .map_err(|error| {
            ErrorPayload::new(
                "rpc_endpoint_invalid",
                error.to_string(),
                None,
            )
        })?;

    unsafe {
        let registration =
            sys::ggml_backend_rpc_add_server(
                endpoint.as_ptr(),
            );

        if registration.is_null() {
            return Err(ErrorPayload::new(
                "rpc_registration_failed",
                "llama.cpp could not register the remote RPC server.",
                None,
            ));
        }

        sys::ggml_backend_register(registration);
    }

    Ok(())
}
