use std::{fs, path::Path};

use crate::types::ErrorPayload;

pub fn store(path: &Path, plaintext: &[u8]) -> Result<(), ErrorPayload> {
    let protected = protect(plaintext)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io_error)?;
    }
    fs::write(path, protected).map_err(io_error)
}

pub fn load(path: &Path) -> Result<Option<Vec<u8>>, ErrorPayload> {
    if !path.exists() {
        return Ok(None);
    }
    let protected = fs::read(path).map_err(io_error)?;
    unprotect(&protected).map(Some)
}

fn io_error(error: std::io::Error) -> ErrorPayload {
    ErrorPayload::new("secret_store_io", error.to_string(), None)
}

#[cfg(windows)]
fn protect(data: &[u8]) -> Result<Vec<u8>, ErrorPayload> {
    use windows::{
        core::w,
        Win32::{
            Foundation::{LocalFree, HLOCAL},
            Security::Cryptography::{
                CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
            },
        },
    };
    let input = CRYPT_INTEGER_BLOB {
        cbData: data.len() as u32,
        pbData: data.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    unsafe {
        CryptProtectData(
            &input,
            w!("SharedLocalLLM"),
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
        .map_err(|e| ErrorPayload::new("dpapi_protect_failed", e.to_string(), None))?;
        let bytes = std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec();
        let _ = LocalFree(Some(HLOCAL(output.pbData.cast())));
        Ok(bytes)
    }
}

#[cfg(windows)]
fn unprotect(data: &[u8]) -> Result<Vec<u8>, ErrorPayload> {
    use windows::Win32::{
        Foundation::{LocalFree, HLOCAL},
        Security::Cryptography::{
            CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
        },
    };
    let input = CRYPT_INTEGER_BLOB {
        cbData: data.len() as u32,
        pbData: data.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    unsafe {
        CryptUnprotectData(
            &input,
            None,
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
        .map_err(|e| {
            ErrorPayload::new(
                "dpapi_unprotect_failed",
                e.to_string(),
                Some("Re-pair this device and regenerate its local API key.".into()),
            )
        })?;
        let bytes = std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec();
        let _ = LocalFree(Some(HLOCAL(output.pbData.cast())));
        Ok(bytes)
    }
}

#[cfg(not(windows))]
fn protect(data: &[u8]) -> Result<Vec<u8>, ErrorPayload> {
    // Development-only fallback: never claimed as secure and not used in supported Windows builds.
    Ok(data.iter().map(|byte| byte ^ 0xa7).collect())
}

#[cfg(not(windows))]
fn unprotect(data: &[u8]) -> Result<Vec<u8>, ErrorPayload> {
    protect(data)
}
