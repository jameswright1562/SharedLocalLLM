use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::types::ErrorPayload;

const CODE_TTL_SECONDS: u64 = 300;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerRecord {
    pub id: String,
    pub name: String,
    pub address: Option<String>,
    pub trusted_at: u64,
    #[serde(default)]
    pub capabilities: Option<crate::types::NodeCapabilities>,
}

#[derive(Default)]
pub struct PairingManager {
    pending: Option<(String, u64)>,
}

impl PairingManager {
    pub fn generate_at(&mut self, now: u64) -> String {
        let code = format!("{:06}", rand::thread_rng().gen_range(0..1_000_000));
        self.pending = Some((code.clone(), now + CODE_TTL_SECONDS));
        code
    }

    pub fn consume_at(&mut self, supplied: &str, now: u64) -> Result<(), ErrorPayload> {
        let normalized: String = supplied.chars().filter(|c| c.is_ascii_digit()).collect();
        let Some((expected, expires)) = self.pending.take() else {
            return Err(ErrorPayload::new(
                "pairing_code_missing",
                "Generate a new pairing code first.",
                None,
            ));
        };
        if now > expires {
            return Err(ErrorPayload::new(
                "pairing_code_expired",
                "The pairing code expired.",
                Some("Generate a fresh code on the other computer.".into()),
            ));
        }
        if normalized != expected {
            return Err(ErrorPayload::new(
                "pairing_code_invalid",
                "The pairing code does not match.",
                None,
            ));
        }
        Ok(())
    }

    pub fn generate(&mut self) -> String {
        self.generate_at(now())
    }
    pub fn consume(&mut self, code: &str) -> Result<(), ErrorPayload> {
        self.consume_at(code, now())
    }
}

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
