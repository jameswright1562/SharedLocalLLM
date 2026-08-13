use std::{
    fs::{self, OpenOptions},
    io::Write,
};

use regex::Regex;

use super::persistence::logs_root;

const MAX_LIVE_LINES: usize = 500;

pub(super) fn append(logs: &mut Vec<String>, level: &str, event: &str, detail: &str) {
    let timestamp = crate::pairing::now();
    let detail = redact(detail);
    let line = if detail.is_empty() {
        format!("{timestamp} {level} {event}")
    } else {
        format!("{timestamp} {level} {event}: {detail}")
    };
    logs.push(line.clone());
    if logs.len() > MAX_LIVE_LINES {
        logs.drain(..logs.len() - MAX_LIVE_LINES);
    }
    let root = logs_root();
    if fs::create_dir_all(&root).is_ok() {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("shared-local-llm.log"))
        {
            let _ = writeln!(file, "{line}");
        }
    }
}

pub(super) fn redact(value: &str) -> String {
    let secrets = Regex::new(r"(?i)(sk-local-[a-z0-9-]+|\b\d{6}\b)").expect("valid regex");
    let paths = Regex::new(r#"(?i)([a-z]:\\|\\\\)[^\s\"']+"#).expect("valid regex");
    paths
        .replace_all(
            &secrets.replace_all(value, "[REDACTED_SECRET]"),
            "[REDACTED_PATH]",
        )
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::redact;

    #[test]
    fn diagnostics_remove_keys_pairing_codes_and_personal_paths() {
        let redacted =
            redact(r#"key sk-local-private code 482916 model C:\Users\James\Models\private.gguf"#);
        assert!(!redacted.contains("sk-local-private"));
        assert!(!redacted.contains("482916"));
        assert!(!redacted.contains("James"));
        assert!(redacted.contains("[REDACTED_SECRET]"));
        assert!(redacted.contains("[REDACTED_PATH]"));
    }
}
