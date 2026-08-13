use std::{fs, path::Path};

use shared_local_llm::{
    capacity::{
        estimate_layer_split, recommend_topology, CapacityOutcome, CapacityRequest,
        DeviceLayerRequest, SplitEstimateRequest,
    },
    models::discover_gguf_models,
    network::{classify_network, NetworkClass, NetworkMetrics},
    pairing::PairingManager,
};

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_string(bytes, key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_string(bytes, value);
}

fn push_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_string(bytes, key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_model_metadata(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&6u64.to_le_bytes());
    push_string_metadata(&mut bytes, "general.architecture", "llama");
    push_u32_metadata(&mut bytes, "llama.block_count", 40);
    push_u32_metadata(&mut bytes, "llama.context_length", 32_768);
    push_u32_metadata(&mut bytes, "llama.embedding_length", 4_096);
    push_u32_metadata(&mut bytes, "llama.attention.head_count", 32);
    push_u32_metadata(&mut bytes, "llama.attention.head_count_kv", 8);
    fs::write(path, bytes).unwrap();
}

fn touch(path: &Path) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, b"GGUF").unwrap();
}

#[test]
fn groups_split_shards_and_associates_projector() {
    let root = tempfile::tempdir().unwrap();
    touch(&root.path().join("vision/model-00001-of-00002.gguf"));
    touch(&root.path().join("vision/model-00002-of-00002.gguf"));
    touch(&root.path().join("vision/mmproj-model-f16.gguf"));

    let records = discover_gguf_models(&[root.path().to_path_buf()], "test-node").unwrap();

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].locations[0].node_id, "test-node");
    assert_eq!(records[0].shard_paths.len(), 2);
    assert!(records[0].projector.is_some());
    assert!(records[0].vision_capable);
}

#[test]
fn deduplicates_nested_custom_roots() {
    let root = tempfile::tempdir().unwrap();
    touch(&root.path().join("models/tiny.gguf"));

    let records = discover_gguf_models(
        &[root.path().to_path_buf(), root.path().join("models")],
        "test-node",
    )
    .unwrap();

    assert_eq!(records.len(), 1);
}

#[test]
fn reads_layer_and_attention_metadata_for_split_planning() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    write_model_metadata(&model);

    let records = discover_gguf_models(&[root.path().to_path_buf()], "test-node").unwrap();

    assert_eq!(records[0].layer_count, Some(40));
    assert_eq!(records[0].context_length, 32_768);
    assert_eq!(records[0].embedding_length, Some(4_096));
    assert_eq!(records[0].attention_head_count, Some(32));
    assert_eq!(records[0].attention_head_count_kv, Some(8));
}

#[test]
fn estimates_vram_for_each_selected_device_and_rejects_too_many_layers() {
    let request = SplitEstimateRequest {
        model_mib: 20_000,
        total_layers: 40,
        context_size: 8_192,
        embedding_length: Some(4_096),
        attention_head_count: Some(32),
        attention_head_count_kv: Some(8),
        devices: vec![
            DeviceLayerRequest::new("computer-a", 24, 16_000),
            DeviceLayerRequest::new("computer-b", 16, 12_000),
        ],
    };

    let estimate = estimate_layer_split(request.clone()).unwrap();

    assert_eq!(estimate.gpu_layers, 40);
    assert_eq!(estimate.cpu_layers, 0);
    assert_eq!(estimate.devices[0].estimated_vram_mib, 13_280);
    assert_eq!(estimate.devices[1].estimated_vram_mib, 9_024);
    assert!(estimate.devices.iter().all(|device| device.fits));

    let invalid = SplitEstimateRequest {
        devices: vec![
            DeviceLayerRequest::new("computer-a", 30, 16_000),
            DeviceLayerRequest::new("computer-b", 20, 12_000),
        ],
        ..request
    };
    assert!(estimate_layer_split(invalid).is_err());
}

#[test]
fn recommends_single_distributed_spill_and_rejection_without_gpu_models() {
    let single = recommend_topology(CapacityRequest::new(8_000, vec![12_000, 8_000], 32_000));
    assert!(matches!(single.outcome, CapacityOutcome::SingleNode { .. }));

    let distributed =
        recommend_topology(CapacityRequest::new(16_000, vec![10_000, 10_000], 32_000));
    assert!(matches!(
        distributed.outcome,
        CapacityOutcome::Distributed { .. }
    ));

    let spill = recommend_topology(CapacityRequest::new(22_000, vec![10_000, 10_000], 32_000));
    assert!(matches!(spill.outcome, CapacityOutcome::RamSpill { .. }));

    let rejected = recommend_topology(CapacityRequest::new(80_000, vec![10_000, 10_000], 16_000));
    assert!(matches!(rejected.outcome, CapacityOutcome::Insufficient));
    assert_eq!(rejected.context_size, 8192);
}

#[test]
fn classifies_network_using_p95_latency() {
    assert_eq!(
        classify_network(NetworkMetrics::new(900.0, 2.5, 0.4, 0.0)),
        NetworkClass::Good
    );
    assert_eq!(
        classify_network(NetworkMetrics::new(400.0, 8.0, 1.0, 0.1)),
        NetworkClass::Usable
    );
    assert_eq!(
        classify_network(NetworkMetrics::new(199.0, 2.0, 0.0, 0.0)),
        NetworkClass::Good
    );
    assert_eq!(
        classify_network(NetworkMetrics::new(900.0, 12.0, 0.0, 0.0)),
        NetworkClass::Poor
    );
}

#[test]
fn pairing_codes_are_six_digit_single_use_and_expire() {
    let mut manager = PairingManager::default();
    let code = manager.generate_at(1_000);
    assert_eq!(code.len(), 6);
    assert!(code.chars().all(|c| c.is_ascii_digit()));
    assert!(manager.consume_at(&code, 1_100).is_ok());
    assert!(manager.consume_at(&code, 1_101).is_err());

    let expired = manager.generate_at(2_000);
    assert!(manager.consume_at(&expired, 2_301).is_err());
}
