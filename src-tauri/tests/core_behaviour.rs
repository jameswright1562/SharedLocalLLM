use std::{fs, path::Path};

use shared_local_llm::{
    capacity::{recommend_topology, CapacityOutcome, CapacityRequest},
    models::discover_gguf_models,
    network::{classify_network, NetworkClass, NetworkMetrics},
    pairing::PairingManager,
};

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

    let records = discover_gguf_models(&[root.path().to_path_buf()]).unwrap();

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].shard_paths.len(), 2);
    assert!(records[0].projector.is_some());
    assert!(records[0].vision_capable);
}

#[test]
fn deduplicates_nested_custom_roots() {
    let root = tempfile::tempdir().unwrap();
    touch(&root.path().join("models/tiny.gguf"));

    let records =
        discover_gguf_models(&[root.path().to_path_buf(), root.path().join("models")]).unwrap();

    assert_eq!(records.len(), 1);
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
fn classifies_network_using_sustained_bandwidth_and_p95_latency() {
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
