# Phase 3 checkpoints

Phase 3A needs the verified English-only XLM-R Base QA checkpoint from Phase 1. Phase 3B uses the checkpoint selected by multilingual validation macro-F1.

| Checkpoint | Selected | `model.safetensors` SHA-256 | Public link |
| --- | --- | --- | --- |
| 3A English fallback, seed 42 | `checkpoint-500` (epoch 5) | `3d350d925798c5f67c27ee4f07a46f1149549b583cfcd5ac885fcba74ea1deea` | Not uploaded |
| 3B multilingual, seed 42 | `checkpoint-1792` (epoch 4) | `34441b90ad42829eb6779c3ed82690da278c8bd39a5d20d2b8273aa5a3017915` | Not uploaded |

Both were trained on 25 November 2026 and are stored on the training server under `outputs/phase_03_multilingual_finetuning/`, in `english-fallback-cuda-span-checked-42/best_checkpoint` and `training-3b-cuda-span-checked-42/best_checkpoint`. Each run manifest in `../results/phase_3a/` and `../results/phase_3b/` holds the full provenance record, including every checkpoint file hash.

Each local model directory should contain the full QA weights, tokenizer and model config. Supply `phase3_checkpoint.json` alongside the files or pass an external JSON file using `--provenance`. The fallback and multilingual training scripts write this manifest automatically.

Required provenance fields are `base_model`, `base_revision`, `training_languages`, `selection_languages`, `seed`, `training_configuration`, `computational_statistics`, `training_data`, and `checkpoint_file_sha256`. The last field maps all other checkpoint-relative file paths to SHA-256 hashes. Include tokenization settings when available.

For a checkpoint trained on the official English data, set `training_data.kind` to `phase1_official_english` and record the pinned English `revision` and `split_sha256` for train and validation. For Phase 2 training inputs, use `phase2_pinned` with the bundle manifest hash, variant and train and validation split hashes. The evaluation runner validates these fields before loading weights.

Use a separately approved model store for binary artifacts and record the immutable checkpoint link and hashes here after upload.
