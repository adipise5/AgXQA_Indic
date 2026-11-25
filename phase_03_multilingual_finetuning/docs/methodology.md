# Phase 3 methodology

## Input and experiment controls

Use the same reviewed Phase 2 version and test hash for both experiments. Train and select 3A on English alone. Reuse Dhiraj’s Phase 1 English-only XLM-R QA checkpoint if it matches the pinned source and base revision. For 3B, initialize from XLM-R Base and fine-tune all parameters jointly. Human review and five English test cases currently gate the reviewed dataset. [Phase 2 adjudication and finalizer](../../phase_02_multilingual_benchmark/README.md)

The runners default to `--variant final`. `--variant span_checked` is available for explicitly provisional work. `--variant document_disjoint` uses the separately pinned subset and requires an English-only checkpoint trained on its matched English rows. The set identity, test hash, per-seed provenance and checkpoint hashes are written with the prediction artifacts.

## Commands

Run from the AgXQA-Indic checkout. The tokenizer-only audit can run using the local files already downloaded. To restore those three files from the official commit:

```sh
phase=phase_03_multilingual_finetuning
out=outputs/phase_03_multilingual_finetuning
py="$phase/.venv/bin/python"
"$phase/.venv/bin/hf" download FacebookAI/xlm-roberta-base \
  config.json tokenizer_config.json tokenizer.json \
  --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 \
  --local-dir "$out/tokenizer-e73636d"
```

Run neural commands only after the reviewed dataset and checkpoint are available and the run is explicitly approved. Provide a local, audited English checkpoint path for 3A. 3B uses the original pretrained Hub weights and saves the checkpoint selected by validation macro-F1.

```sh
"$py" "$phase/scripts/evaluate_zero_shot.py" --run-model \
  --checkpoint /path/to/english-xlmr --output "$out/phase_3a"
"$py" "$phase/scripts/train_multilingual.py" --run-model --seed 42 \
  --output "$out/training-42"
"$py" "$phase/scripts/evaluate_multilingual.py" --run-model \
  --checkpoint "$out/training-42/best_checkpoint" --output "$out/phase_3b"
"$py" "$phase/scripts/generate_plots.py" --a "$out/phase_3a" --b "$out/phase_3b" \
  --training-dir "$out/training-42" --bootstrap-unit context --output "$out/comparison-42"
```

If the English checkpoint is unavailable, the separate fallback trains only on the pinned official English train and validation data. It reports English validation scores separately. The subsequent 3A evaluation uses the frozen Phase 2 test set. Record this fallback as a new English checkpoint in the experiment provenance.

```sh
"$py" "$phase/scripts/train_english.py" --run-model --source-dir ../../data \
  --output "$out/english-fallback"
```

For the sensitivity comparison, run both training commands with `--variant document_disjoint` and omit `--source-dir` from the English command. Use distinct output directories. Use `--device cpu`, `--device mps` or `--device cuda` to override automatic selection. A local cache can be used for pretrained weights with `--offline`. If needed, supply reviewed Phase 1 hyperparameters or hardware adjustments as a YAML `training` mapping with `--training-config` and document changes to effective batch size or optimizer updates.

## Memory-capped Mac pilot

The first run stopped under memory pressure. Use the pinned base weights and dataset and start from step zero. The Mac configuration trains all parameters with batch size 1, accumulation 16, and gradient checkpointing. The MPS cap is 65% of Metal's recommended working set. The allocator raises an error if a device allocation exceeds that limit. [PyTorch MPS memory cap](https://docs.pytorch.org/docs/2.8/generated/torch.mps.set_per_process_memory_fraction.html)

Run this command only after explicit approval. It uses the local Hugging Face cache, logs each optimizer step, and exits after five updates if it fits. The pilot skips validation and checkpoint creation. Its elapsed time and memory report are diagnostics.

```sh
phase=phase_03_multilingual_finetuning
out=outputs/phase_03_multilingual_finetuning
HF_HUB_CACHE="$PWD/$out/hf-cache" "$phase/.venv/bin/python" -u \
  "$phase/scripts/train_multilingual.py" --run-model --offline --device mps \
  --variant span_checked --seed 42 --training-config "$phase/configs/mac_low_memory.yaml" \
  --mps-memory-fraction 0.65 --pilot-steps 5 --output "$out/mac-memory-pilot-42"
```

The MPS pilot hit the 7.70 GiB cap before step one. The CPU pilot completed all five optimizer steps in 26 seconds with four threads. Its result is in `outputs/phase_03_multilingual_finetuning/mac-cpu-pilot-42/memory_pilot.json`.

The first full CPU run stopped after 66 optimizer steps, before its first epoch checkpoint. The command below starts from the pinned pretrained weights on the same span-checked snapshot. It disables gradient checkpointing to use available RAM and avoid recomputing activations. It uses four threads, batch size 1, and accumulation 16. Validation and model checkpoints run at each epoch boundary.

```sh
phase=phase_03_multilingual_finetuning
out=outputs/phase_03_multilingual_finetuning
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  HF_HUB_CACHE="$PWD/$out/hf-cache" "$phase/.venv/bin/python" -u \
  "$phase/scripts/train_multilingual.py" --run-model --offline --device cpu \
  --variant span_checked --seed 42 --training-config "$phase/configs/mac_cpu_throughput.yaml" \
  --output "$out/training-cpu-throughput-span-checked-42"
```

## Mac GPU training

The CPU run above stopped after 54 optimizer steps for the requested GPU restart. The 10.06 GiB MPS cap still failed before one update with variable-length batches. The active GPU configuration right-pads each batch to the existing 384-token window limit so Metal can reuse one input shape. It preserves the dataset, token windows, span labels, and the five-epoch AdamW schedule. The padded positions participate in the model's standard QA training loss. The decoder still considers only eligible context positions. [Transformers guidance on the MPS graph cache](https://github.com/huggingface/transformers/blob/main/docs/source/en/apple_silicon.md#limit-graph-cache-growth)

```sh
phase=phase_03_multilingual_finetuning
out=outputs/phase_03_multilingual_finetuning
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.70 \
  HF_HUB_CACHE="$PWD/$out/hf-cache" "$phase/.venv/bin/python" -u \
  "$phase/scripts/train_multilingual.py" --run-model --offline --device mps \
  --variant span_checked --seed 42 --training-config "$phase/configs/mac_mps_fixed_padding.yaml" \
  --mps-memory-fraction 0.85 --output "$out/training-mps-fixed-padding-span-checked-42"
```

## Continue the GPU run on CPU

After the GPU completes its first validation and epoch checkpoint, stop the GPU process and use the original output directory. The runner checks the pinned data, seed, and training configuration, then restores the checkpoint's optimizer, scheduler, and Trainer state. The CPU configuration keeps the 384-token padding and disables activation recomputation. The continued run keeps the original 2,240-step schedule and validation history. The saved training time covers the resumed segment.

```sh
phase=phase_03_multilingual_finetuning
out=outputs/phase_03_multilingual_finetuning
run="$out/training-mps-fixed-padding-span-checked-42"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  HF_HUB_CACHE="$PWD/$out/hf-cache" "$phase/.venv/bin/python" -u \
  "$phase/scripts/train_multilingual.py" --run-model --offline --device cpu \
  --variant span_checked --seed 42 --training-config "$phase/configs/mac_cpu_resume.yaml" \
  --output "$run" --resume-from-checkpoint "$run/epoch_checkpoints/checkpoint-448"
```

## Continue on a cluster GPU

Transfer the complete epoch checkpoint privately to the cluster. Keep the checkout's relative run path so the saved best-checkpoint path also resolves on the cluster. The run's `launch_inputs.json` and the Phase 2 manifest and three flat splits are needed to verify its identity. Use the current Phase 3 source and configurations from the Mac checkout. If a later epoch is transferred, include its previous best checkpoint when it differs from the resume checkpoint.

If a teammate handles cluster access, they can verify the private handoff archive with `sha256sum -c` and extract it into an empty directory with `tar -xf`. The archive includes the checkout files, complete checkpoint, run metadata, and pinned splits. Run the continuation command below from the extracted directory.

Run this from the Mac checkout after the first epoch checkpoint is complete. Set `cluster` to the SSH login or host alias and `cluster_repo` to the cluster checkout's path.

```sh
run=outputs/phase_03_multilingual_finetuning/training-mps-fixed-padding-span-checked-42
bundle=phase_02_multilingual_benchmark/outputs/gemini-v2
rsync -avP --relative \
  "$run/launch_inputs.json" "$run/epoch_checkpoints/checkpoint-448/" \
  "$bundle/manifest.json" "$bundle/span_checked/train.jsonl" \
  "$bundle/span_checked/validation.jsonl" "$bundle/span_checked/test.jsonl" \
  "$cluster:$cluster_repo/"
```

Run the continuation from the cluster checkout root inside a job with one visible NVIDIA GPU and a Linux Python environment with CUDA-enabled PyTorch 2.8.0 and the Phase 3 requirements.

```sh
phase=phase_03_multilingual_finetuning
run=outputs/phase_03_multilingual_finetuning/training-mps-fixed-padding-span-checked-42
python -u "$phase/scripts/train_multilingual.py" --run-model --device cuda \
  --variant span_checked --seed 42 --training-config "$phase/configs/mac_cpu_resume.yaml" \
  --output "$run" --resume-from-checkpoint "$run/epoch_checkpoints/checkpoint-448"
```

The command loads the pinned base model revision from Hugging Face. If the compute node lacks internet access, prepopulate a Hugging Face cache with that revision, point `HF_HUB_CACHE` at the cache and pass `--offline`. The optimizer and scheduler carry over. The CUDA random-number stream differs from the Mac's Metal stream.

## Predictions, comparisons and handoff

The evaluation commands save one row per question and language with the exact predicted text and half-open character bounds, all gold references, seed, per-item EM/F1 and context lengths. Supply `--save-window-logits` when replayable window logits are needed. The standalone scoring and postprocessing scripts check their artifact hashes before use.

The reporter compares the same test hashes, reference annotations and language IDs. It emits score tables, performance gaps, correctness distributions, pairwise agreement, matched slices, paired bootstrap intervals and figures. Error samples retain both predictions for the same question and leave taxonomy fields blank for reviewers. Supply `--annotations` with the completed worksheet to add a human-coded error chart. Evaluation outputs use the shared multilingual metric for comparisons and preserve a separate English SQuAD score.

Repeat 3B using seeds 123 and 2026 if approved and feasible. Keep each checkpoint and prediction directory distinct. Pass the evaluated run directories to `summarize_seeds.py --runs` to obtain means and sample standard deviations. Bootstrap uncertainty over questions and variation across training seeds are separate quantities.

For Phase 5 and Phase 7, provide the two versioned checkpoints, dataset and test hashes, configuration, question-level predictions, measurements and review status. Phase 5 should retain the same QA recipe and use the approved Indic-AEC adaptation checkpoint as its initialization. Checkpoint links and measured performance remain pending.
