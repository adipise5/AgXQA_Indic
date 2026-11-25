#!/usr/bin/env bash
# Run the full Phase 3 pipeline for one seed on one CUDA GPU:
#   preflight -> 3B multilingual training -> 3B test evaluation
#   -> 3A English fallback training -> 3A zero-shot evaluation -> 3A vs 3B comparison.
# Run from the AgXQA-Indic checkout root. Completed steps are skipped on rerun.
#
# Environment overrides:
#   SEED (42)  VARIANT (span_checked)  TRAINING_CONFIG (auto by GPU memory)
#   OFFLINE (1: use the bundled Hugging Face cache; 0: download the pinned revision)
set -euo pipefail

phase=phase_03_multilingual_finetuning
out=outputs/phase_03_multilingual_finetuning
py="${PYTHON:-$phase/.venv/bin/python}"
seed="${SEED:-42}"
variant="${VARIANT:-span_checked}"
tag="cuda-${variant//_/-}-$seed"
logs="$out/hpc-logs"
mkdir -p "$logs"

export HF_HUB_CACHE="$PWD/$out/hf-cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
offline=""
if [ "${OFFLINE:-1}" = 1 ]; then offline="--offline"; export HF_HUB_OFFLINE=1; fi

# The runners require exactly one visible GPU. Without a scheduler, pick the GPU with the most free memory.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1)"
fi
gpu_mib="$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.total --format=csv,noheader,nounits)"
if [ -z "${TRAINING_CONFIG:-}" ]; then
  if [ "$gpu_mib" -ge 10000 ]; then TRAINING_CONFIG="$phase/configs/cuda_hpc.yaml"
  else TRAINING_CONFIG="$phase/configs/mac_low_memory.yaml"; fi
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "$logs/$tag.pipeline.log"; }
step() {  # step NAME DONE_MARKER COMMAND...
  local name="$1" done_marker="$2"; shift 2
  if [ -e "$done_marker" ]; then log "skip $name (found $done_marker)"; return; fi
  log "start $name"
  "$@" 2>&1 | tee "$logs/$tag.$name.log"
  log "done $name"
}

log "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES (${gpu_mib} MiB) seed=$seed variant=$variant config=$TRAINING_CONFIG"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" | tee -a "$logs/$tag.pipeline.log"

train3b="$out/training-3b-$tag"
eval3b="$out/phase_3b-$tag"
train3a="$out/english-fallback-$tag"
eval3a="$out/phase_3a-$tag"
compare="$out/comparison-$tag"

step preflight "$out/preflight-$tag.json" \
  "$py" "$phase/scripts/validate_dataset.py" --variant "$variant" --output "$out/preflight-$tag.json"

step train_3b "$train3b/best_checkpoint/phase3_checkpoint.json" \
  "$py" -u "$phase/scripts/train_multilingual.py" --run-model $offline --device cuda \
  --variant "$variant" --seed "$seed" --training-config "$TRAINING_CONFIG" --output "$train3b"

step eval_3b "$eval3b/run_manifest.json" \
  "$py" -u "$phase/scripts/evaluate_multilingual.py" --run-model --device cuda --variant "$variant" \
  --checkpoint "$train3b/best_checkpoint" --output "$eval3b"

step train_3a "$train3a/best_checkpoint/phase3_checkpoint.json" \
  "$py" -u "$phase/scripts/train_english.py" --run-model $offline --device cuda \
  --source-dir data/agxqa_original/raw --seed "$seed" --training-config "$TRAINING_CONFIG" --output "$train3a"

step eval_3a "$eval3a/run_manifest.json" \
  "$py" -u "$phase/scripts/evaluate_zero_shot.py" --run-model --device cuda --variant "$variant" \
  --checkpoint "$train3a/best_checkpoint" --output "$eval3a"

step compare "$compare/comparison_manifest.json" \
  "$py" "$phase/scripts/generate_plots.py" --a "$eval3a" --b "$eval3b" \
  --training-dir "$train3b" --output "$compare"

step bootstrap_context "$compare-context-intervals.csv" \
  "$py" "$phase/scripts/statistical_significance.py" --a "$eval3a" --b "$eval3b" \
  --bootstrap-unit context --output "$compare-context-intervals.csv"

log "PHASE3_PIPELINE_COMPLETE seed=$seed"
