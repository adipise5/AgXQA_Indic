#!/usr/bin/env bash
# Phase 5 adaptation-strength sweep on one CUDA GPU. Run from the AgXQA-Indic checkout root:
#   bash phase_05_domain_adaptation/hpc/run_alpha_sweep.sh
# Order: preflight -> alpha 0 and SWEEP_ALPHA at seed 42 (+ validation of the existing alpha-1 run) -> validation
# selection -> span-cap check -> extra paired seeds for the control and frozen candidate -> test -> comparison.
# Completed steps are skipped on rerun. A crashed training run leaves a partial directory: move it aside, then rerun.
set -euo pipefail

py="${PYTHON:-phase_03_multilingual_finetuning/.venv/bin/python}"
out="${OUT:-outputs/phase_05_domain_adaptation/alpha-sweep}"
adapt="${ADAPTATION_DIR:-outputs/phase_05_domain_adaptation/mlm_runs/3f3481d6662a2b69-len256-last-layer-1ep-lr2e-05-seed42-cpu}"
alpha1="${ALPHA1_RUN:-outputs/phase_05_domain_adaptation/matched-qa-seed42}"
sweep_alpha="${SWEEP_ALPHA:-0.5}"
extra_seeds="${EXTRA_SEEDS:-123 2026}"
larger_cap="${LARGER_CAP:-160}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/outputs/phase_03_multilingual_finetuning/hf-cache}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
export CUDA_VISIBLE_DEVICES
data="--adaptation-dir $adapt --offline${BUNDLE_DIR:+ --bundle-dir $BUNDLE_DIR}"
mkdir -p "$out/logs"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used,memory.total --format=csv > "$out/logs/gpu-$(date +%Y%m%dT%H%M%S).csv"
echo "GPU $CUDA_VISIBLE_DEVICES, output $out"

sweep() {  # sweep <log-name> <subcommand> [args...]
  local name=$1; shift
  echo "== $name $(date -Is)"
  "$py" -m phase_05_domain_adaptation.alpha_sweep "$@" 2>&1 | tee "$out/logs/$name.log"
}
train() {  # train <alpha> <seed> [extra flags]
  local run="$out/alpha$1-seed$2"
  if [ -f "$run/validation/metrics.json" ]; then echo "skip $run"; return; fi
  if [ -e "$run" ]; then echo "Partial run $run: move it aside and rerun." >&2; exit 1; fi
  sweep "train-alpha$1-seed$2" train $data --alpha "$1" --seed "$2" --output "$run" "${@:3}"
}
test_run() {
  if [ -f "$1/test/metrics.json" ]; then echo "skip test $1"; return; fi
  sweep "test-$(basename "$1")" test $data --run "$1" --selection "$out/selection.json"
}

[ -f "$out/preflight.json" ] || sweep preflight check $data --require-cuda --larger-cap "$larger_cap" --record "$out/preflight.json"

train 0 42 --save-validation-logits
train "$sweep_alpha" 42
candidates="$out/alpha$sweep_alpha-seed42"
if [ -f "$alpha1/training/best_checkpoint/model.safetensors" ]; then
  [ -f "$alpha1/validation/metrics.json" ] || sweep validate-alpha1-seed42 validate $data --run "$alpha1"
  candidates="$candidates $alpha1"
else
  echo "Existing alpha-1 run not found at $alpha1; selecting among the sweep runs only."
fi
[ -f "$out/selection.json" ] || sweep select select --control "$out/alpha0-seed42" --candidates $candidates --output "$out/selection.json"
[ -f "$out/span_cap_validation.json" ] || sweep span-cap span-cap $data --run "$out/alpha0-seed42" \
  --larger-cap "$larger_cap" --output "$out/span_cap_validation.json"

read -r selected_alpha selected_run < <("$py" -c "
import json, sys; s = json.load(open(sys.argv[1]))
print(format(s['selected_alpha'], 'g') if s['selected_alpha'] is not None else '-', s['selected_run'] or '-')" \
  "$out/selection.json")
controls="$out/alpha0-seed42"
picked=""
if [ "$selected_alpha" != "-" ]; then
  echo "Frozen candidate: alpha $selected_alpha ($selected_run)"
  picked="$selected_run"
  for seed in $extra_seeds; do
    train 0 "$seed"
    train "$selected_alpha" "$seed"
    controls="$controls $out/alpha0-seed$seed"
    picked="$picked $out/alpha$selected_alpha-seed$seed"
  done
else
  echo "No candidate kept English and beat Indic validation F1; keeping the alpha-0 control, no extra adaptation runs."
fi
for run in $controls $picked; do test_run "$run"; done
if [ "$selected_alpha" = "-" ]; then controls=""; fi

[ -d "$out/comparison" ] || sweep compare compare --selection "$out/selection.json" \
  --control-runs $controls --candidate-runs $picked --output "$out/comparison"

extra=""
for path in "$alpha1/validation" "$alpha1/test"; do
  if [ -d "$path" ]; then extra="$extra $path"; fi
done
archive="$out/alpha-sweep-results.tar"
tar cf "$archive.partial" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' \
  --exclude='epoch_checkpoints' --exclude='window_logits.jsonl' --exclude='tokenizer.json' \
  --exclude='sentencepiece.bpe.model' --exclude='alpha-sweep-results.tar*' "$out" $extra
mv "$archive.partial" "$archive"
echo "Done. Copy back: $archive"
