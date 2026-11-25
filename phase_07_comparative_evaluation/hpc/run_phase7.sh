#!/usr/bin/env bash
# Phase 7 Native frozen-checkpoint evaluation on one CUDA GPU. Run from the AgXQA-Indic checkout root:
#   bash phase_07_comparative_evaluation/hpc/run_phase7.sh
# Steps: environment and dataset checks -> tests -> checkpoint verification -> smoke runs (same four IDs) ->
# full inference + shared scoring -> paired statistics, error sample, translated reuse -> text-free report and plots.
# Primary models (3A, 3B) are fail-closed. The declared Phase 5 diagnostic (alpha 1 vs alpha 0) is reported
# separately; if one of its checkpoints fails verification, the diagnostic is skipped and the blocker is recorded.
# Rerun with the same PHASE7_RUN to resume: completed steps are skipped.
set -euo pipefail

py="${PHASE7_PYTHON:-$PWD/phase_03_multilingual_finetuning/.venv/bin/python}"
step1="${PHASE7_STEP1:-$PWD/outputs/phase_07_comparative_evaluation/step1}"
run="${PHASE7_RUN:-$PWD/outputs/phase_07_comparative_evaluation/runs/native-$(date -u +%Y%m%dT%H%M%SZ)}"
batch="${PHASE7_BATCH_SIZE:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
export CUDA_VISIBLE_DEVICES
manifest="$step1/manifest.json"
gold="$step1/phase6_test.shared.jsonl"
mkdir -p "$run/logs" "$run/analysis"
echo "Phase 7 run: $run (GPU $CUDA_VISIBLE_DEVICES)"
log() { echo "== $1 $(date -Is)"; }
evaluate() { "$py" -m phase_07_comparative_evaluation.evaluate_frozen --manifest "$manifest" "$@"; }

log environment
test -x "$py"
{ df -h "$run"; nvidia-smi; } > "$run/logs/environment.txt"
"$py" -c 'import torch, transformers; print(torch.__version__, transformers.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' | tee -a "$run/logs/environment.txt"
"$py" -m shared.validate "$gold" | tee "$run/logs/native_validate.json"

log tests
# Full suite recorded for the run log; the modules this run depends on must pass.
"$py" -m unittest discover -s tests > "$run/logs/tests_full.log" 2>&1 || echo "Full test suite reported failures; see logs/tests_full.log"
"$py" -m unittest tests.test_phase07_native tests.test_phase03_pipeline tests.test_phase05_alpha_sweep 2>&1 | tail -3

log verify
models="phase_3a phase_3b"
evaluate --model phase_3a --verify-only | tee "$run/logs/verify-phase_3a.log"
evaluate --model phase_3b --verify-only | tee "$run/logs/verify-phase_3b.log"
diagnostic=1
for model in phase_5_alpha_0 phase_5_alpha_1; do
  if ! evaluate --model "$model" --verify-only > "$run/logs/verify-$model.log" 2>&1; then
    diagnostic=0; echo "Diagnostic blocked: $(tail -1 "$run/logs/verify-$model.log")" | tee "$run/logs/diagnostic_blocker.txt"
  fi
done
if [ "$diagnostic" = 1 ]; then models="$models phase_5_alpha_0 phase_5_alpha_1"; fi

log smoke
for model in $models; do
  [ -f "$run/smoke-$model/run_manifest.json" ] || evaluate --model "$model" --output "$run/smoke-$model" \
    --device cuda --batch-size 4 --smoke 2>&1 | tee "$run/logs/smoke-$model.log"
done
"$py" - "$run" $models <<'PY'
import json, sys
from pathlib import Path
run, models = Path(sys.argv[1]), sys.argv[2:]
ids = {}
for model in models:
    rows = [json.loads(line) for line in (run / f"smoke-{model}" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = [(row["id"], row["language"]) for row in rows]
    if len(keys) != 4 or len(set(keys)) != 4 or sorted(lang for _, lang in keys) != ["bn", "hi", "mr", "ta"]:
        sys.exit(f"Smoke check failed for {model}: {keys}")
    if not json.loads((run / f"smoke-{model}" / "run_manifest.json").read_text(encoding="utf-8"))["smoke"]:
        sys.exit(f"{model}: smoke marker missing")
    ids[model] = sorted(keys)
if len({tuple(v) for v in ids.values()}) != 1:
    sys.exit(f"Smoke runs cover different IDs: {ids}")
print("Smoke runs passed: four IDs, one per language, identical across", ", ".join(models))
PY

log inference
for model in $models; do
  if [ ! -f "$run/$model/run_manifest.json" ]; then
    evaluate --model "$model" --output "$run/$model" --device cuda --batch-size "$batch" 2>&1 | tee "$run/logs/$model.log"
  fi
  [ -f "$run/$model/metrics.json" ] || "$py" -m shared.evaluate "$gold" "$run/$model/predictions.jsonl" \
    --output "$run/$model/metrics.json"
done
"$py" - "$run" "$gold" $models <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
run, gold, models = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
expected = Counter(json.loads(line)["language"] for line in gold.read_text(encoding="utf-8").splitlines() if line.strip())
for model in models:
    metrics = json.loads((run / model / "metrics.json").read_text(encoding="utf-8"))
    counts = {lang: value["count"] for lang, value in metrics["by_language"].items()}
    if metrics["count"] != sum(expected.values()) or counts != dict(expected):
        sys.exit(f"{model}: scored {metrics['count']} rows {counts}, expected {dict(expected)}")
    f1 = {lang: round(value["f1"], 2) for lang, value in metrics["by_language"].items()}
    print(f"{model}: {metrics['count']} rows scored; F1 {f1}; Indic macro F1 {sum(v['f1'] for v in metrics['by_language'].values()) / 4:.2f}")
PY

log statistics
stats() { "$py" -m phase_07_comparative_evaluation.statistical_analysis "$@"; }
[ -f "$run/analysis/native_phase_3b_vs_phase_3a.json" ] || stats native --gold "$gold" \
  --control "$run/phase_3a/predictions.jsonl" --candidate "$run/phase_3b/predictions.jsonl" \
  --control-label phase_3a --candidate-label phase_3b --output "$run/analysis/native_phase_3b_vs_phase_3a.json"
if [ "$diagnostic" = 1 ] && [ ! -f "$run/analysis/native_phase_5_alpha_1_vs_alpha_0.json" ]; then
  stats native --gold "$gold" --control "$run/phase_5_alpha_0/predictions.jsonl" \
    --candidate "$run/phase_5_alpha_1/predictions.jsonl" --control-label phase_5_alpha_0 \
    --candidate-label phase_5_alpha_1 --output "$run/analysis/native_phase_5_alpha_1_vs_alpha_0.json"
fi
if [ "$diagnostic" = 1 ] && [ ! -f "$run/analysis/native_phase_5_alpha_0_vs_phase_3b.json" ]; then
  # Rerun variation: the alpha-0 rerun repeats the 3B recipe, so this shows how much an identical rerun moves.
  stats native --gold "$gold" --control "$run/phase_3b/predictions.jsonl" \
    --candidate "$run/phase_5_alpha_0/predictions.jsonl" --control-label phase_3b \
    --candidate-label phase_5_alpha_0 --output "$run/analysis/native_phase_5_alpha_0_vs_phase_3b.json"
fi
[ -f "$run/analysis/translated.json" ] || stats translated --output "$run/analysis/translated.json"
[ -f "$run/analysis/native_error_counts.json" ] || stats errors --gold "$gold" --control-run "$run/phase_3a" \
  --candidate-run "$run/phase_3b" --private-output "$run/analysis/native_error_sample.private.csv" \
  --counts-output "$run/analysis/native_error_counts.json"

log report
[ -d "$run/report" ] || stats report --run "$run" --output-dir "$run/report"
"$py" -m phase_07_comparative_evaluation.generate_plots --results "$run/report" --output-dir "$run/report/plots"

archive="$run.tar"
tar cf "$archive.partial" -C "$(dirname "$run")" "$(basename "$run")"
mv "$archive.partial" "$archive"
echo "Done. Copy back: $archive"
