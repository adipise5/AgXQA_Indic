#!/usr/bin/env bash
# Create the Phase 3 Python 3.11 environment on a Linux GPU host.
# Run from the AgXQA-Indic checkout root:  bash phase_03_multilingual_finetuning/hpc/setup_env.sh
# Optional: TORCH_INDEX=https://download.pytorch.org/whl/cu126 to pick a different CUDA wheel.
set -euo pipefail

phase=phase_03_multilingual_finetuning
venv="${VENV:-$phase/.venv}"
torch_index="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ ! -x "$venv/bin/python" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$venv"
  elif command -v conda >/dev/null 2>&1; then
    conda create -y -p "$PWD/$phase/.conda311" python=3.11
    "$PWD/$phase/.conda311/bin/python" -m venv "$venv"
  elif command -v uv >/dev/null 2>&1; then
    uv venv --seed --python 3.11 "$venv"
  else
    echo "Python 3.11 not found. Load a module (e.g. 'module load python/3.11') or install conda/uv." >&2
    exit 1
  fi
fi

py="$venv/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install "torch==2.8.0" --index-url "$torch_index"
"$py" -m pip install -r "$phase/requirements.txt"
"$py" - <<'EOF'
import sys, torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| transformers", transformers.__version__)
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch. On a scheduler, run this check inside a GPU job.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu {i}: {p.name} {p.total_memory / 2**30:.1f} GiB")
EOF
"$py" -m unittest tests.test_phase03_pipeline
