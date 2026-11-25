# Phase 5 — Domain Adaptation Using Indic-AEC

**Owner:** Aditya Pise  
**Status:** Masked-language adaptation pilot and literal overlap audit complete. Matched QA training and evaluation pending.

## Measured work

The pilot updated the final encoder layer of pinned XLM-R Base on reviewed Indic-AEC release `3f3481d6662a2b69`. It used seed 42, 39 training sequences, and 77 validation sequences. On the same 2,948 masked validation positions, loss moved from **1.745622** to **1.738547** after one epoch. Reloading the saved layer reproduced the final loss. [Pilot metrics](results/mlm_pilot_metrics.json) · [Input manifest](results/mlm_input_manifest.json) · [Earlier corpus pilot](results/earlier_mlm_pilot_metrics.json)

The three Gemini v2 exports plus the pinned English source reproduced all three Phase 3 split checksums. The six exact training excerpts produced zero word 5-gram or character 10-gram flags at thresholds 0.30, 0.40, and 0.50 against **3,370** retained Phase 2 validation/test candidates and **1,960** Phase 6 reference passages and windows. The Phase 2 count includes the five provisional test groups held out of scoring. The Phase 6 reservation scan produced zero hits. Four reconstructed Native source pages differ from their original HTML snapshots. These checks cover literal reuse. [Text-free overlap audit](results/phase3_span_checked_overlap.json)

## Run the matched QA comparison

Use a CUDA GPU, the Phase 3 training environment, and enough scratch space for full model and optimizer checkpoints. Budget about 12 GiB of free space for checkpoints with the base model already cached. Use a fresh output directory.

Obtain the exact pilot's two files privately and place them together in the adaptation directory:

- `metrics.json`
- `final_encoder_layer.safetensors` with SHA-256 `1e7122873ee756397f1dc1dbcf6a4e788a429967124c6fc7d9af12ff53e7cd76`

The default QA data location is the original Phase 3 bundle root, `phase_02_multilingual_benchmark/outputs/gemini-v2`. Pass `--bundle-dir` for a different original location. If you only have the three original flat translations, use `--export-dir` plus `--source-dir` for the pinned English files. The latter path rebuilds the splits in memory and requires the same serialized SHA-256s. It records the split-level verification alongside the comparison.

```sh
python -m phase_05_domain_adaptation.run_matched_qa \
  --adaptation-dir /path/to/private/phase5-mlm-run \
  --output outputs/phase_05_domain_adaptation/matched-qa-seed42 \
  --check-only

python -m phase_05_domain_adaptation.run_matched_qa \
  --adaptation-dir /path/to/private/phase5-mlm-run \
  --output outputs/phase_05_domain_adaptation/matched-qa-seed42
```

Add `--offline` if the pinned base is already present in the Hugging Face cache. The runner initializes the same XLM-R QA model and head as 3B, loads the adapted encoder layer, and trains all parameters with the recorded CUDA training configuration. Selection uses validation macro-F1. The evaluation pairs the predictions on the same **322 provisional question groups** against the recorded Phase 3B outputs using `agxqa-indic-v1`.

Keep `comparison.json`, saved training logs, both run manifests, and the checkpoint and prediction hashes. Keep the raw inputs and model weights in the private research workflow. Prediction redistribution follows the dataset's source terms.

## Reproduce the MLM pilot

From the repository root, run the following with the verified Phase 4 package, its pinned tokenizer cache, and a local directory for the pinned base weights and config. The official audit CLI can verify the supplied complete Phase 2 archive and rebuild Native reference coverage when you have both data sources.

```sh
python phase_05_domain_adaptation/prepare_mlm_data.py \
  --corpus-release /path/to/releases/3f3481d6662a2b69 \
  --tokenizer-cache /path/to/tokenizer_cache

HF_HUB_OFFLINE=1 python phase_05_domain_adaptation/train_mlm.py \
  --inputs outputs/phase_05_domain_adaptation/mlm_inputs/3f3481d6662a2b69-len256 \
  --cache-dir /path/to/tokenizer_cache \
  --model-dir /path/to/pinned/xlm-roberta-base \
  --device cpu --epochs 1
```

The existing matched audit binds the recorded pilot checkpoint checksum. A separately reproduced checkpoint with a different hash needs its own provenance and a separately labeled comparison. [Phase 4 instructions](../phase_04_indic_aec_corpus/README.md) · [Phase 3 instructions](../phase_03_multilingual_finetuning/README.md)
