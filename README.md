# AgXQA-Indic

Multilingual agricultural question answering for English, Hindi, Marathi, Bengali and Tamil.

This repository organizes seven experiments to study cross-lingual transfer, multilingual fine-tuning, agricultural domain adaptation and evaluation on native-language QA. It contains working shared utilities and a [measured English AgRoBERTa reimplementation](phase_01_original_baseline/README.md): **54.24% EM / 78.23% F1** under SQuAD-style English scoring on all 330 official test questions. Published reference values are 55.15% / 78.89%. Raw research datasets and model weights are not committed. English XLM-R training and the other phases remain pending.

## Research questions
1. How much does English-trained QA degrade across Indic languages?
2. Does multilingual fine-tuning reduce that gap?
3. Does Indic-AEC domain adaptation improve performance under controlled conditions?
4. Do gains transfer to independently authored native Indic QA?

## Phases and owners
| Phase | Experiment | Owner | Status |
|---|---|---|---|
| 1 | [Reproduce the Original AgXQA Baseline](phase_01_original_baseline/README.md) | Saunak; review by Dhiraj | Partial: AgRoBERTa measured; English XLM-R pending |
| 2 | [Create AgXQA-Indic](phase_02_multilingual_benchmark/README.md) | Saunak + language reviewers | Span-checked splits prepared. Human review and English source adjudication pending |
| 3 | [Multilingual Fine-Tuning and Cross-Lingual Evaluation](phase_03_multilingual_finetuning/README.md) | Saunak | Pending |
| 4 | [Build Our Own Indic-AEC](phase_04_indic_aec_corpus/README.md) | Aditya Pise | Pending |
| 5 | [Domain Adaptation Using Indic-AEC](phase_05_domain_adaptation/README.md) | Aditya Pise | Pending |
| 6 | [Build a Native Indic Agricultural QA Dataset](phase_06_native_indic_qa/README.md) | Aditya Saxena | Pending |
| 7 | [Final Comparative Evaluation](phase_07_comparative_evaluation/README.md) | Dhiraj | Pending |

Phase 2 uses the existing Gemini 3.5 Flash-Lite translations. The local bundle includes 1,304 train, 347 validation, and 322 provisional test question groups in all five languages. The final test targets 327 groups after adjudication. [Phase 2 preparation and review](phase_02_multilingual_benchmark/README.md#prepared-dataset)

## Workflow and datasets
Phase 1 establishes the English baseline. Phase 2 produces aligned AgXQA-Indic records for phase 3 (3A English-only training; 3B multilingual training). Phase 4 builds Indic-AEC for phase 5 adaptation. Phase 6 reserves native documents and constructs native QA. Phase 7 compares frozen predictions. Phase 6 test documents must be excluded from phase 4/5 training.

See `data/` dataset cards, `schemas/` contracts, `docs/annotation.md` and `docs/reproducibility.md`. Phase 1 records the pinned sources, verified official split counts and actual setup. Dataset terms remain upstream; original QA weights/settings are incomplete, so the result is labeled a reimplementation. Other dataset counts are not assumed.

## Quick start
Python 3.10+; the shared utilities and tests use only the standard library.
```sh
python -m unittest discover -s tests -v
python -m shared.validate data/examples/qa.jsonl
python -m shared.evaluate data/examples/qa.jsonl data/examples/predictions.jsonl --output /tmp/agxqa-demo.json
python -m shared.plan phase_03_multilingual_finetuning/config.json
```
The tiny example is explicitly synthetic and must never appear in reported research results. See [evaluation definitions](docs/evaluation.md). Phase 1 includes a tested, pinned AgRoBERTa workflow; Phase 2 includes a synthetically tested data pipeline for source preparation, validation and review. Other training/generation scripts remain explicit starters. The [Phase 1 instructions](phase_01_original_baseline/README.md#16-steps-to-reproduce) explain installation and saved-prediction rescoring; [Phase 2 instructions](phase_02_multilingual_benchmark/README.md#source-audit) explain how to audit the pinned English source.

## Results and remaining experiments
| Approach | EN F1 | HI F1 | MR F1 | BN F1 | TA F1 |
|---|---|---|---|---|---|
| AgRoBERTa reimplementation (SQuAD-style English) | 78.23 | N/A | N/A | N/A | N/A |
| Same frozen AgRoBERTa predictions (shared protocol v1) | 78.24 | N/A | N/A | N/A | N/A |
| XLM-R English-trained | Pending | Pending | Pending | Pending | Pending |
| XLM-R multilingual | Pending | Pending | Pending | Pending | Pending |
| XLM-R + Indic-AEC | Pending | Pending | Pending | Pending | Pending |

The two English rows use distinct normalizers and must not be conflated. Phase 1 reports SQuAD-style **54.24% EM / 78.23% F1** for comparison with the paper; shared-protocol rescoring yields **51.52% EM / 78.24% F1**. See [actual measurements, provenance and error counts](phase_01_original_baseline/results/baseline_results.json), [all test predictions](phase_01_original_baseline/results/test_predictions.jsonl), and [protocol differences](phase_01_original_baseline/README.md#9-evaluation-metrics). Native QA results, XLM-R, ablations and uncertainty estimates remain pending. `results/status.json` remains a tracker pointing to real artifacts.

## Team review and plan
- Aditya Saxena: Hindi review; phase 6.
- Dhiraj: Tamil review; Phase 1 completion review and Phase 7.
- Saunak: English Phase 1 delivery; Bengali review; phases 2 and 3.
- Aditya Pise: Marathi review; phases 4 and 5.

Day 1: freeze schemas/splits, audit translations, acquire sources and baseline setup. Day 2: review data, implement/run baseline and multilingual experiments, build corpus and native QA. Day 3: finish feasible adaptation, integrate predictions and report measured results and unfinished work honestly. This is a relative  plan, not a claim of completion or a newly confirmed calendar deadline.

## Limitations, references and license
Whitespace token F1 is a transparent starter metric with known Indic tokenization limits. Review coverage, translation provenance and data rights need verification. Compute availability may limit the day scope; optional RAG/agentic experiments are deferred. Record verified original AgXQA and model citations in `docs/references.md`. No publication DOI or citation is invented. Code license selection is pending owner decision; third-party data retain their original terms and are not automatically redistributable.

See [contribution workflow](CONTRIBUTING.md).
