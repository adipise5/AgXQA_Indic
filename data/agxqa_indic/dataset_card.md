# AgXQA-Indic — draft dataset card

**Selected dataset:** Existing Gemini 3.5 Flash-Lite translations. The local bundle contains a span-checked subset. Human review and English source adjudication are pending. [Phase 2 methodology](../../phase_02_multilingual_benchmark/README.md)

Source: [`msu-ceco/agxqa_v1`](https://huggingface.co/datasets/msu-ceco/agxqa_v1), revision `9fb0cf2a6d03ccfccbd67d5966179b43b5b77e2c`. File hashes and actual source defects are in [source_preflight.json](../../phase_02_multilingual_benchmark/results/source_preflight.json). Raw official source bytes remain unchanged. Language: English; target additions: Hindi, Marathi, Bengali, Tamil. Domain: agriculture/irrigation; task: extractive question answering.

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| Verified official English source | 1,503 | 353 | 330 |
| Verified export IDs shared across available languages | 1,304 | 347 | 327 |
| Span-checked five-language groups | 1,304 | 347 | 322 |

Within each prepared split, all five languages contain the same retained IDs. Every English row comes from the pinned source. The provisional test holds out five groups pending source adjudication. The `gemini-v2` bundle preserves the original exports and records Unicode normalization, span corrections, and the retained and excluded IDs.

Human review target: all test questions; seeded 5% train and 10% validation samples; all flagged candidates; separate English source adjudication. Actual coverage is currently zero. Assigned reviewers: Aditya Saxena (Hindi), Aditya Pise (Marathi), Saunak (Bengali), Dhiraj (Tamil). Ratings, identity, decisions and timestamps must come from actual reviewers. No inter-annotator agreement is claimed.

Known limitations: nine invalid English gold offsets across five test questions, including a placeholder-like gold string; 39 citation groups and 207 exact-context groups cross official splits. Preserve this overlap for baseline comparability and disclose it. Citation-based document groups are approximate and do not constitute semantic deduplication. Validate extractability on the exported passages before construction. Test annotation changes must be versioned separately from Phase 1's original-source results.

Intended use: controlled multilingual agricultural QA research and cross-lingual consistency analysis. Quality metrics, exclusions, final release version, hosted data URL, file hashes, actual execution metadata and downstream handoff remain pending. Do not infer field reliability or agricultural advice quality from this draft. Keep independently authored native test documents out of later adaptation corpora.

Rights: data retain upstream terms. Dataset/source-document redistribution rights and the final dataset license require verification before publishing research JSONL. No credential or model weight is included. Synthetic test fixtures live separately under `tests/` and `data/examples/` and never count as research data.
