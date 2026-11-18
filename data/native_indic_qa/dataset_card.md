# Dataset card: IndicAgriQA-Native (Phase 6)

**Status (2026-11-18):** the QA pairs are generated and have passed automatic validation. **Human review is still pending.** Passages are **not included in this repository** (see Licence).

## Summary

IndicAgriQA-Native is an extractive question-answering benchmark in Hindi, Marathi, Bengali and Tamil. It is built from agricultural extension articles written in those languages. Every question was generated directly in the language of its passage. Nothing was translated from English. Each answer is an exact contiguous span of its context, and the answer offset is computed in code. The dataset is **test-only**: it must not be used for training or domain adaptation.

| Language | QA pairs | Source documents | Contexts | Irrigation QA | Mean words (context / question / answer) | Human-validated |
|---|---|---|---|---|---|---|
| Hindi (hi) | 1,107 | 21 | 392 | 50 | 123 / 12.6 / 3.6 | 0 |
| Marathi (mr) | 581 | 20 | 203 | 117 | 123 / 8.3 / 3.2 | 0 |
| Bengali (bn) | 420 | 18 | 146 | 27 | 102 / 9.1 / 3.3 | 0 |
| Tamil (ta) | 243 | 11 | 85 | 71 | 108 / 7.6 / 3.4 | 0 |
| **Total** | **2,351** | 70 | 826 | 265 | | 0 |

The domain and question-type labels come from the generator and have not been reviewed.

- **Question types** (all languages): numerical 771, factual 687, procedural 299, temporal 214, identification 117, causal 108, conditional 107, benefit 47, unlabelled 1.
- **Difficulty:** easy 1,248, medium 1,054, hard 49.

## Format

Two files are produced, with the same records in each:

- `final/test.jsonl` uses the SQuAD-style schema from the Phase 6 plan: `id`, `title`, `language`, `context`, `question`, `answers.text`, `answers.answer_start` and `metadata`.
- `final/test.shared.jsonl` uses the shared contract in `schemas/qa.schema.json`. Each record has `split: "test"` and `source_document_id`, and `answers` is a list of `{text, answer_start}`. It passes `python -m shared.validate`.

IDs are `NATIVE_HI_000001`, `NATIVE_MR_000001` and so on. They are **not aligned across languages**, so no cross-lingual consistency score applies.

`metadata` holds:

- provenance: `document_id`, `context_id`, `candidate_id`, `source_url`
- labels: `domain`, `question_type`, `difficulty`
- generation: `generation_model`, `prompt_version`
- review: `human_validated`, `review_status`, `reviewer`

`metadata.csv` (in this folder) lists the same metadata per ID, without passages.

## Construction

1. **Sources.** 70 Unicode HTML articles from Vikaspedia (MeitY / C-DAC). Each article credits its original publisher, and the credits are recorded per document in `phase_06_native_indic_qa/sources/*_sources.csv`:
   - Bihar, Jharkhand, UP and MP agriculture departments; Kisan Portal; ICAR-NCIPM; ICAR-CAZRI
   - Maharashtra agriculture department; Agrowon
   - TNAU, *Valarum Velanmai*, KVK Needamangalam, Sugarcane Research Station Sirugamani
   - Vikaspedia West Bengal team

   University and ICAR PDFs were excluded because their text layers were corrupt: legacy fonts, mis-mapped glyphs, or doubled vowel signs with dropped half-letters. They are listed as `include=no`, with reasons.
2. **Extraction and cleaning.** The article HTML embedded in each page was extracted with headings, paragraphs and table rows kept. Then:
   - NFC normalisation, with ZWJ/ZWNJ preserved
   - removal of source-credit lines
   - script and language filtering, including Hindi vs Marathi function-word votes
   - removal of exact and near-duplicate paragraphs
3. **Contexts.** 827 passages of 100–200 words, each within one section, with XLM-R token counts. 61 exceed 384 tokens and are flagged.
4. **Generation.** `gemini-3.5-flash-lite`, temperature 0.2, prompt v2 (`prompts/qa_generation_prompt.txt`, sha256 prefix `24e20dd19be5`). One call per context; 2,506 candidates in total.
5. **Automatic validation.** 2,460 candidates passed. Checks:
   - the answer is an exact substring of the context
   - the evidence sentence disambiguates repeated answers
   - the question's language and script are correct
   - answer length is within limits
   - the answer does not appear inside the question

   2,351 remained after removing exact and fuzzy duplicate questions and capping each context at 3 questions.
6. **Human review.** Pending. One native-speaker reviewer per language (Hindi: Aditya Saxena; Marathi: Aditya Pise; Bengali: Saunak; Tamil: Dhiraj). Native-speaker review is not agricultural-expert verification.
7. **Leakage control.** `phase_06_native_indic_qa/sources/phase4_exclusion_list.csv` lists the 84 documents that Phase 4 (Indic-AEC) must not ingest. The hash, URL, title and passage-overlap check (`check_data_leakage.py`) has **not run yet**, because Phase 4 has no corpus.

## Known limitations

- **Topic imbalance.** Irrigation is 11% of QA pairs overall, against a 30% target. Hindi is dominated by pest-management content, and Bengali has the least irrigation (6%).
- **Language sizes are uneven:** Hindi has 4.6 times as many pairs as Tamil. Report macro-averages across languages.
- **Generated questions.** An LLM wrote the questions, so they may be more formal or regular than farmers' real questions. About 80 use a deictic "this …", and 3 contain a mixed-script typo. These are flagged for review.
- **Origin.** Some Kisan Portal and Bengali "portal content team" articles may be translations. Their credits are recorded, and reviewers should confirm native origin.

## Licence

- **Passages:** Vikaspedia content may be reproduced only with permission (vikaspedia@cdac.in) and attribution. Nine articles are third-party newspaper or magazine content: Agrowon (7 documents), Dinamani, Pasumai Vikatan and "contemporary newspapers". Vikaspedia's permission does not cover them. Until permission is granted, passages and QA files are distributed privately, not in this repo.
- **Annotations:** licence to be decided by the team.
