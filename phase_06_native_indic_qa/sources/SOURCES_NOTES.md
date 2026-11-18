# Phase 6 source collection notes (checked 2026-11-18)

Each `include=yes` row in `<language>_sources.csv` was fetched in memory and returned clean Unicode text in its language. "Clean" means a script ratio of at least 0.96 and a garbled ratio of 0.00. Rows marked `include=no` are candidates kept for the record, and their `notes` column gives the reason.

| Language | Usable documents | ≈ Words | Irrigation documents | Main source |
|---|---|---|---|---|
| Tamil | 11 | 11.5k | 5 | Vikaspedia Tamil |
| Marathi | 20 | 27k | 5 | Vikaspedia Marathi |
| Hindi | 21 | 52k | 4 | Vikaspedia Hindi (Bihar, Jharkhand, UP and MP pages) |
| Bengali | 18 | 17k | 2 | Vikaspedia Bengali (West Bengal pages) |

The word counts include some page navigation text. As a rough guide, a 250-QA language needs about 120 contexts, or roughly 18k words of clean prose. Tamil falls short of that and Bengali is borderline. Both should still clear the 100-QA minimum.

## Decisions the team needs to make

1. **Vikaspedia licence.** The portal policy says content "may be reproduced free of charge after taking proper permission" by emailing vikaspedia@cdac.in, with attribution. Until that permission is granted, publish only the manifest and the scripts, not the passages.
2. **Phase 4 overlap (Aditya Pise).** Checked 2026-11-18 against adipise5/AgXQA-Indic. Phase 4 has not started: `phase_04_indic_aec_corpus/sources.csv` has a header only and its scripts are stubs. That means there is no overlap today. To keep it that way:
   - Phase 4 should skip every URL in `phase4_exclusion_list.csv` (84 documents: 70 reserved, 14 OCR candidates). Compare URLs only after passing them through `common.normalize_url`, because one Vikaspedia article has many URL spellings: language subdomains, hex vs Unicode slugs, and `?lgn=`.
   - Simplest rule to agree on: Phase 4 uses no Vikaspedia agriculture pages at all, since Phase 6 depends on them.
   - Once the Indic-AEC corpus exists, run `check_data_leakage.py --phase4-corpus`. A URL list cannot catch the same text republished on another site.
3. **Native origin.** All Vikaspedia rows are set to `language_origin=unknown`. Many pages are clearly regional: TNAU trials, Marathwada soils, West Bengal crops. Some Hindi pages could still be translations from English Vikaspedia. Reviewers should check each page for a translation or source note.
4. **TNAU Agritech Tamil pages** (`include=no`). These pages mirror the English TNAU portal file for file and include North-Indian crops such as wheat and berseem. They are most likely translations. They are also "© TNAU, all rights reserved".

## Article credits (from the downloaded pages)

Every Vikaspedia article ends with a source line. These lines are now copied into each row's `notes` as "Article credit". Most are government bodies:

- **Hindi:** Bihar, Jharkhand, UP and MP agriculture departments; Kisan Portal; ICAR-NCIPM; CAZRI.
- **Marathi:** Maharashtra agriculture department.
- **Tamil:** TNAU, TNAU's *Valarum Velanmai*, KVK Needamangalam, Sugarcane Research Station Sirugamani.
- **Bengali:** Vikaspedia team, West Bengal.

**Third-party press content.** Vikaspedia's permission does not extend to these, so they are flagged "THIRD-PARTY COPYRIGHT" in `notes`. Use them for evaluation, but do not republish their passages:

- Agrowon (MR_DOC_004, 005, 008, 010, 014, 017, 020)
- Dinamani (TA_DOC_004)
- Pasumai Vikatan (TA_DOC_011)
- "contemporary newspapers" (BN_DOC_015)

BN_DOC_018 is credited to a Bangladeshi scientist, so its content may be Bangladeshi rather than West Bengal practice.

## Why most university and ICAR PDFs were excluded

Three text-layer failure modes were found in real documents. The pipeline now detects all three (`garbled_ratio` in `scripts/common.py`), so `extract_text.py` sends affected pages to OCR instead of the LLM.

| Failure | Example | Seen in |
|---|---|---|
| Legacy non-Unicode font | `‘hmË‘m ’w$bo` for महात्मा फुले | MPKV Krishidarshani, Shri Sugi; TNAU farmer handbook |
| Glyphs mapped to the wrong code points | `षिद्यापीठाने`, `िाधनिंपत्ती` | MPKV 2025 Marathi recommendations |
| Doubled vowel signs and dropped half-letters | `कीी`, `उत्ादन`, `किलोग्ाम` | ICAR-CRRI bulletin 251, ICAR-IIMR millet books |

Image-only PDFs (VNMKV leaflets, TNAU *Valarum Velanmai* issues) have no text layer at all.

**OCR backlog.** These are the largest native sources, if someone can install Tesseract with the `mar`/`hin`/`tam` packs and spot-check the output:

- MPKV *Krishidarshani 2025*: 548 pages of Marathi.
- MPKV *Shri Sugi*: seasonal issues from 2018 to 2026.
- TNAU *Uzhavarin Valarum Velanmai*: monthly issues from 2022 to 2025.
- ICAR-IIMR Hindi millet books.
- VNMKV leaflets: 13 load, and about 19 more links currently return 404.

## Gaps

- **Bengali irrigation.** Only 2 documents cover it. Searches for Bengali irrigation material from BCKV and UBKV returned English research papers. A Bengali source on সেচ is still needed, for example from Matir Katha. Its site was unreachable from here on 2026-11-18.
- **Hindi irrigation.** The drip and sprinkler pages are short (296 and 143 words).
- **Fertilizer and harvest coverage** is thin in Tamil and Bengali.
