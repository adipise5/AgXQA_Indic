# Indic Agricultural Extension Corpus

**Version `3f3481d6662a2b69` · 25 November 2026.** Indic-AEC is a curated corpus of original-language agricultural and rural-development excerpts for Phase 5 masked-language adaptation. The research submission covers this specific local pilot release. [Methods and verification](../../phase_04_indic_aec_corpus/README.md#research-submission) · [Machine-readable audit](../../phase_04_indic_aec_corpus/results/submission-audit.json)

## Composition

| Language | Train extracts | Train words | Train tokens | Validation extracts | Validation words | Validation tokens |
|---|---:|---:|---:|---:|---:|---:|
| Hindi | 1 | 629 | 946 | 1 | 732 | 1,003 |
| Marathi | 2 | 608 | 1,451 | 1 | 116 | 187 |
| Bengali | 1 | 1,325 | 2,898 | 1 | 975 | 2,140 |
| Tamil | 2 | 1,387 | 3,825 | 3 | 5,947 | 15,581 |
| **Total** | **6** | **3,949** | **9,120** | **6** | **7,770** | **18,911** |

The 12 excerpts represent eight document families and four Indian public-institution publishers. Labels cover agricultural economics, irrigation, crop management, pest management, and soil management. Tokens use `FacebookAI/xlm-roberta-base` at revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, excluding special tokens. [Per-source counts](../../phase_04_indic_aec_corpus/results/submission-audit.json)

## Collection and quality

The source registry records the edition, publisher, origin, family, and reuse evidence. Extraction preserves Unicode NFC and Indic combining marks. Selected excerpts may omit tables, non-agricultural material, or modern editorial content. All 12 exact cleaned-text hashes match an accepted decision by the repository owner. The release has one reviewer per excerpt. [Registry](../../phase_04_indic_aec_corpus/sources.csv) · [Review log](../../phase_04_indic_aec_corpus/quality_reviews.csv) · [Capture notes](../../phase_04_indic_aec_corpus/results/source-expansion.md)

The seed-42 build assigns each family to one split. Both splits contain all four languages. The submission audit found zero exact cross-family paragraph duplicates for paragraphs of at least 20 normalized words. The Phase 6 comparison checked 827 verified QA passages and 1,133 full-document windows using normalized word 5-grams and whitespace-insensitive character 10-grams. Both methods returned zero flags at containment thresholds 0.30, 0.40 and 0.50. Four reconstructed Phase 6 pages differ from their original HTML hashes. The frozen multilingual Phase 2 evaluation passages were unavailable. The completed checks measure literal and character-level similarity. [Audit methods](../../phase_04_indic_aec_corpus/README.md#research-submission)

## Intended use and limits

Use this release to exercise and document a controlled multilingual adaptation pipeline. Eight families and 9,120 training tokens provide limited evidence for broad agricultural or linguistic coverage. The ten-extract-per-language collection target remains unmet. Tamil makes up most of the validation text, and much of the material is historical. The rice extract preserves a documented implausible temperature printed in the source. The corpus is unsuitable as current farming guidance. [Source limitations](../../phase_04_indic_aec_corpus/results/source-expansion.md#step-6-technical-agriculture-expansion)

Source rights apply per record. The Tamil extracts carry CC BY-SA 4.0, and the Bengali and Marathi records retain the registry's India-specific public-domain designation for the underlying prose. The two Hindi extracts use the Hindi Samay notice for the local research release. Public redistribution of those cleaned extracts still requires confirmation of the applicable terms for the exact editions. The submission’s metadata archive omits corpus and held-out text. The separate reproduction archive contains those files for local use. [Hindi Samay reuse notice](https://www.hindisamay.com/index2.htm) · [Tamil Nadu release order](https://commons.wikimedia.org/wiki/File:GoTN_Tamil_Development_Departments_order_on_creative_commons_cc_by_sa.pdf)
