# AI spot review of Phase 4 OCR

Reviewer type: AI (Codex). Each row compares one rendered source PDF page with the current extracted paragraphs for that page. The SHA-256 identifies the full extract containing that page. These findings do not record human review or approve a document for training.

| Document ID | PDF page | Extract SHA-256 | Result | Finding |
| --- | ---: | --- | --- | --- |
| `AEC_HI_KHETI_1856` | 1 | `15f200c26fec57b51746579b14a2037fcbb442fb0ec50fae6a571e338bdd5538` | needs_correction | Dropped decorative initial, spurious leading digits before the abstract, and an interrupted sentence continuation across columns. |
| `AEC_HI_KHETI_944` | 1 | `699fc84f64fc9544a83fa654f07ad0f9bc2c87d7dc3cf54fe99ec154bf3cc8b6` | needs_correction | The title is readable, but the OCR misreads author-footnote marks and inserts digits at the start of the abstract. |
| `AEC_MR_VA_001` | 8 | `443e5eb00cf0999269920e52f264e8cee4157abb21e48e975f7d39515cda0baf` | needs_correction | The seed-count and equal-group values survived in the body. The flow chart lost its ordering and corrupted the unit next to the coriander seed quantity. |
| `AEC_BN_BOU_3202_01` | 1 | `b5ffb995a2d4299d7bdc7341bf01c36d95bb8e2c36b33ca33dfbae6090718148` | replacement_needed | The source page already renders scrambled Bengali glyphs in the body. OCR output reproduces the corruption. Obtain a corrected copy or exclude this edition. |
| `AEC_BN_BOU_2301_06` | 3 | `7104b87cd45e8414eacabb578ffbf6df4a34ea02a7309704b13856cd828d4452` | needs_correction | English classification terms and family names become spurious numbers or marks, and the sidebar enters the body text. Check the table layout before acceptance. |
| `AEC_BN_BOU_1204_01` | 2 | `71bb0666de627639fe41231d1a3c932055651e4cd0b8b5d95eb90f3a4203d382` | needs_correction | English soil definitions and parenthetical terms become mostly gibberish, and the sidebars enter the main reading order. |

A targeted rerun on the soil-science page at 250 DPI recovered all four sampled English strings using Tesseract `ben+eng`. The `ben` run at the same DPI recovered none. This test covers one page. It does not verify table or sidebar order and did not change the corpus extraction.

## Bengali extraction rerun

Phase 4 now uses `ben+eng` when a Bengali PDF needs OCR. The two corrected
extraction runs yielded the following spot-check results. These hashes
supersede the older hashes in the table only for the two rerun documents.

| Document | Page | New extract SHA-256 | Observed result |
| --- | ---: | --- | --- |
| `AEC_BN_BOU_1204_01` | 2 | `62d64d932a3495d780e3163bf6fa45ad32430eca78275274a4c802426a9ff186` | All four sampled English strings survived in the cleaned paragraphs. |
| `AEC_BN_BOU_2301_06` | 3 | `5a2fb8b39ea9127ff780c4fdcb41f816fe83001cba822d426034211912fc6bc4` | Phloem, Cortex, Wood and Vascular survived. Corchorus was still missing; the table needs correction. |

Separately, a body-only crop of soil-science page 2 began with the soil
definition rather than the running header and preserved the same four
sampled strings. The crop was an experiment and is not applied by the
pipeline. The scrambled irrigation PDF still needs a clean replacement.

The refreshed corpus audit checked 14 documents against 1,960 Phase 6
windows and reported zero hits at the configured threshold. The refreshed
local review queue keeps all 14 rows pending.
