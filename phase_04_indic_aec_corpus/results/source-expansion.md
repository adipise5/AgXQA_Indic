# Indic-AEC source expansion

The Hindi pilot is the baseline. The expanded overlap audit scanned 14 extracted documents across 1,960 Phase 6 windows and found no hits at the configured threshold. The [measured results](source-expansion.json) include counts and hashes. The source registry adds Marathi, Bengali and
Tamil candidates with document-level origin and reuse references. The new
entries are candidates until their extracted text passes review.

| Language | Source | Scope and decision |
| --- | --- | --- |
| Marathi | Vigyan Ashram farming and animal husbandry handbook | Practical training. Visual spot check on PDF page 8 shows a readable scan with a flow chart. Verify that OCR preserves reading order and quantities. |
| Marathi | कृषी शब्दधन | Agricultural terminology. Assess as supplementary vocabulary material before counting it as extension prose. |
| Bengali | BOU irrigation unit BAE 3202-01 | The body of rendered PDF page 1 shows scrambled glyphs, while headings are legible. Obtain a corrected copy or reject after checking additional pages. |
| Bengali | BOU fibre-crop unit BAE 2301-06 | Source PDF page 3 is readable, with a three-column classification table. Verify the table and accompanying prose in extracted text. |
| Bengali | BOU soil science unit BAE 1204-01 | Separate course family. Downloaded and OCR processed. PDF page 2 is readable, with mixed Bengali and English prose and sidebars. Verify reading order in the extracted text. |
| Tamil | CAG traditional seed booklet | CC BY-SA 4.0. The publisher has Tamil and English editions. Original-language provenance remains unverified. |
| Tamil | Tamil Wikibooks mushroom chapter | CC BY-SA site terms. Verify the chapter's original-language provenance and any upstream source before using it. |

The BOU materials cover agriculture in Bangladesh. Report their region and
course-material genre separately from Indian extension documents.

A spot check of Hindi article AEC_HI_KHETI_1856 found a dropped decorative
initial, stray characters before the abstract, and text affected by column
layout. These spot checks do not constitute whole-document quality reviews.
The release requires matching review decisions for the exact text hashes.

## Additional source checks

- `AEC_MR_JMS_001` is a Marathi guide to community watershed development,
  published by Jagrut Mahila Samaj. Its repository record confirms the
  rightsholder's CC BY-SA 4.0 permission. It is independent of the Vigyan
  Ashram handbook. The direct PDF is identified, but the local environment
  could not resolve the download host, so the guide still needs acquisition,
  extraction and screening.
- `AEC_TA_TDL_001` is a Tamil Nadu Textbook Society irrigation book listed
  in the Tamil Digital Library. The catalog records Tamil as the language
  and open access as the access status. Corpus reuse rights and original
  language provenance remain pending.
- The earlier CAG and Tamil Wikibooks candidates remain on hold for their
  original-language evidence. The CAG project page confirms its team
  developed the booklet but does not establish which language came first.

## Indian government book search

- **Bengali.** Netaji Subhas Open University's organic agriculture and
  horticulture diploma book has relevant Bengali chapters. The university's
  OER manual covers its OER repository. The exact diploma PDF is in the
  self-learning materials section, so the OER declaration has not been
  established for this edition.
- **Tamil.** The Tamil Nadu textbook society irrigation book and the Class XI
  agricultural science water-management chapter are listed by NDLI as open
  access. Reuse terms for the editions are pending. Tamil University's
  encyclopedia has a relevant Tamil entry. The [Tamil Nadu policy note](https://it.tn.gov.in/sites/default/files/2019-07/Policy_Note_2019_2020_English.pdf)
  confirms the Creative Commons release order for Tamil University
  publications, and the source volume lists CC BY-SA 4.0. The signed Tamil
  entry is pinned to revision 1933800. The original HTTP acquisition failed
  at DNS lookup in the local environment. A later browser capture of six
  crop-related paragraphs yielded a 300-word excerpt and zero overlaps at
  the configured threshold across 1,960 Phase 6 reference windows. The
  excerpt is a pre-review candidate.
- **Marathi.** YCMOU's e-book catalogue has a School of Agricultural Sciences
  filter. The draft OER policy does not identify which individual farming
  books are covered. The catalogue itself is a discovery lead.
- **Hindi.** MANAGE hosts a Hindi agricultural extension diploma book. The
  PDF includes credits to outside works, so its book-level and third-party
  reuse terms need confirmation. ICAR Kheti remains a journal pilot.

The Bangladesh Open University units are outside the India-only scope.

The Maharashtra Department of Agriculture and Marketing's 2014
[farmer handbook](https://krishi.maharashtra.gov.in/Site/Upload/Pdf/Publications/Farmer_Friendly_Handbook_2014.pdf)
is an additional Marathi candidate. Reuse and original-language evidence
remain pending. MANAGE's [natural-farming series](https://www.manage.gov.in/NaturalFarming/TranslatedMaterial.aspx)
lists the regional editions as translations. Keep them outside the native
text corpus and in one family if a separate translated corpus is planned.

## Step 2 source verification

The table records the published evidence and what each candidate needs before
acquisition for the corpus. Edition-level terms apply to the specific material
they cover.

| Language | Source evidence | Decision |
| --- | --- | --- |
| Hindi | The cover of the [organic farming training booklet](https://nconf.dac.gov.in/uploads/books_manual/One_day_farmer_training_Booklet_Hindi.pdf) identifies the Government of India and its Regional Centre of Organic Farming, Ghaziabad. | `AEC_HI_NCOF_001` is a direct book lead. Confirm reuse terms and whether the text was originally written in Hindi. |
| Hindi | The [Krishi Sakhi training module](https://nconf.dac.gov.in/uploads/books_manual/Awareness-Module-Krishi-Sakhis-hindi.pdf) credits two Hindi translators on the imprint. | `AEC_HI_NCONF_001` is a documented translation and is ineligible for the native-language corpus. |
| Marathi | The [Maharashtra Marathi Language Department charter](https://marathi.gov.in/wp-content/uploads/2023/04/sanad2023.pdf) identifies the state board and its published encyclopedia volumes. The online [irrigation](https://vishwakosh.marathi.gov.in/25762/) and [farming systems](https://vishwakosh.marathi.gov.in/34088/) chapters are publicly readable. The board reserves rights in the footer. | `AEC_MR_MV_001` and `AEC_MR_MV_002` share one encyclopedia family. Hold both for explicit permission covering the intended reuse. |
| Marathi | [YCMOU's CC BY-SA 4.0 prospectus](https://ycmou.ac.in/wp-content/uploads/2026/05/Agri_Prosp_-31-May-2024.pdf) identifies Marathi-medium courses, including AGR-201 on soils and AGR-204 on crop production. The [university's draft OER policy](https://ycmou.ac.in/media/post_image/a5ded0aaeb432a685e8e1e159cfac88e.pdf) describes release through its OER repository. | Locate the exact AGR-201 and AGR-204 books and each edition's reuse terms. The prospectus supplies course codes and its own licence. Both courses are part of the same agricultural foundation programme. |
| Bengali | [NSOU's vocational catalogue](https://www.wbnsou.ac.in/online_services/slm_VOCATIONAL.shtml) links the exact diploma PDF for `AEC_BN_NSOU_001`. No agriculture listing was found on the [university OER repository index](https://www.wbnsou.ac.in/OER_Repository/repository.shtml). | Confirm rights for the diploma PDF. The indexed Bengali text layer is garbled, so acquisition would also require a visual check and Bengali OCR. |
| Bengali | [BCKV's publications page](https://www.bckv.edu.in/index.php/en/bengal-aromatic-publication-en) lists Bengali rice production books for sale. The [entomology department](https://www.bckv.edu.in/index.php/en/agricultural-entomology-student-activities-en) lists two Bengali beekeeping manuals. | Discovery leads. A freely available full-text edition and reuse terms are still needed before adding a book to the acquisition registry. |
| Tamil | The [2009 agricultural science volume](https://tamildigitallibrary.in/Articles/நூல்-228-அறிவியல்%20தொழில்நுட்பம்) from the Tamil Development Academy has a copyright notice in the [edition's imprint](https://tamildigitallibrary.in/assets/docs/uploads/primary_files/book/TVA_BOK_0000228_அறிவியல்_தொழில்நுட்பம்_தொகுதி_-_3.pdf). The Academy's [separate licence grant](https://commons.wikimedia.org/wiki/File:கலைக்களஞ்சியம்_1.pdf) identifies its two encyclopedia sets. | The agricultural science volume needs its own reuse evidence and confirmation of its publisher's government status under the project scope. The [Tamil Nadu Textbook Society agriculture catalogue](https://ndl.iitkgp.ac.in/he_document/tamilinaiyam_dl_he/tamil_digital_library/900001) remains a lead for a second family. |

The acquisition dry run found 1 ready row, 9 rows pending reuse permission,
and 26 rows outside the Indian government book scope. A separate Phase 6
URL and title check over the ten in-scope rows found no reserved URLs and one
title collision. `AEC_TA_TNTBSC_001` has the same title (நீர் நிர்வாகம்) as
reserved Vikaspedia document `TA_DOC_003`. The two records point to different
publishers and URLs, so the title match alone cannot establish text overlap.
Keep this textbook chapter on hold pending both reuse evidence and a
comparison of the underlying text. The four added registry rows had no
reserved URL or similar title. Passage-level screening requires the acquired
text.

The acquisition target is two independent eligible families per language.
The Tamil University encyclopedia currently supplies one acquisition-ready
family. Hindi, Marathi and Bengali each still need two, and Tamil needs one
more. Human text-quality review remains deferred.

## Step 3 acquisition and screening

Tamil University publishes the social-science encyclopedia and science
encyclopedia as distinct series. The [publisher catalogue](https://www.tamiluniversity.ac.in/english/publications-2/books/களஞ்சியம்/)
lists both, and [Tamil Nadu Government Order 105](https://commons.wikimedia.org/wiki/File:GoTN_Tamil_Development_Departments_order_on_creative_commons_cc_by_sa.pdf)
covers Tamil University publications. The [land-fragmentation chapter](https://ta.wikisource.org/w/index.php?title=வாழ்வியற்_களஞ்சியம்_4/உட்பிரிவும்_நிலச்_சிதறல்களும்&oldid=1948855)
comes from volume 4 of the social-science series, with scan PDF pages
763–767 linked from the transcription. The transcription is under
CC BY-SA 4.0.

The local downloader could not resolve the source host. The Wikisource
export reached an automated verification page. The acquired snapshot uses
the 19 normalized prose paragraphs from the pinned transcription, checked
against fingerprints of all 19 corresponding rendered paragraphs. It
omits both tables. The chapter discusses historical data and land policy.
The prepared text should be read in that context when human review takes
place.

| Document | Family | Extracted words | Clean text SHA-256 |
| --- | --- | ---: | --- |
| `AEC_TA_TU_001` | Tamil University science encyclopedia | 300 | `44b90cd350c715dcff86e3a8f9a15868ef56b4eabcb06daf96a81463b6c55f70` |
| `AEC_TA_TU_SOC_001` | Tamil University social-science encyclopedia | 1,196 | `435643b8781a249beb45034b210e6d846d4ae58a03c72f93b1d1c1aedb807075` |

The pipeline retained both extracts in the local pre-review packet under
`outputs/phase_04_indic_aec_corpus/india_book_acquisition/step_03_acquisition`.
It found zero hits at the configured threshold across 827 QA contexts and
1,133 full-document windows. The Phase 6 reference loader verified all
2,351 released QA passage hashes. Four reconstructed full-source snapshots
differed from the original raw hashes. Both Tamil review decisions remain
pending.

- **Hindi.** The [NCONF site](https://nconf.dac.gov.in/) and the
  indexed text of the Ghaziabad training booklet did not establish a reuse
  grant for that edition. Keep the booklet pending.
- **Marathi.** The [Department of Agriculture site terms](https://krishi.maharashtra.gov.in/Termandcondition.aspx)
  describe ownership and reserve rights. The 2014 farmer handbook still
  needs an edition-specific grant. The YCMOU agricultural foundation books
  still need verifiable download and licence records.
- **Bengali.** [NSOU's OER policy](https://www.wbnsou.ac.in/about_us/20170325_OER_Policy_NSOU_2017.pdf)
  covers the School of Vocational Studies, describes the credit-page
  statement for CC-licensed materials, and provides for courses to remain
  outside the CC release. The exact diploma edition still needs a matching
  credit-page statement or equivalent grant.

Tamil now has two pre-review families. Hindi, Marathi and Bengali still
need two eligible families each before the four-language corpus can be
built.

## Step 4 Marathi, Bengali and Hindi expansion

The pre-review packet now covers six families across Marathi, Bengali and
Tamil. This pass used historical agricultural economics and irrigation
prose. It does not establish present-day extension guidance or complete
the ten-document-per-language pilot target.

| Document | Book family and captured material | Extracted words |
| --- | --- | ---: |
| `AEC_MR_PHULE_ASUD_001` | Jyotirao Phule, [शेतकऱ्याचा असूड, chapter 2 excerpt](https://mr.wikisource.org/w/index.php?title=शेतकऱ्याचा_असूड/पान_४&oldid=149459). Selected passages on land subdivision, farm labour, land assessment and water sharing. | 367 |
| `AEC_MR_PHULE_ISHARA_001` | Jyotirao Phule, [इशारा, irrigation excerpt](https://sahitya.marathi.gov.in/ebooks/महात्मा%20फुले%20समग्र%20वाङ्मय.pdf). Three passages on the role of water in farming, drawn from the indexed scan text. OCR needs comparison with the original scan during human review. | 116 |
| `AEC_BN_VB_COOP_001` | Rabindranath Tagore, [সমবায়নীতি](https://bn.wikisource.org/w/index.php?title=সমবায়নীতি/সমবায়নীতি&oldid=1627557). Twelve selected paragraphs on cooperation and agrarian livelihoods from the 1953 Visva-Bharati edition of prose issued as a booklet in 1929. | 1,325 |
| `AEC_BN_VB_PALLI_001` | Rabindranath Tagore, পল্লীপ্রকৃতি: ভূমিলক্ষ্মী. Prose across both pages of the [West Bengal government-supported transcription](https://rabindra-rachanabali.nltr.org/node/7359) of the underlying essay in the [Visva-Bharati edition](https://bn.wikisource.org/wiki/চিত্র:পল্লীপ্রকৃতি_-_রবীন্দ্রনাথ_ঠাকুর.pdf). | 975 |

The Marathi passages use Phule's original text from two separately authored
works reprinted by the Maharashtra State Board for Literature and Culture.
The [Savitribai Phule Pune University bibliography](https://www.unipune.ac.in/chairs/mahatma_phule_chair/mphule_webfiles/pdf/Collected_Resource.pdf)
dates the first two chapters of शेतकऱ्याचा असूड to an 1883 serial
publication and the full collected text to 1967. The excerpt is limited to
chapter 2. The [state encyclopedia](https://vishwakosh.marathi.gov.in/27756/)
dates इशारा separately to 1885. The Bengali records are two independently
published books from [Visva-Bharati](https://www.indiacode.nic.in/bitstream/123456789/1652/3/A1951-29.pdf).
For both authors, the registry tracks the Indian public-domain basis for
the underlying prose. The captures exclude modern editorial content and
website material.

The six-document run found zero matches at the configured threshold
across 1,960 Phase 6 windows. All six entries are pending human review.
The local packet is under
`outputs/phase_04_indic_aec_corpus/india_book_acquisition/step_04_expansion`.

Hindi remains at zero eligible families under the government-book scope.
The [VMOU watershed management book](https://assets.vmou.ac.in/DWSM01.pdf)
was added to the registry with reuse and original-language evidence
pending. Equivalent evidence is still needed for the Ghaziabad training
booklet and MANAGE diploma material. [India Water Portal's contributor
terms](https://www.indiawaterportal.org/terms-of-use) provide a licence for
contributor submissions. The provenance reviewed for its reproduced
government watershed material did not establish a rightsholder grant for
those chapters. The earlier Hindi ICAR Kheti journal remains a pilot
outside the book scope.

## Step 5 Hindi acquisition and four-language review packet

The Hindi candidates use Swami Sahajanand Saraswati's original prose in
volume 4 of the [Hindi Samay digital edition](https://hindisamay.com/vividh/swami-sahjanand-saraswati-rachnavali/Index-Sahajanand-Samagra.htm).
The index lists the two works separately. The volume is the shared digital
edition. The independently titled works define the two document families.
Hindi Samay is a [digital publishing project](https://hindisamay.com/contact.htm)
of [Mahatma Gandhi Antarrashtriya Hindi Vishwavidyalaya](https://www.indiacode.nic.in/bitstream/123456789/1987/1/A1997-03.pdf),
a central university established by Parliament.

| Document | Captured material | Extracted words |
| --- | --- | ---: |
| `AEC_HI_MGAHV_KHET_MAZDOOR_001` | [खेत-मजदूर](https://hindisamay.com/vividh/swami-sahjanand-saraswati-rachnavali/Sahjanand-vol-4/Rachnavali4-part10.htm), four paragraphs on farm labour, land access and historical employment counts. | 732 |
| `AEC_HI_MGAHV_JHARKHAND_KISAN_001` | [झारखंड के किसान](https://hindisamay.com/vividh/swami-sahjanand-saraswati-rachnavali/Sahjanand-vol-4/Rachnavali4-part12.htm), four paragraphs on customary forest and agricultural land use. | 629 |

The local preparation run uses [Hindi Samay's public reuse notice for
its digital collection](https://www.hindisamay.com/index2.htm) and credits
the author and portal in the capture manifest. Both captures reconstruct
selected paragraphs from the rendered pages and exclude editorial and
translated material. Their manifest hashes identify those local excerpts.
The [current portal notice](https://hindisamay.com/) describes securing
permission to make works available, and the chapter pages carry a
site-wide copyright footer. Confirm that the reuse notice covers
redistribution in a public dataset or training release before publishing
either excerpt. The registry does not claim public-domain status for
either work.

The combined run prepared eight candidates with two families apiece in
Hindi, Marathi, Bengali and Tamil. All eight passed the configured Phase 6
overlap checks across 1,960 windows and the internal duplicate check.
The Phase 6 reference loader verified 2,351 QA passage hashes. Four
full-source snapshots had changed bytes, so the full-document audit uses
reconstructed public copies for those sources.

The local packet is under
`outputs/phase_04_indic_aec_corpus/india_book_acquisition/step_05_hindi`.
On 25 November 2026, the repository owner confirmed all eight text reviews
in chat. The accepted decisions are recorded against the exact text hashes
in `quality_reviews.csv`.

### Reviewed pilot build

The local build produced version `6e462c895198136c` with eight documents,
5,640 words and 12,176 tokens using the configured XLM-R tokenizer.
The four training documents have 7,939 tokens and the four validation
documents have 4,237 tokens. Each split contains one document family per
language, and no family occurs in both splits. The build repeated the Phase 6
audit and found zero hits across 1,960 comparison windows. Review hashes
and all three JSONL file checksums were verified after the build.

For Phase 5 pilot preparation, the local inputs are `train.jsonl` and
`validation.jsonl` under
`outputs/phase_04_indic_aec_corpus/india_book_acquisition/step_05_hindi/releases/6e462c895198136c`.
`statistics.json` records the tokenizer revision, file checksums and
Phase 6 baseline. The ten-document-per-language pilot target remains open.
Public redistribution of the two Hindi excerpts still depends on
confirming the scope of the reuse notice.

## Step 6 technical agriculture expansion

Two Tamil University chapters were downloaded from pinned Wikisource revisions and extracted with the Phase 4 pipeline. The original volumes and transcriptions identify Tamil University and the CC BY-SA 4.0 release. Both chapters belong to the existing `TU_SCIENCE_ENCYCLOPEDIA` family, so they add technical material within that family. [Tamil University catalogue](https://www.tamiluniversity.ac.in/english/publications-2/books/களஞ்சியம்/) · [Government Order 105](https://commons.wikimedia.org/wiki/File:GoTN_Tamil_Development_Departments_order_on_creative_commons_cc_by_sa.pdf)

| Document | Subject | Printed pages | Extracted words | XLM-R tokens |
| --- | --- | ---: | ---: | ---: |
| `AEC_TA_TU_SOILS_001` | [Problem soils, volume 10, revision 1888830](https://ta.wikisource.org/w/index.php?title=அறிவியல்_களஞ்சியம்_10/சிக்கலான_மண்_வகைகள்&oldid=1888830) | 165–166 | 384 | 1,119 |
| `AEC_TA_TU_RICE_001` | [Rice cultivation, volume 14, revision 1939659](https://ta.wikisource.org/w/index.php?title=அறிவியல்_களஞ்சியம்_14/நெல்‌&oldid=1939659) | 176–188 | 5,263 | 13,555 |

The combined packet contains ten extracts across eight families. On 25 November 2026, the repository owner approved the two new chapters in chat. The decisions are recorded against the exact cleaned text in `quality_reviews.csv`, alongside the eight earlier approvals. The rice chapter contains an implausible growing temperature printed in Celsius. A scan spot check confirmed that the value is present in the printed edition. The approved extract preserves the source wording and includes historical pesticide and fertilizer recommendations. [Printed rice chapter, page 177](https://ta.wikisource.org/wiki/பக்கம்:அறிவியல்_களஞ்சியம்_14.pdf/199)

The Phase 6 audit scanned all ten extracts against 1,960 reference windows and returned zero matches at the configured threshold. The reference includes 827 verified QA contexts and 1,133 full-document windows from reconstructed public copies. Four raw source hashes still differ from their originals. A preliminary word 5-gram check against the 217 distinct passages in the hash-verified official English AgXQA validation and test splits also returned zero matches. This preliminary check covers literal English overlap. The final audit needs the retained multilingual Phase 2 evaluation passages and checks for translated copies.

The new manifests, extraction report and review packet are under `outputs/phase_04_indic_aec_corpus/india_book_acquisition/step_06_technical_agriculture/`. The packet retains the downloaded HTML checksums and cleaned text checksums. Token counts use XLM-R revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` without special tokens.

The reviewed local build is version `54ab970c4c13c4a9`, under `step_06_technical_agriculture/releases/`. It contains 11,287 words and 26,850 XLM-R tokens. Training contains four extracts and 7,939 tokens. Validation contains six extracts and 18,911 tokens. Both new chapters belong to the Tamil University science encyclopedia family, which the seeded build assigns to validation. Both splits cover all four languages. The training JSONL checksum matches version `6e462c895198136c`. Additional technical training coverage requires eligible material from families assigned to train.

- **Hindi.** The NITI Aayog natural farming training manual is now in the registry as pending. The publisher provides both Hindi and English editions, and its policy requires permission for reproduction. Confirm the original-language edition and any reuse grant before acquisition into the corpus. [Hindi manual](https://www.niti.gov.in/sites/default/files/2026-03/Training-manual-Hindi.pdf) · [Copyright policy](https://www.niti.gov.in/hi/copyright-policy)
- **Marathi.** The 1988 edition of Jayantkumar Patil's *जगाची शेती* is now in the registry as pending. Its preface identifies the author's original comparative account, and the imprint reserves rights to the publisher. [Publisher's edition](https://sahitya.marathi.gov.in/ebooks/जगाची%20शेती.pdf)
- **Bengali.** The 1916 *বাৎসরিক কৃষি সমাচার* scan is a historical lead. Confirm the imprint, source type and fit with the current books-only scope before adding it to the acquisition registry. For modern technical material, the NSOU organic agriculture diploma still needs an edition-level reuse statement. [1916 scan record](https://commons.wikimedia.org/wiki/File:বঙ্গীয়_কৃষিবিভাগের_বাৎসরিক_কৃষি_সমাচার,_১৩২১_সাল_(১৯১৬).pdf) · [NSOU diploma catalogue](https://www.wbnsou.ac.in/online_services/slm_VOCATIONAL.shtml)

## Step 7 technical training candidates

The new excerpts extend two families already assigned to training in version `54ab970c4c13c4a9`. Both excerpts document historical agriculture. On 25 November 2026 the repository owner approved the exact new extracts in chat. The decisions are recorded in `quality_reviews.csv`, alongside the ten earlier approvals.

| Document | Content and evidence | Words | XLM-R tokens |
| --- | --- | ---: | ---: |
| `AEC_MR_PHULE_ASUD_WATERSHED_001` | Jyotirao Phule's chapter 5 proposals for conserving runoff, repairing field bunds, and improving access to irrigation water. The capture uses the erosion-control passage and the following cluster of watershed and well proposals. It retains the Wikisource transcription and uses the state edition's indexed text to corroborate the passage. [Pinned transcription](https://mr.wikisource.org/w/index.php?title=शेतकऱ्याचा_असूड/पान_१९&oldid=149450) · [State publisher's edition](https://sahitya.marathi.gov.in/ebooks/महात्मा%20फुले%20समग्र%20वाङ्मय.pdf) | 241 | 606 |
| `AEC_TA_TU_SOC_AGRI_001` | The historical Chengalpattu district's terrain, rainfall, rivers, irrigation infrastructure, and crops. The two downloaded scan pages were checked against the transcription. The capture corrects transcription errors including the printed irrigation-well count of 55,689. The district boundaries and counts reflect the source period. [Printed page 850](https://ta.wikisource.org/w/index.php?title=பக்கம்:வாழ்வியற்_களஞ்சியம்_9.pdf/878&oldid=1960196) · [Printed page 851](https://ta.wikisource.org/w/index.php?title=பக்கம்:வாழ்வியற்_களஞ்சியம்_9.pdf/879&oldid=1960198) | 191 | 575 |

The local packet is under `outputs/phase_04_indic_aec_corpus/india_book_acquisition/step_07_technical_training_candidates/`. The manifest preserves checksums for the source HTML, the Tamil scan images, and the selected captures used by the extractor. The pipeline kept all twelve candidates, with zero duplicate or source exclusions and zero matches against 1,960 Phase 6 reference windows. The preliminary literal check against 217 distinct passages in the pinned official English validation and test data also returned zero matches. Cross-language comparison with the retained Phase 2 benchmark remains pending.

The reviewed local build `3f3481d6662a2b69` contains 12 extracts and 28,031 XLM-R tokens. Both additions entered training, which rose by 1,181 tokens to 9,120 across six extracts. Validation has six extracts and 18,911 tokens, with the same JSONL checksum as version `54ab970c4c13c4a9`. The release verification checked all twelve reviewed text hashes, all three release JSONL checksums, and zero document, family or source URL overlap between splits.

The Phase 5 pilot inputs are under `outputs/phase_05_domain_adaptation/mlm_inputs/3f3481d6662a2b69-len256/`. The pinned XLM-R tokenizer produced 39 training sequences and 77 validation sequences at a maximum length of 256. The MLM manifest binds the sequence files to the exact Phase 4 split checksums and tokenizer revision.

## Edition checks for the research submission

- The April 2013 Bengali organic agriculture and horticulture volume reserves copyright to Netaji Subhas Open University in its front matter and requires written permission for reproduction. Its registry entry remains pending and records the edition's explicit terms. [NSOU volume, front matter](https://www.wbnsou.ac.in/online_services/SLM/VOCATIONAL/Diploma_Agri_%26_Horti_Paper_1_%26_2.pdf)
- The Maharashtra Water and Land Management Institute lists native Marathi publications on crop, soil, and water management. Its website requires prior written permission for reproduction. Treat the catalogue as a discovery source until a qualifying edition and a reuse grant are available. [WALMI catalogue](https://walmi.org/publications/) · [Copyright policy](https://walmi.org/copyright-policy/)
- Hindi Samay's legacy page explicitly invites attributed reuse. The current portal describes obtaining permission to make works available on Hindi Samay. The metadata archive includes the source ledger. The two Hindi extracts and saved captures remain in the local reproduction archive while exact public redistribution terms are unresolved. [Legacy reuse notice](https://www.hindisamay.com/index2.htm) · [Current portal](https://hindisamay.com/)
