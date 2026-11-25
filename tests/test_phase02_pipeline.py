"""Synthetic pipeline invariants only: no neural/tokenizer/language model is loaded."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unicodedata import normalize

from phase_02_multilingual_benchmark import pipeline as p
from phase_02_multilingual_benchmark.gemini_audit import ALL_LANGUAGES, document_disjoint, validate_parallel
from phase_02_multilingual_benchmark.gemini_exports import repair_span
from phase_02_multilingual_benchmark.operations import assemble, statistics, validate_candidates
from shared.qa_release import load_qa_release, sha256


def source(identifier="toy", split="test", question="Where is the water?"):
    row = {"id":identifier,"original_id":identifier,"split":split,"context":"Water reaches roots.",
           "question":question,"answers":{"text":["roots"],"answer_start":[14]},
           "source_document_id":"fixture-document-"+split,"source_document_id_kind":"synthetic_fixture",
           "source_invalid_answer_indices":[]}
    row["source_row_sha256"] = p.digest(row)
    return row


def candidate(row, language="bn"):
    result=p.source_candidate(row)
    result.update(id=row["id"]+"_"+language,language=language,context="জল মূলে যায়।",question="জল কোথায় যায়?",
                  answers={"text":["মূলে"],"answer_start":[3]},translation_status="complete",
                  raw_record_sha256=p.digest([row["id"],language]),automated_issues=[])
    result["candidate_sha256"]=p.candidate_fingerprint(result)
    return result


def judgment(row, status="accept", reviewer="Synthetic reviewer A"):
    row["candidate_sha256"]=p.candidate_fingerprint(row)
    result={field:"" for field in p.REVIEW_FIELDS}
    result.update(question_id=row["original_id"],language=row["language"],split=row["split"],reviewer=reviewer,
                  status=status,reviewed_at="2026-11-18T10:00:00+05:30",raw_record_sha256=row["raw_record_sha256"])
    result["candidate_sha256"]=row["candidate_sha256"]
    result.update({field:"5" for field in p.RATING_FIELDS})
    return result


class ImportedSpanTests(unittest.TestCase):
    def test_boundary_and_internal_whitespace_preserve_exact_passage_offsets(self):
        context = "one  roots."
        self.assertEqual(repair_span(context, context, "  roots", 3), ("roots", 5, "nfc_mapped_offset"))
        self.assertEqual(repair_span(context, context, "one roots", 0),
                         ("one  roots", 0, "unique_whitespace_equivalent_span"))
        self.assertEqual(repair_span(context, context, "  ", 3)[2], "unresolved")

    def test_overlapping_whitespace_matches_are_ambiguous(self):
        context = "a  a  a"
        self.assertEqual(repair_span(context, context, "a a", -1)[2], "unresolved")

    def test_nfc_remaps_offset_to_the_supplied_occurrence(self):
        original = "e\u0301 roots roots"
        normalized = normalize("NFC", original)
        self.assertEqual(repair_span(original, normalized, "roots", 9), ("roots", 8, "nfc_mapped_offset"))

    def test_joiner_equivalence_preserves_the_passage_spelling(self):
        context = "अ\u200dॅप"
        self.assertEqual(repair_span(context, context, "अॅप", 0), (context, 0, "unique_joiner_equivalent_span"))
        ambiguous = f"{context} {context}"
        self.assertEqual(repair_span(ambiguous, ambiguous, "अॅप", 0)[2], "unresolved")

    def test_review_can_correct_a_span_with_the_imported_reference_count(self):
        row = candidate(source(), language="en")
        row.update(source_reference_count=2, required_reference_count=1,
                   answers={"text": ["মূলে"], "answer_start": [0]}, automated_issues=["answer_span_mismatch"])
        review = judgment(row, status="correct")
        review.update(corrected_answers_json=json.dumps({"text": ["মূলে"], "answer_start": [3]}),
                      comments="Select the existing answer span.")
        corrected, issues = p.apply_review(row, [review])
        self.assertFalse(issues)
        self.assertTrue(corrected["human_reviewed"])


class GeminiDatasetTests(unittest.TestCase):
    def test_parallel_coverage_rejects_missing_and_duplicate_rows(self):
        item = source()
        rows = [candidate(item, language) for language in ALL_LANGUAGES]
        validate_parallel(rows, [item])
        for invalid in (rows[:-1], rows + rows[:1], [dict(rows[0], split="train"), *rows[1:]]):
            with self.assertRaises(ValueError):
                validate_parallel(invalid, [item])

    def test_disjoint_subset_filters_shared_urls_and_makes_validation_from_train(self):
        test = dict(source("test"), references="https://example.org/shared", context="test")
        dropped = dict(source("drop", "train"), references="https://example.org/shared", context="train")
        validation = dict(source("val", "validation"), references="", context="test")
        train = [dict(source(identifier, "train"), references=f"https://example.org/{identifier}",
                      source_document_id=identifier, context=identifier) for identifier in ("a", "b")]
        cohort = [test, dropped, validation, *train]
        kept, excluded, reassigned = document_disjoint(cohort, cohort, seed=42)
        self.assertEqual({row["original_id"] for row in excluded}, {"drop", "val"})
        self.assertEqual({row["split"] for row in kept}, {"train", "validation", "test"})
        self.assertEqual(len(reassigned), 1)
        self.assertEqual(next(row for row in kept if row["id"] in reassigned)["official_split"], "train")

    def test_release_loader_rejects_changed_test_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            p.write_json(bundle / "manifest.json", {"dataset_version": "fixture"})
            hashes = {}
            for split in ("train", "validation", "test"):
                item = source(split, split)
                path = bundle / "span_checked" / f"{split}.jsonl"
                p.write_jsonl(path, [candidate(item, language) for language in ALL_LANGUAGES])
                hashes[split] = sha256(path)
            config = bundle / "config.json"
            p.write_json(config, {"dataset_version": "fixture", "bundle_dir": str(bundle),
                                  "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
                                  "primary_variant": "span_checked", "published_evaluation_variant": "final",
                                  "variant_split_sha256": {"span_checked": hashes}})
            experiment = {"dataset_contract": str(config)}
            _, identity = load_qa_release(experiment)
            self.assertEqual(identity["split_sha256"]["test"], hashes["test"])
            with (bundle / "span_checked/test.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "test checksum"):
                load_qa_release(experiment)


class AlignmentTests(unittest.TestCase):
    def test_unique_alignment_and_unicode_offsets(self):
        context="জল মূলে যায়।"
        answers,alignment,issues=p.align_answers(context,["  মূলে  "])
        self.assertEqual(answers,[{"text":"মূলে","answer_start":3}])
        self.assertFalse(issues)
        self.assertEqual(alignment[0]["source_answer_index"],0)
        self.assertTrue(p.span_valid(context,"মূলে",3))
        self.assertFalse(p.span_valid(context,"মূলে",True))

    def test_ambiguous_or_missing_answers_are_not_injected(self):
        context="root root"
        answers,alignment,issues=p.align_answers(context,["root","absent"])
        self.assertEqual(answers,[])
        self.assertEqual(alignment[0]["candidate_starts"],[0,5])
        self.assertEqual(set(issues),{"answer_ambiguous_multiple_matches","answer_not_found"})
        self.assertEqual(context,"root root")

    def test_numeric_and_unit_heuristics(self):
        self.assertEqual(p.numeric_values("1,500 litres, 25mm and 3.5%"),p.numeric_values("১,৫০০ লিটার, ২৫মিমি এবং ৩.৫%"))
        self.assertEqual(p.units("25mm and 3.5%"),{"millimetre","percent"})
        self.assertNotEqual(p.numeric_values("25"),p.numeric_values("250"))

    def test_script_and_unicode_checks_have_hindi_marathi_limit(self):
        self.assertFalse(p.script_report("पाणी मुळांपर्यंत पोहोचते","hi")["issues"])
        self.assertFalse(p.script_report("पाणी मुळांपर्यंत पोहोचते","mr")["issues"])
        self.assertIn("unicode_corruption",p.script_report("জল\ufffd","bn")["issues"])
        self.assertIn("low_expected_script_ratio",p.script_report("Water reaches roots","ta")["issues"])

    def test_malformed_answers_fail_without_crashing(self):
        row=candidate(source())
        for answers in (None,[],{"text":None,"answer_start":None},{"text":["মূলে"],"answer_start":[True]}):
            changed={**row,"answers":answers}
            self.assertTrue(p.schema_issues(changed,require_answer=True))
            self.assertTrue(p.answer_issues(changed))

    def test_raw_retry_order_and_success_immutability(self):
        failed={"original_id":"toy","language":"bn","attempt":1,"translation_status":"failed"}
        success={**failed,"attempt":2,"translation_status":"complete"}
        self.assertEqual(p.latest_raw([failed,success]),[success])
        with self.assertRaises(ValueError):
            p.latest_raw([success,{**success,"attempt":3}])
        with self.assertRaises(ValueError):
            p.latest_raw([success,failed])


class ReviewTests(unittest.TestCase):
    def test_test_split_review_is_mandatory_in_all_languages(self):
        rows=[source("test"),source("train","train")]
        selected=p.review_sample_ids(rows,p.load_config())
        self.assertTrue(all(("test",language) in selected for language in p.LANGUAGES))
        self.assertEqual(selected,p.review_sample_ids(rows,p.load_config()))

    def test_unfilled_worksheets_do_not_count_as_human_reviews(self):
        row=candidate(source())
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"bengali_review.csv"
            with path.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=p.REVIEW_FIELDS)
                writer.writeheader();writer.writerow({"question_id":"toy","language":"bn"})
            self.assertFalse(p.read_reviews(temporary))
            review=judgment(row);review["semantic_equivalence"]="6"
            with path.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=p.REVIEW_FIELDS)
                writer.writeheader();writer.writerow(review)
            with self.assertRaisesRegex(ValueError,"1–5"):
                p.read_reviews(temporary)

    def test_hash_and_split_bind_review_to_original_raw_output(self):
        row=candidate(source())
        review=judgment(row);review["raw_record_sha256"]="0"*64
        self.assertEqual(p.apply_review(row,[review])[1],["review_raw_hash_or_split_mismatch"])

    def test_cleaned_candidate_changes_invalidate_prior_review(self):
        row=candidate(source());review=judgment(row)
        row["question"]="জল কেন যায়?"
        self.assertEqual(p.apply_review(row,[review])[1],["review_candidate_hash_mismatch"])

    def test_correction_requires_changes_and_valid_spans(self):
        row=candidate(source())
        review=judgment(row,"correct");review["comments"]="Synthetic correction rationale."
        self.assertIn("correct_decision_contains_no_changes",p.apply_review(row,[review])[1])
        review["corrected_answers_json"]=json.dumps({"text":["মূলে"],"answer_start":[4]})
        self.assertIn("answer_span_mismatch",p.apply_review(row,[review])[1])

    def test_flags_need_explicit_resolution_and_corrected_numbers_are_checked(self):
        row=candidate(source())
        row["automated_issues"]=["language_id_not_run"]
        review=judgment(row)
        self.assertTrue(p.apply_review(row,[review])[1])
        review.update(override_issues="language_id_not_run",comments="Synthetic native-language verification.")
        accepted,issues=p.apply_review(row,[review])
        self.assertFalse(issues);self.assertTrue(accepted["human_reviewed"])
        review.update(status="correct",corrected_context="জল মূলে যায়। 250")
        self.assertIn("context_numeric_values_changed",";".join(p.apply_review(row,[review],source())[1]))

    def test_reviewer_disagreement_requires_named_adjudication(self):
        row=candidate(source())
        first=judgment(row);second=judgment(row,"reject","Synthetic reviewer B")
        self.assertEqual(p.apply_review(row,[first,second])[1],["reviewer_disagreement_requires_adjudication"])
        third=judgment(row,reviewer="Synthetic adjudicator");third["adjudicator"]="Synthetic adjudicator"
        self.assertFalse(p.apply_review(row,[first,second,third])[1])

    def test_missing_translation_or_review_blocks_final_output(self):
        row=source()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"final"
            report=assemble([row],[p.source_candidate(row)],{},p.load_config(),output)
            self.assertEqual(report["status"],"blocked_pending_validation_or_review")
            self.assertFalse(output.exists())

    def test_complete_synthetic_group_writes_both_schemas_and_parallel_view(self):
        sources=[source("toy-"+split,split) for split in p.COUNTS]
        candidates=[item for row in sources for item in [p.source_candidate(row)]+[candidate(row,l) for l in p.LANGUAGES]]
        reviews={}
        # Supply language-specific fixture text so correction revalidation can check script.
        fixtures={"hi":("जल जड़ों तक जाता है।","जल कहाँ जाता है?","जड़ों"),
                  "mr":("पाणी मुळांपर्यंत पोहोचते.","पाणी कुठे पोहोचते?","मुळांपर्यंत"),
                  "bn":("জল মূলে যায়।","জল কোথায় যায়?","মূলে"),
                  "ta":("நீர் வேர்களை அடைகிறது.","நீர் எங்கு செல்கிறது?","வேர்களை")}
        for item in [c for c in candidates if c["language"] != "en"]:
            context,question,answer=fixtures[item["language"]]
            item.update(context=context,question=question,answers={"text":[answer],"answer_start":[context.index(answer)]})
            reviews[(item["original_id"],item["language"])]=[judgment(item)]
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"final"
            report=assemble(sources,candidates,reviews,p.load_config(),output)
            self.assertEqual(report["status"],"ready")
            self.assertEqual(len(p.read_jsonl(output/"test.jsonl")),5)
            self.assertEqual({r["id"] for r in p.read_jsonl(output/"shared/test.jsonl")},{"toy-test"})
            parallel=p.read_jsonl(output/"parallel_multilingual.jsonl")
            self.assertEqual(set(parallel[0]["languages"]),{"en",*p.LANGUAGES})
            with self.assertRaisesRegex(ValueError,"nonempty"):
                assemble(sources,candidates,reviews,p.load_config(),output)

    def test_unmeasured_ratings_and_agreement_are_null(self):
        row=source()
        report=statistics([row],[p.source_candidate(row)],{},p.load_config())
        self.assertIsNone(report["human_quality"]["bn"]["fluency"])
        self.assertIsNone(report["cohens_kappa"])
        self.assertEqual(report["observed"]["bn/test"]["translation_completion_rate"],0)
        self.assertIsNone(report["observed"]["bn/test"]["span_validity_rate"])

    def test_kappa_requires_explicit_independent_same_example_pairs(self):
        rows=[source("a"),source("b")]
        candidates=[candidate(row) for row in rows]
        reviews={(r["original_id"],"bn"):[judgment(r),judgment(r,"reject","Synthetic reviewer B")] for r in candidates}
        self.assertIsNone(statistics(rows,candidates,reviews,p.load_config())["cohens_kappa"])
        for entries in reviews.values():
            for review in entries:
                review["independent_review"]="true"
        reviews[("b","bn")][1]["status"]="accept"
        result=statistics(rows,candidates,reviews,p.load_config())["cohens_kappa"]
        self.assertEqual(result[0]["paired_examples"],2)
        self.assertEqual(result[0]["cohens_kappa"],0)


class CandidateValidationTests(unittest.TestCase):
    def test_split_mismatch_and_missing_coverage_are_reported(self):
        row=source()
        item=candidate(row);item["split"]="train"
        checks,coverage=validate_candidates([item],[row])
        self.assertIn("unknown_id_or_split_changed",checks[0]["issues"])
        self.assertEqual(coverage["missing_pair_count"],4)

    def test_schema_validator_reports_malformed_json_objects_and_arrays(self):
        row=source()
        checks,coverage=validate_candidates([[],None,{"original_id":[],"language":{},"answers":{"text":None},"automated_issues":None}],[row])
        self.assertTrue(all(not entry["schema_valid"] for entry in checks))
        self.assertEqual(coverage["missing_pair_count"],5)


if __name__ == "__main__":
    unittest.main()
