"""Synthetic regression tests; fixtures are not research corpus data."""
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase4_pipeline', ROOT / 'phase_04_indic_aec_corpus/pipeline.py')
p = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(p)
except ModuleNotFoundError:
    p = None

@unittest.skipIf(p is None, 'Install phase_04_indic_aec_corpus/requirements.txt')
class Phase4Tests(unittest.TestCase):
    def test_reserved_candidates_and_aliases(self):
        guard = p.Exclusions()
        rows = p.read_rows(p.P6 / 'sources/phase4_exclusion_list.csv')
        for row in rows:
            self.assertEqual(guard.check(row['source_url']), 'phase6_reserved_url')
        self.assertTrue(guard.check('https://hi.vikaspedia.in/new-page'))

    def test_missing_heldout_fails(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                p.load_heldout(d)

    def test_passage_inside_long_document(self):
        text = 'one two three four five six seven eight nine ten'
        docs = [{'document_id': 'new', 'text': 'extra ' * 100 + text}]
        findings = p.audit_documents(docs, [{'context_id': 'test', 'context': text}])
        self.assertEqual(findings[0]['document_id'], 'new')

    def test_distributed_overlap(self):
        ref = 'one two three four five six seven eight nine ten eleven twelve'
        docs = [{'document_id': 'a', 'text': 'one two three four five six seven'},
                {'document_id': 'b', 'text': 'six seven eight nine ten eleven twelve'}]
        findings = p.audit_documents(docs, [{'context_id': 'test', 'context': ref}])
        self.assertEqual({f['document_id'] for f in findings}, {'a', 'b'})

    def test_unrelated_text(self):
        self.assertEqual(p.audit_documents([{'document_id':'a','text':'red green blue orange black white'}],
            [{'context_id':'test','context':'one two three four five six'}]), [])

    def test_indic_marks_preserved(self):
        self.assertIn('सिंचन', p.common.norm_for_match('सिंचन!'))

    def test_build_writes_disjoint_validated_release(self):
        from unittest.mock import patch
        with TemporaryDirectory() as d:
            docs = []
            for number, text in enumerate(('alpha beta gamma delta epsilon zeta eta theta',
                                           'red blue green black white orange purple yellow')):
                docs.append({'document_id': f'AEC_HI_{number}', 'language':'hi',
                    'source_url': f'https://example.org/{number}', 'title':f'fixture {number}',
                    'license':'synthetic fixture', 'permission_status':'approved',
                    'permission_reference':'test fixture', 'language_origin':'native_verified',
                    'document_family_id':f'family{number}', 'quality_status':'accepted', 'text':text})
            p.common.write_jsonl(Path(d)/'documents.jsonl', docs)
            cfg={'work_dir':d,'report_dir':d,'phase6_contexts':d,'min_document_words':1,
                 'seed':42,'validation_fraction':0.5,'languages':['hi'],'phase6_baseline':'fixture'}
            with patch.object(p, 'load_heldout', return_value=[{'context_id':'held',
                    'context':'unrelated synthetic reference text only for tests'}]):
                result=p.build(cfg)
            release=Path(d)/'releases'/result['version']
            train=p.read_lines(release/'train.jsonl')
            validation=p.read_lines(release/'validation.jsonl')
            self.assertEqual(len(train),1)
            self.assertEqual(len(validation),1)
            self.assertNotEqual(train[0]['document_family_id'],validation[0]['document_family_id'])

    def test_targeted_extraction_keeps_only_unchanged_other_documents(self):
        from unittest.mock import patch

        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'new.html'
            raw.write_text('<p>Fixture</p>')
            selected = {
                'document_id': 'AEC_MR_NEW', 'language': 'mr',
                'file_path': str(raw), 'sha256': p.common.sha256_file(raw),
            }
            retained = {'document_id': 'AEC_HI_KEEP', 'sha256': 'same', 'text': 'retained'}
            stale = {'document_id': 'AEC_HI_STALE', 'sha256': 'old', 'text': 'discarded'}
            p.common.write_jsonl(root / 'manifest.jsonl', [
                selected, retained, {**stale, 'sha256': 'changed'},
            ])
            p.common.write_jsonl(root / 'documents.jsonl', [retained, stale])
            cfg = {'work_dir': directory, 'report_dir': directory}

            with patch.object(p.extractor, 'extract_html', return_value=[]):
                p.extract(cfg, selected_ids=[selected['document_id']])

            result = p.read_lines(root / 'documents.jsonl')
            self.assertEqual([row['document_id'] for row in result],
                             [retained['document_id'], selected['document_id']])
            self.assertEqual(result[0]['text'], 'retained')

    def test_phase4_ocr_language_override_preserves_legacy_default(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        ocr = Mock(return_value='recognized')
        with patch.dict('sys.modules', {'pytesseract': SimpleNamespace(image_to_string=ocr)}):
            p.extractor.ocr_image(object(), 'bn')
            self.assertEqual(ocr.call_args.kwargs['lang'], 'ben')
            p.extractor.ocr_image(object(), 'bn', tesseract_language='ben+eng')
            self.assertEqual(ocr.call_args.kwargs['lang'], 'ben+eng')

    def test_india_books_scope_rejects_foreign_and_nonbook_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / 'sources.csv'
            sources.write_text(
                'document_id,language,source_url,publication_format,publisher_country,'
                'government_publisher_reference,permission_status,permission_reference,'
                'language_origin,license,document_family_id\n'
                'AEC_BN_FOREIGN,bn,https://example.org/bn,book,BD,https://example.org/bd,'
                'approved,ref,native_verified,CC-BY,FOREIGN\n'
                'AEC_HI_ARTICLE,hi,https://example.org/hi,journal_article,IN,https://example.org/in,'
                'approved,ref,native_verified,CC-BY,ARTICLE\n'
                'AEC_TA_BOOK,ta,https://example.org/ta,book,IN,https://example.org/in,'
                'pending,,unknown,unverified,BOOK\n'
            )
            cfg = {'sources': str(sources), 'work_dir': directory, 'report_dir': directory,
                   'indian_government_books_only': True}
            result = p.acquire(cfg, dry_run=True)
            self.assertEqual(result, {'outside_indian_government_book_scope': 2, 'permission_pending': 1})
            p.common.write_jsonl(root / 'documents.jsonl', [
                {'document_id': 'AEC_BN_FOREIGN', 'language': 'bn',
                 'source_url': 'https://example.org/bn', 'text': 'Prior acquired source'},
            ])
            with self.assertRaisesRegex(ValueError, 'No extracted Indian government books'):
                p.build(cfg)

    def test_prepare_review_keeps_unreviewed_text_out_of_training_splits(self):
        from unittest.mock import patch

        with TemporaryDirectory() as directory:
            root = Path(directory)
            document = {
                'document_id': 'AEC_HI_REVIEW', 'language': 'hi',
                'source_url': 'https://example.org/book', 'title': 'Review fixture',
                'license': 'CC-BY', 'permission_status': 'approved',
                'permission_reference': 'fixture', 'language_origin': 'native_verified',
                'document_family_id': 'review_family',
                'text': 'fixture words for a candidate document awaiting review',
            }
            p.common.write_jsonl(root / 'documents.jsonl', [document])
            cfg = {'work_dir': directory, 'report_dir': directory, 'phase6_contexts': directory,
                   'languages': ['hi'], 'min_document_words': 1}
            with patch.object(p, 'audit', return_value=[]):
                result = p.build(cfg, prepare_review=True)
                with self.assertRaisesRegex(ValueError, 'No accepted documents'):
                    p.build(cfg)
            self.assertEqual(result['candidate_documents'], 1)
            self.assertEqual(result['languages_short_of_two_families'], ['hi'])
            saved = p.read_lines(root / 'pre_review/candidates.jsonl')
            self.assertEqual(saved[0]['quality_status'], 'pending')
            self.assertEqual(saved[0]['clean_text_sha256'], p.digest(document['text']))
            self.assertFalse((root / 'releases').exists())

    def test_no_permission_means_no_download(self):
        with TemporaryDirectory() as d:
            sources = Path(d) / 'sources.csv'
            sources.write_text('document_id,language,source_url,permission_status\n'
                               'AEC_HI_TEST,hi,https://example.org/document,pending\n')
            cfg={'sources':str(sources), 'work_dir':d,'report_dir':d}
            result=p.acquire(cfg, dry_run=True)
            self.assertEqual(result['permission_pending'],1)
            self.assertFalse((Path(d)/'manifest.jsonl').exists())

if __name__ == '__main__':
    unittest.main()
