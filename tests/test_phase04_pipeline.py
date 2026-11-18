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
