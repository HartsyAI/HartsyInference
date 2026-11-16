import hashlib
import json
import pathlib
import tempfile
import unittest

import fixtures

PUBLISHED = pathlib.Path(__file__).parent / 'published-fixtures' / 'f16-mat' / 'fixture.json'
PINNED_SHA256 = '5439493c9d92847a3be8ae32dceb830e91c08e405f1bab3faaaecd3dea4b20a3'


class PublishedFixtureTests(unittest.TestCase):
    def test_manifest_bytes_are_the_pinned_bytes(self):
        self.assertEqual(PINNED_SHA256, hashlib.sha256(PUBLISHED.read_bytes()).hexdigest())

    def test_acceptance_stays_unreviewed_with_the_contract_tolerance(self):
        document = json.loads(PUBLISHED.read_text(encoding='utf-8'))
        self.assertFalse(document['acceptance']['reviewed'])
        self.assertEqual(0.003, document['acceptance']['rel_l2'])
        self.assertIsNone(document['acceptance']['mean_abs'])
        self.assertEqual('docs/ATTENTION_CONTRACT.md section 8', document['acceptance']['source'])
        self.assertFalse(document['call']['legacy_asserted_bounded'])

    def test_recorded_invocation_regenerates_the_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / 'f16-mat'
            path = fixtures.create(root, 'f16-mat', batch=1, heads=4, kv_heads=4, sq=256, skv=516, dim=64,
                                   dtype='f16', seed=1234, precision='f16', route='Dense', kv_dtype='f16',
                                   causal=False, query_offset=0, legacy_asserted_bounded=False)
            self.assertEqual(PUBLISHED.read_bytes(), path.read_bytes())
            regenerated = json.loads(path.read_text(encoding='utf-8'))
            for name in ('q', 'k', 'v'):
                digest = hashlib.sha256((root / regenerated['tensors'][name]['file']).read_bytes()).hexdigest()
                self.assertEqual(regenerated['tensors'][name]['sha256'], digest)


if __name__ == '__main__':
    unittest.main()
