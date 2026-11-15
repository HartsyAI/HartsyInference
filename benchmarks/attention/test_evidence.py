import copy
import pathlib
import tempfile
import unittest
from fixtures import create, load, logical_values
from statistics_protocol import analyze
from telemetry_check import compare
from cell_fixture import materialize


class FixtureTests(unittest.TestCase):
    def test_all_storage_types_are_shared_exactly_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dtype in ('f32', 'f16', 'bf16'):
                path = create(pathlib.Path(temporary) / dtype, dtype, heads=2, kv_heads=1, sq=2, skv=3, dim=4, dtype=dtype)
                document = load(path)
                self.assertEqual(16, len(logical_values(path.parent, document['tensors']['q'])))
                (path.parent / 'q.bin').write_bytes(b'corrupted')
                with self.assertRaises(ValueError):
                    load(path)

    def test_fixture_rejects_escape_and_invalid_head_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                create(pathlib.Path(temporary) / 'invalid', 'bad', heads=3, kv_heads=2)

    def test_strided_and_broadcast_metadata_retain_exact_storage_hashes(self):
        cell = {'id': 'strided-mask', 'api': 'sage', 'cohort': 'test', 'parameters': dict(
            batch=2, heads=2, kv_heads=1, sq=2, skv=3, dim=4, dtype='f16', precision='int8',
            layout='strided', mask_dtype='bool', mask_broadcast='query')}
        with tempfile.TemporaryDirectory() as temporary:
            path = materialize(pathlib.Path(temporary) / 'cell', cell)
            document = load(path)
            self.assertEqual([16, 4, 8, 1], document['tensors']['q']['strides'])
            mask = logical_values(path.parent, document['tensors']['bias'])
            self.assertFalse(any(mask[:3]))
            self.assertEqual(32, len(logical_values(path.parent, document['tensors']['q'])))


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.pair = {'candidate_ms': 90, 'reference_ms': 100, 'valid': True}
        self.sessions = [[dict(self.pair) for _ in range(5)] for _ in range(3)]
        self.confirmation = [dict(self.pair) for _ in range(5)]

    def test_improvement_requires_separate_confirmation(self):
        result = analyze(self.sessions, self.confirmation, 'improvement')
        self.assertTrue(result['passed'])
        self.confirmation[0]['candidate_ms'] = 1000
        self.assertFalse(analyze(self.sessions, self.confirmation, 'improvement')['passed'])

    def test_fail_closed_on_missing_pairs_or_quality_failure(self):
        with self.assertRaises(ValueError):
            analyze(self.sessions[:2], self.confirmation, 'improvement')
        self.sessions[0][0]['valid'] = False
        with self.assertRaises(ValueError):
            analyze(self.sessions, self.confirmation, 'noninferiority')

    def test_regression_has_a_confidence_requirement(self):
        for session in self.sessions + [self.confirmation]:
            for pair in session:
                pair['candidate_ms'] = 110
        self.assertTrue(analyze(self.sessions, self.confirmation, 'regression')['passed'])
        self.assertFalse(analyze(self.sessions, self.confirmation, 'noninferiority')['passed'])


class TelemetryTests(unittest.TestCase):
    def test_output_or_route_divergence_invalidates_attribution(self):
        base = {'quality_passed': True, 'milliseconds': [1], 'fixture_sha256': 'fixture',
                'output_sha256': 'output', 'branch': 'route', 'core_sha256': 'B0',
                'backend_sha256': 'backend', 'telemetry': 'disabled'}
        disabled = dict(base, core_sha256='diagnostic')
        enabled = dict(disabled, telemetry='enabled')
        observations = [{'Implementation': 'executed'}]
        self.assertTrue(compare(base, disabled, enabled, observations)['exact_output_identity'])
        for field in ('output_sha256', 'branch', 'fixture_sha256'):
            changed = dict(enabled, **{field: 'changed'})
            with self.assertRaises(ValueError): compare(base, disabled, changed, observations)


if __name__ == '__main__':
    unittest.main()
