import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import timing_analysis_v4 as v4
from statistics_protocol import analyze

FIXTURE = '5439493c9d92847a3be8ae32dceb830e91c08e405f1bab3faaaecd3dea4b20a3'
TOOL = 'b' * 64
JOB = 'TEST-AA-v4'
EVIDENCE = Path('/home/hartsy/evidence/attention-4090/diag-20261115-c/v3')
ELIGIBILITY_REASONS = {'timing_eligible is not true', 'acceptance_reviewed is not true'}
SETTINGS = {'numerics.sageAttn': 'False', 'numerics.fa2Prefill': 'True', 'vram.mempoolKeep': 'True'}
KNOBS = 'numerics.sageAttn=request profile,numerics.sdpaCudnn=request profile'
NATIVE = {'/usr/lib/libcublasLt.so.13': 'c' * 64}
PTX = {'softmax_f16.ptx': 'd' * 64}
CORE = {'candidate': 'e' * 64, 'reference': 'f' * 64}
BACKEND = {'candidate': '1' * 64, 'reference': '2' * 64}


def samples_for(role, seed):
    rng = random.Random(seed)
    centre = 0.15 if role == 'candidate' else 0.148
    return [centre * (1 + rng.uniform(-0.05, 0.05)) for _ in range(200)]


def base_record(role, **changes):
    record = {
        'measurement_mode': 'acceptance', 'boundary': 'public', 'branch': 'S.15', 'route': 'Dense',
        'telemetry': 'disabled', 'cohort': 'linux-4090', 'device': 'NVIDIA GeForce RTX 4090',
        'arm': 'mat-f16-unasserted', 'legacy_asserted_bounded': False,
        'query_key_permissions': 'F16', 'prob_value_permissions': 'F16',
        'fixture_sha256': FIXTURE, 'tool_sha256': TOOL,
        'core_sha256': CORE[role], 'backend_sha256': BACKEND[role],
        'resolved_settings': dict(SETTINGS), 'knobs': KNOBS, 'native_libraries': dict(NATIVE),
        'ptx_sha256': dict(PTX), 'spirv_sha256': {},
        'quality_passed': True, 'timing_eligible': True, 'acceptance_reviewed': True,
        'milliseconds': samples_for(role, 1),
    }
    record.update(changes)
    return record


def records_for_layout(seed=1, **changes_for_every_process):
    """Synthetic eligible records for all 40 processes. Changes apply to every process."""
    records = {}
    for index, name in enumerate(v4.process_names()):
        role = 'candidate' if name.endswith('-cand') else 'reference'
        record = base_record(role, milliseconds=samples_for(role, seed * 1000 + index))
        record.update(changes_for_every_process)
        records[name] = record
    return records


def registration_document():
    candidate = base_record('candidate')
    reference = base_record('reference')
    identities = {'candidate': v4.identities_from_record(candidate), 'reference': v4.identities_from_record(reference)}
    expected = {key: candidate[key] for key in v4.REQUIRED_EXPECTED}
    return {
        'schema': 'timing-registration-v4', 'registration_id': 'TEST-v4', 'trials': 200, 'statistic': 'mean',
        'seed': 274019, 'resamples': 20000, 'thresholds': {'noninferiority': 1.03, 'regression': 1.03},
        'expected': expected,
        'campaigns': {
            'control': {'job_id': JOB + '-control', **identities},
            'campaign': {'job_id': JOB + '-campaign', **identities},
        },
    }


class Scratch:
    """A temporary directory holding a registration and campaign folders for one test."""

    def __init__(self, test):
        self._dir = tempfile.TemporaryDirectory()
        test.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.registration = self.root / 'registration.json'
        self.registration.write_text(json.dumps(registration_document(), indent=2), encoding='utf-8')

    def campaign(self, key, records, *, exit_codes=None, status=None, logs=None, job_id=None):
        folder = self.root / key
        folder.mkdir()
        job = job_id or registration_document()['campaigns'][key]['job_id']
        ledger = ['job start 2026-11-16T00:00:00-05:00']
        for name in v4.process_names():
            code = (exit_codes or {}).get(name, 0)
            if code is not None:
                ledger.append(f'{job} {name} {code} 1.0 2.0 load=0.0 0.0 0.0')
        (folder / 'timing-exits.txt').write_text('\n'.join(ledger) + '\n', encoding='utf-8')
        status_text = status if status is not None else (
            'job start 2026-11-16T00:00:00-05:00\nquiet wait passed 2026-11-16T00:00:30-05:00\n'
            'job rc=0 watch rc=0 2026-11-16T00:02:00-05:00\nquiet verify passed\nBATCH DONE 2026-11-16T00:02:00-05:00\n')
        (folder / 'status.txt').write_text(status_text, encoding='utf-8')
        for name, record in records.items():
            if record is None:
                continue
            (folder / f'{name}.json').write_text(json.dumps(record, allow_nan=True), encoding='utf-8')
            log = (logs or {}).get(name, 'harness started\nsample complete\n')
            (folder / f'{name}.log').write_text(log, encoding='utf-8')
        return folder

    def eligible_pair(self, control_records=None, campaign_records=None, **kwargs):
        control = self.campaign('control', control_records if control_records is not None else records_for_layout(1), **kwargs)
        campaign = self.campaign('campaign', campaign_records if campaign_records is not None else records_for_layout(2), **kwargs)
        return control, campaign

    def report(self, mode, control, campaign):
        return v4.build_report(self.registration, control, campaign, mode)


class RegistrationTests(unittest.TestCase):
    def test_rejects_seed_or_threshold_changes(self):
        for field, value in (('seed', 1), ('resamples', 2000), ('thresholds', {'noninferiority': 1.05, 'regression': 1.03})):
            with self.subTest(field=field):
                document = registration_document()
                document[field] = value
                scratch = Path(tempfile.mkdtemp())
                path = scratch / 'registration.json'
                path.write_text(json.dumps(document), encoding='utf-8')
                with self.assertRaises(v4.RegistrationError):
                    v4.load_registration(path)

    def test_rejects_missing_identity_fields(self):
        document = registration_document()
        del document['campaigns']['campaign']['candidate']['core_sha256']
        path = Path(tempfile.mkdtemp()) / 'registration.json'
        path.write_text(json.dumps(document), encoding='utf-8')
        with self.assertRaises(v4.RegistrationError):
            v4.load_registration(path)


class AcceptanceTests(unittest.TestCase):
    def test_eligible_campaigns_are_accepted_and_produce_results(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair()
        report = scratch.report('acceptance', control, campaign)
        self.assertEqual('acceptance', report['claim_status'])
        self.assertTrue(report['acceptance_eligible'])
        self.assertEqual([], report['refusals'])
        for key in ('control', 'campaign'):
            self.assertIn('noninferiority', report['results'][key]['claims'])
            self.assertIn('regression', report['results'][key]['claims'])

    def test_ineligible_records_are_refused_for_acceptance_and_remain_descriptive(self):
        scratch = Scratch(self)
        ineligible = records_for_layout(1, timing_eligible=False, acceptance_reviewed=False)
        control, campaign = scratch.eligible_pair(control_records=ineligible, campaign_records=records_for_layout(2, timing_eligible=False, acceptance_reviewed=False))
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual('refused', acceptance['claim_status'])
        self.assertFalse(acceptance['acceptance_eligible'])
        self.assertIsNone(acceptance['results'])
        self.assertEqual(80, len(acceptance['refusals']))
        for line in acceptance['refusals']:
            self.assertTrue(any(reason in line for reason in ELIGIBILITY_REASONS), line)

        descriptive = scratch.report('descriptive', control, campaign)
        self.assertEqual('descriptive only', descriptive['claim_status'])
        self.assertFalse(descriptive['acceptance_eligible'])
        self.assertIsNotNone(descriptive['results'])

    def test_acceptance_reviewed_false_alone_refuses_acceptance(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(control_records=records_for_layout(1, acceptance_reviewed=False))
        self.assertEqual('refused', scratch.report('acceptance', control, campaign)['claim_status'])
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_memory_event_in_a_process_log_refuses_acceptance_only(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(logs={'s2-p2-cand': '[WRN] [CudaMemory] OOM on async first attempt\n'})
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual('refused', acceptance['claim_status'])
        self.assertTrue(any('memory event' in line for line in acceptance['refusals']))
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_missing_quiet_verify_or_job_exit_refuses_acceptance_only(self):
        for status in ('job rc=0 watch rc=0\nBATCH DONE\n', 'quiet verify passed\nBATCH DONE\n'):
            with self.subTest(status=status):
                scratch = Scratch(self)
                control, campaign = scratch.eligible_pair(status=status)
                self.assertEqual('refused', scratch.report('acceptance', control, campaign)['claim_status'])
                self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])


class IntegrityRefusalTests(unittest.TestCase):
    """Each of these is refused in both modes."""

    def assert_refused_both(self, scratch, control, campaign, fragment):
        for mode in ('acceptance', 'descriptive'):
            with self.subTest(mode=mode):
                report = scratch.report(mode, control, campaign)
                self.assertEqual('refused', report['claim_status'])
                self.assertIsNone(report['results'])
                self.assertTrue(any(fragment in line for line in report['refusals']), report['refusals'][:3])

    def test_diagnostic_trace_is_refused(self):
        scratch = Scratch(self)
        records = records_for_layout(1)
        records['s1-p1-cand']['measurement_mode'] = 'diagnostic-trace'
        control, campaign = scratch.eligible_pair(control_records=records)
        self.assert_refused_both(scratch, control, campaign, 'diagnostic trace')

    def test_observe_boundary_is_refused(self):
        scratch = Scratch(self)
        records = records_for_layout(1, boundary='observe')
        control, campaign = scratch.eligible_pair(control_records=records)
        self.assert_refused_both(scratch, control, campaign, 'boundary is')

    def test_telemetry_enabled_is_refused(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(control_records=records_for_layout(1, telemetry='enabled'))
        self.assert_refused_both(scratch, control, campaign, 'telemetry is')

    def test_failed_quality_is_refused(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(campaign_records=records_for_layout(2, quality_passed=False))
        self.assert_refused_both(scratch, control, campaign, 'quality did not pass')

    def test_nonzero_missing_exit_status_is_refused(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(campaign_records=records_for_layout(2), exit_codes={'s3-p1-ref': 1})
        self.assert_refused_both(scratch, control, campaign, 'exit status 1')
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(campaign_records=records_for_layout(2), exit_codes={'s3-p1-ref': None})
        self.assert_refused_both(scratch, control, campaign, 'no exit status recorded')

    def test_wrong_identities_are_refused_for_each_field(self):
        changes = {
            'fixture_sha256': ('fixture is not the registered fixture', 'a' * 64),
            'tool_sha256': ('tool is not the registered tool', 'a' * 64),
            'branch': ('branch is', 'S.14'),
            'route': ('route is', 'Flash'),
            'arm': ('arm is', 'mat-f16'),
            'cohort': ('cohort is', 'linux-3060'),
            'device': ('device is', 'NVIDIA GeForce RTX 3060'),
            'core_sha256': ('core_sha256 is not the registered production library', 'a' * 64),
            'backend_sha256': ('backend_sha256 is not the registered production library', 'a' * 64),
            'resolved_settings': ('resolved_settings_sha256 differs', {'numerics.sageAttn': 'True'}),
            'knobs': ('knobs_sha256 differs', 'numerics.sageAttn=changed'),
            'native_libraries': ('native_libraries_sha256 differs', {'/usr/lib/libcublasLt.so.13': 'a' * 64}),
            'ptx_sha256': ('ptx_sha256 differs', {'softmax_f16.ptx': 'a' * 64}),
            'spirv_sha256': ('spirv_sha256 differs', {'extra.spv': 'a' * 64}),
        }
        for field, (fragment, value) in changes.items():
            with self.subTest(field=field):
                scratch = Scratch(self)
                control, campaign = scratch.eligible_pair(campaign_records=records_for_layout(2, **{field: value}))
                self.assert_refused_both(scratch, control, campaign, fragment)

    def test_a_single_process_with_a_wrong_role_identity_is_refused(self):
        scratch = Scratch(self)
        records = records_for_layout(2)
        records['s4-p5-cand']['core_sha256'] = 'b' * 64
        control, campaign = scratch.eligible_pair(campaign_records=records)
        self.assert_refused_both(scratch, control, campaign, 's4-p5-cand: core_sha256')

    def test_missing_fields_are_refused_in_both_modes(self):
        for field in ('measurement_mode', 'boundary', 'branch', 'route', 'telemetry', 'cohort', 'device', 'arm',
                      'legacy_asserted_bounded', 'query_key_permissions', 'prob_value_permissions', 'fixture_sha256',
                      'tool_sha256', 'core_sha256', 'backend_sha256', 'resolved_settings', 'knobs', 'native_libraries',
                      'ptx_sha256', 'spirv_sha256', 'quality_passed', 'milliseconds'):
            with self.subTest(field=field):
                records = records_for_layout(2)
                del records['s1-p2-ref'][field]
                scratch = Scratch(self)
                control, campaign = scratch.eligible_pair(campaign_records=records)
                for mode in ('acceptance', 'descriptive'):
                    self.assertEqual('refused', scratch.report(mode, control, campaign)['claim_status'], mode)

    def test_missing_review_or_eligibility_fields_refuse_acceptance_only(self):
        for field in ('timing_eligible', 'acceptance_reviewed'):
            with self.subTest(field=field):
                records = records_for_layout(2)
                del records['s1-p2-ref'][field]
                scratch = Scratch(self)
                control, campaign = scratch.eligible_pair(campaign_records=records)
                self.assertEqual('refused', scratch.report('acceptance', control, campaign)['claim_status'])
                self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_malformed_samples_are_refused(self):
        bad_lists = {
            'short': [0.15] * 199,
            'long': [0.15] * 201,
            'nan': [0.15] * 199 + [float('nan')],
            'zero': [0.15] * 199 + [0.0],
            'negative': [0.15] * 199 + [-0.15],
            'string': [0.15] * 199 + ['0.15'],
            'boolean': [0.15] * 199 + [True],
            'null': [0.15] * 199 + [None],
            'not_a_list': 'samples',
        }
        for label, samples in bad_lists.items():
            with self.subTest(samples=label):
                records = records_for_layout(2)
                records['s2-p1-cand']['milliseconds'] = samples
                scratch = Scratch(self)
                control, campaign = scratch.eligible_pair(campaign_records=records)
                self.assertEqual('refused', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_layout_must_be_complete(self):
        scratch = Scratch(self)
        records = records_for_layout(2)
        records['s4-p5-ref'] = None
        control, campaign = scratch.eligible_pair(campaign_records=records)
        self.assert_refused_both(scratch, control, campaign, 's4-p5-ref: process record is missing')

    def test_invalid_json_is_refused_not_crashed(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair()
        (campaign / 's2-p2-ref.json').write_text('{not json', encoding='utf-8')
        self.assert_refused_both(scratch, control, campaign, 'not valid JSON')


class OutputTests(unittest.TestCase):
    def test_output_is_write_once(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair()
        report = scratch.report('descriptive', control, campaign)
        target = scratch.root / 'derived' / 'report.json'
        target.parent.mkdir()
        v4.write_fresh(target, report, (control, campaign))
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            v4.write_fresh(target, report, (control, campaign))
        self.assertEqual(original, target.read_bytes())

    def test_output_inside_an_input_folder_is_refused(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair()
        report = scratch.report('descriptive', control, campaign)
        with self.assertRaises(v4.OutputRefused):
            v4.write_fresh(control / 'report.json', report, (control, campaign))
        self.assertFalse((control / 'report.json').exists())

    def test_command_line_returns_refusal_code_and_writes_a_report(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(control_records=records_for_layout(1, timing_eligible=False))
        out = scratch.root / 'out.json'
        code = v4.main(['--registration', str(scratch.registration), '--control-dir', str(control),
                        '--campaign-dir', str(campaign), '--mode', 'acceptance', '--out', str(out)])
        self.assertEqual(2, code)
        self.assertEqual('refused', json.loads(out.read_text(encoding='utf-8'))['claim_status'])


class MirrorTests(unittest.TestCase):
    def test_descriptive_mirror_matches_the_frozen_analysis_exactly(self):
        rng = random.Random(7)

        def session():
            return [{'candidate_ms': 0.15 * (1 + rng.uniform(-0.1, 0.1)),
                     'reference_ms': 0.148 * (1 + rng.uniform(-0.1, 0.1)), 'valid': True} for _ in range(5)]

        sessions = [session(), session(), session()]
        confirmation = session()
        for claim in v4.CLAIMS:
            with self.subTest(claim=claim):
                frozen = analyze(sessions, confirmation, claim)
                mirror = v4.descriptive_statistics(sessions, confirmation, claim)
                for key in ('ratio', 'upper95_one_sided', 'lower95_one_sided', 'confirmation_ratio', 'passed', 'claim'):
                    self.assertEqual(frozen[key], mirror[key], key)
                self.assertTrue(all(math.isfinite(mirror[key]) for key in ('ratio', 'upper95_one_sided')))

    def test_frozen_gate_still_rejects_pairs_without_a_true_valid_flag(self):
        sessions = [[{'candidate_ms': 1.0, 'reference_ms': 1.0, 'valid': False}] * 5] * 3
        with self.assertRaises(ValueError):
            analyze(sessions, [{'candidate_ms': 1.0, 'reference_ms': 1.0, 'valid': True}] * 5, 'noninferiority')


@unittest.skipUnless(EVIDENCE.exists(), 'the v3 evidence folder is not present on this machine')
class RealV31RecordTests(unittest.TestCase):
    """The stored v3.1 records. A registration is built from their own identities, so only eligibility can fail.
    This builds a retrospective descriptive registration for the test; it is not an acceptance registration."""

    def test_real_ineligible_records_are_rejected_for_acceptance_and_described_descriptively(self):
        control = EVIDENCE / 'aa-v3.1-mean200'
        campaign = EVIDENCE / 'p1-vs-b0-v3-mean200'
        control_cand = json.loads((control / 's1-p1-cand.json').read_text(encoding='utf-8'))
        control_ref = json.loads((control / 's1-p1-ref.json').read_text(encoding='utf-8'))
        campaign_cand = json.loads((campaign / 's1-p1-cand.json').read_text(encoding='utf-8'))
        campaign_ref = json.loads((campaign / 's1-p1-ref.json').read_text(encoding='utf-8'))
        expected = {key: control_cand[key] for key in v4.REQUIRED_EXPECTED}
        document = {
            'schema': 'timing-registration-v4', 'registration_id': 'RETRO-v3.1-descriptive', 'trials': 200,
            'statistic': 'mean', 'seed': 274019, 'resamples': 20000,
            'thresholds': {'noninferiority': 1.03, 'regression': 1.03}, 'expected': expected,
            'campaigns': {
                'control': {'job_id': 'AA-v3.1-mean200', 'candidate': v4.identities_from_record(control_cand),
                            'reference': v4.identities_from_record(control_ref)},
                'campaign': {'job_id': 'P1-vs-B0-v3-mean200', 'candidate': v4.identities_from_record(campaign_cand),
                             'reference': v4.identities_from_record(campaign_ref)},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            registration = Path(directory) / 'retro-registration.json'
            registration.write_text(json.dumps(document), encoding='utf-8')
            acceptance = v4.build_report(registration, control, campaign, 'acceptance')
            descriptive = v4.build_report(registration, control, campaign, 'descriptive')

        self.assertEqual('refused', acceptance['claim_status'])
        self.assertEqual(80, len(acceptance['refusals']))
        for line in acceptance['refusals']:
            reasons = set(line.split(': ', 1)[1].split('; '))
            self.assertEqual(ELIGIBILITY_REASONS, reasons, line)
        self.assertEqual('descriptive only', descriptive['claim_status'])
        stored = json.loads((EVIDENCE / 'analysis-v3.json').read_text(encoding='utf-8'))
        for key, campaign_key in (('control', 'control'), ('campaign', 'campaign')):
            for claim in ('noninferiority', 'regression'):
                with self.subTest(campaign=key, claim=claim):
                    ours = descriptive['results'][key]['claims'][claim]
                    theirs = stored[campaign_key]['gating_statistic'][claim]
                    self.assertAlmostEqual(theirs['ratio'], ours['ratio'], places=12)
                    self.assertAlmostEqual(theirs['upper95_one_sided'], ours['upper95_one_sided'], places=12)
                    self.assertAlmostEqual(theirs['lower95_one_sided'], ours['lower95_one_sided'], places=12)


if __name__ == '__main__':
    unittest.main()
