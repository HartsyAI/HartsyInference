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
CONTROL_JOB = JOB + '-control'
CAMPAIGN_JOB = JOB + '-campaign'
STAMP = '2026-11-16T00:00:00-05:00'
EVIDENCE = Path('/home/hartsy/evidence/attention-4090/diag-20261115-c/v3')
ELIGIBILITY_REASONS = {'timing_eligible is not true', 'acceptance_reviewed is not true'}
SETTINGS = {'numerics.sageAttn': 'False', 'numerics.fa2Prefill': 'True', 'vram.mempoolKeep': 'True'}
KNOBS = 'numerics.sageAttn=request profile,numerics.sdpaCudnn=request profile'
NATIVE = {'/usr/lib/libcublasLt.so.13': 'c' * 64}
PTX = {'softmax_f16.ptx': 'd' * 64}
CORE = {'candidate': 'e' * 64, 'reference': 'f' * 64}
BACKEND = {'candidate': '1' * 64, 'reference': '2' * 64}
# The status lines the batch script writes for a clean run, as in the stored v3.1 campaigns.
STATUS_OK = (f'lock acquired {STAMP}\nquiet wait passed {STAMP}\njob rc=0 watch rc=0 {STAMP}\n'
             f'quiet verify passed\nBATCH DONE {STAMP}\n')


def samples_for(role, seed):
    rng = random.Random(seed)
    centre = 0.15 if role == 'candidate' else 0.148
    return [centre * (1 + rng.uniform(-0.05, 0.05)) for _ in range(200)]


def base_record(role, *, identity_role=None, **changes):
    identity = identity_role or role
    record = {
        'measurement_mode': 'acceptance', 'boundary': 'public', 'branch': 'S.15', 'route': 'Dense',
        'telemetry': 'disabled', 'cohort': 'linux-4090', 'device': 'NVIDIA GeForce RTX 4090',
        'arm': 'mat-f16-unasserted', 'legacy_asserted_bounded': False,
        'query_key_permissions': 'F16', 'prob_value_permissions': 'F16',
        'fixture_sha256': FIXTURE, 'tool_sha256': TOOL,
        'core_sha256': CORE[identity], 'backend_sha256': BACKEND[identity],
        'resolved_settings': dict(SETTINGS), 'knobs': KNOBS, 'native_libraries': dict(NATIVE),
        'ptx_sha256': dict(PTX), 'spirv_sha256': {},
        'quality_passed': True, 'timing_eligible': True, 'acceptance_reviewed': True,
        'milliseconds': samples_for(role, 1),
    }
    record.update(changes)
    return record


def records_for_layout(seed=1, *, same_binary=False, **changes_for_every_process):
    """Synthetic eligible records for all 40 processes. Changes apply to every process. With same_binary, both roles
    bind the candidate identities, as an A/A control does."""
    records = {}
    for index, name in enumerate(v4.process_names()):
        role = 'candidate' if name.endswith('-cand') else 'reference'
        record = base_record(role, identity_role='candidate' if same_binary else role,
                             milliseconds=samples_for(role, seed * 1000 + index))
        record.update(changes_for_every_process)
        records[name] = record
    return records


def aa_records(seed=1, **changes_for_every_process):
    """Control-campaign records. The control runs one binary, so its two roles bind identical identities."""
    return records_for_layout(seed, same_binary=True, **changes_for_every_process)


def registration_document():
    candidate = base_record('candidate')
    reference = base_record('reference')
    control_identity = v4.identities_from_record(candidate)
    expected = {key: candidate[key] for key in v4.REQUIRED_EXPECTED}
    return {
        'schema': 'timing-registration-v4', 'registration_id': 'TEST-v4', 'trials': 200, 'statistic': 'mean',
        'seed': 274019, 'resamples': 20000, 'thresholds': {'noninferiority': 1.03, 'regression': 1.03},
        'expected': expected,
        'campaigns': {
            'control': {'job_id': CONTROL_JOB, 'candidate': control_identity, 'reference': dict(control_identity)},
            'campaign': {'job_id': CAMPAIGN_JOB, 'candidate': v4.identities_from_record(candidate),
                         'reference': v4.identities_from_record(reference)},
        },
    }


def default_ledger(job, exit_codes=None, extra=(), job_done=True):
    """The exit ledger of a clean run: a job start line, one line per process, then JOB DONE. Extra lines go before JOB DONE."""
    lines = [f'job start {STAMP}']
    for name in v4.process_names():
        code = (exit_codes or {}).get(name, 0)
        if code is not None:
            lines.append(f'{job} {name} {code} 1.0 2.0 load=0.0 0.0 0.0')
    lines.extend(extra)
    if job_done:
        lines.append(f'{job} JOB DONE {STAMP}')
    return lines


def campaign_reasons_of(report, key):
    """The campaign-level reasons of one campaign, split from its joined refusal line."""
    prefix = f'{key}: '
    reasons = []
    for line in report['refusals']:
        if line.startswith(prefix):
            reasons += line[len(prefix):].split('; ')
    return reasons


class Scratch:
    """A temporary directory holding a registration and campaign folders for one test."""

    def __init__(self, test):
        self._dir = tempfile.TemporaryDirectory()
        test.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)
        self.registration = self.root / 'registration.json'
        self.registration.write_text(json.dumps(registration_document(), indent=2), encoding='utf-8')

    def campaign(self, key, records, *, exit_codes=None, status=None, logs=None, job_id=None, ledger=None):
        folder = self.root / key
        folder.mkdir()
        job = job_id or registration_document()['campaigns'][key]['job_id']
        lines = ledger if ledger is not None else default_ledger(job, exit_codes)
        (folder / 'timing-exits.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (folder / 'status.txt').write_text(STATUS_OK if status is None else status, encoding='utf-8')
        for name, record in records.items():
            if record is None:
                continue
            (folder / f'{name}.json').write_text(json.dumps(record, allow_nan=True), encoding='utf-8')
            log = (logs or {}).get(name, 'harness started\nsample complete\n')
            (folder / f'{name}.log').write_text(log, encoding='utf-8')
        return folder

    def eligible_pair(self, control_records=None, campaign_records=None, **kwargs):
        control = self.campaign('control', control_records if control_records is not None else aa_records(1), **kwargs)
        campaign = self.campaign('campaign', campaign_records if campaign_records is not None else records_for_layout(2), **kwargs)
        return control, campaign

    def report(self, mode, control, campaign):
        return v4.build_report(self.registration, control, campaign, mode)


class RegistrationTests(unittest.TestCase):
    def write_registration(self, document):
        path = Path(tempfile.mkdtemp()) / 'registration.json'
        path.write_text(json.dumps(document), encoding='utf-8')
        return path

    def test_rejects_seed_or_threshold_changes(self):
        for field, value in (('seed', 1), ('resamples', 2000), ('thresholds', {'noninferiority': 1.05, 'regression': 1.03})):
            with self.subTest(field=field):
                document = registration_document()
                document[field] = value
                with self.assertRaises(v4.RegistrationError):
                    v4.load_registration(self.write_registration(document))

    def test_rejects_missing_identity_fields(self):
        document = registration_document()
        del document['campaigns']['campaign']['candidate']['core_sha256']
        with self.assertRaises(v4.RegistrationError):
            v4.load_registration(self.write_registration(document))

    def test_control_must_bind_identical_identities_in_every_field(self):
        for field in v4.REQUIRED_IDENTITY:
            with self.subTest(field=field):
                document = registration_document()
                document['campaigns']['control']['reference'] = dict(document['campaigns']['control']['reference'],
                                                                     **{field: 'a' * 64})
                with self.assertRaisesRegex(v4.RegistrationError, 'identical'):
                    v4.load_registration(self.write_registration(document))

    def test_campaign_may_bind_different_candidate_and_reference_identities(self):
        loaded = v4.load_registration(self.write_registration(registration_document()))
        campaign = loaded['campaigns']['campaign']
        self.assertNotEqual(campaign['candidate']['core_sha256'], campaign['reference']['core_sha256'])


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
        control, campaign = scratch.eligible_pair(control_records=aa_records(1, timing_eligible=False, acceptance_reviewed=False),
                                                  campaign_records=records_for_layout(2, timing_eligible=False, acceptance_reviewed=False))
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual('refused', acceptance['claim_status'])
        self.assertFalse(acceptance['acceptance_eligible'])
        self.assertIsNone(acceptance['results'])
        self.assertEqual(80, len(acceptance['refusals']))
        for line in acceptance['refusals']:
            self.assertTrue(any(reason in line for reason in ELIGIBILITY_REASONS), line)
        self.assertEqual([], campaign_reasons_of(acceptance, 'control'))
        self.assertEqual([], campaign_reasons_of(acceptance, 'campaign'))

        descriptive = scratch.report('descriptive', control, campaign)
        self.assertEqual('descriptive only', descriptive['claim_status'])
        self.assertFalse(descriptive['acceptance_eligible'])
        self.assertIsNotNone(descriptive['results'])

    def test_acceptance_reviewed_false_alone_refuses_acceptance(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(control_records=aa_records(1, acceptance_reviewed=False))
        self.assertEqual('refused', scratch.report('acceptance', control, campaign)['claim_status'])
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_memory_event_in_a_process_log_refuses_acceptance_only(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(logs={'s2-p2-cand': '[WRN] [CudaMemory] OOM on async first attempt\n'})
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual('refused', acceptance['claim_status'])
        self.assertTrue(any('memory event' in line for line in acceptance['refusals']))
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_missing_quiet_verify_or_job_line_refuses_acceptance_only(self):
        cases = {
            'no quiet verify': (f'job rc=0 watch rc=0 {STAMP}\nBATCH DONE {STAMP}\n', 'quiet verify passed 0 times'),
            'no job line': (f'quiet verify passed\nBATCH DONE {STAMP}\n', 'status has no job rc line'),
        }
        for label, (status, fragment) in cases.items():
            with self.subTest(case=label):
                scratch = Scratch(self)
                control = scratch.campaign('control', aa_records(1))
                campaign = scratch.campaign('campaign', records_for_layout(2), status=status)
                acceptance = scratch.report('acceptance', control, campaign)
                self.assertEqual('refused', acceptance['claim_status'])
                reasons = campaign_reasons_of(acceptance, 'campaign')
                self.assertEqual(1, len(reasons), reasons)
                self.assertIn(fragment, reasons[0])
                self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_failed_watcher_refuses_acceptance_even_with_a_passed_verify(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2), status=(
            f'lock acquired {STAMP}\nquiet wait passed {STAMP}\njob rc=0 watch rc=1 {STAMP}\n'
            f'quiet verify passed\nBATCH DONE {STAMP}\n'))
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual('refused', acceptance['claim_status'])
        self.assertEqual(['watch rc=1 (watcher evidence)'], campaign_reasons_of(acceptance, 'campaign'))
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_failed_job_refuses_acceptance_only(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2), status=(
            f'job rc=1 watch rc=0 {STAMP}\nquiet verify passed\nBATCH DONE {STAMP}\n'))
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual(['job rc=1 (job exit evidence)'], campaign_reasons_of(acceptance, 'campaign'))
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_quiet_verify_failure_refuses_acceptance_and_keeps_the_descriptive_numbers(self):
        traceback = ('Traceback (most recent call last):\n'
                     '  File "quiet_window.py", line 91, in main\n'
                     'ValueError: Swarm activity or observation failure invalidated the operation\n')
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2), status=(
            f'lock acquired {STAMP}\nquiet wait passed {STAMP}\njob rc=0 watch rc=0 {STAMP}\n'
            f'{traceback}quiet verify FAILED\nBATCH DONE {STAMP}\n'))
        acceptance = scratch.report('acceptance', control, campaign)
        reasons = campaign_reasons_of(acceptance, 'campaign')
        self.assertEqual('refused', acceptance['claim_status'])
        self.assertIn('quiet verify FAILED (cohort evidence)', reasons)
        self.assertEqual(2, len(reasons), reasons)
        self.assertIn('without a recognised form', reasons[1])
        descriptive = scratch.report('descriptive', control, campaign)
        self.assertEqual('descriptive only', descriptive['claim_status'])
        self.assertEqual([], campaign_reasons_of(descriptive, 'campaign'))

    def test_two_job_lines_refuse_acceptance_as_ambiguous(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        line = f'job rc=0 watch rc=0 {STAMP}'
        campaign = scratch.campaign('campaign', records_for_layout(2), status=(
            f'{line}\n{line}\nquiet verify passed\nBATCH DONE {STAMP}\n'))
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual(['status has 2 job rc lines (ambiguous job exit)'], campaign_reasons_of(acceptance, 'campaign'))
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_unrecognised_status_line_refuses_acceptance_only(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2), status=STATUS_OK + 'something unexpected\n')
        acceptance = scratch.report('acceptance', control, campaign)
        reasons = campaign_reasons_of(acceptance, 'campaign')
        self.assertEqual(1, len(reasons), reasons)
        self.assertIn('status has 1 line(s) without a recognised form', reasons[0])
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])

    def test_malformed_job_line_is_unrecognised_not_a_pass(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2), status=(
            f'job rc=zero watch rc=0 {STAMP}\nquiet verify passed\nBATCH DONE {STAMP}\n'))
        acceptance = scratch.report('acceptance', control, campaign)
        reasons = campaign_reasons_of(acceptance, 'campaign')
        self.assertIn('status has no job rc line (job exit evidence)', reasons)
        self.assertTrue(any('without a recognised form' in reason for reason in reasons), reasons)

    def test_failed_batch_step_refuses_acceptance(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2), status=f'lock acquired {STAMP}\nquiet wait failed {STAMP}\n')
        acceptance = scratch.report('acceptance', control, campaign)
        self.assertEqual('refused', acceptance['claim_status'])
        self.assertIn(f'batch step failed: quiet wait failed {STAMP}', campaign_reasons_of(acceptance, 'campaign'))
        self.assertEqual('descriptive only', scratch.report('descriptive', control, campaign)['claim_status'])


class LedgerTests(unittest.TestCase):
    """The exit ledger is integrity evidence. Each finding below refuses in both modes, with exactly one campaign reason."""

    def assert_one_campaign_reason_in_both_modes(self, scratch, control, campaign, fragment):
        for mode in ('acceptance', 'descriptive'):
            with self.subTest(mode=mode):
                report = scratch.report(mode, control, campaign)
                self.assertEqual('refused', report['claim_status'])
                self.assertIsNone(report['results'])
                reasons = campaign_reasons_of(report, 'campaign')
                self.assertEqual(1, len(reasons), reasons)
                self.assertIn(fragment, reasons[0])

    def test_failed_then_successful_duplicate_is_refused_in_both_modes(self):
        # The coordinator's reproduction: a failed s1-p1-cand entry followed by a successful duplicate. The clean ledger already
        # holds a success, so the statuses are 0, 1, 0. A later success must not erase the failed attempt.
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = default_ledger(CAMPAIGN_JOB, extra=[f'{CAMPAIGN_JOB} s1-p1-cand 1 1.0 2.0 load=0.0 0.0 0.0',
                                                    f'{CAMPAIGN_JOB} s1-p1-cand 0 1.0 2.0 load=0.0 0.0 0.0'])
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign,
                                                      'duplicate ledger entries for s1-p1-cand with statuses [0, 1, 0]')
        report = scratch.report('acceptance', control, campaign)
        self.assertTrue(any('ledger entries are ambiguous (statuses 0, 1, 0)' in line for line in report['refusals']))

    def test_success_then_failed_duplicate_is_refused_in_both_modes(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = default_ledger(CAMPAIGN_JOB, extra=[f'{CAMPAIGN_JOB} s1-p1-cand 1 1.0 2.0 load=0.0 0.0 0.0'])
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign,
                                                      'duplicate ledger entries for s1-p1-cand with statuses [0, 1]')

    def test_identical_duplicates_are_refused_in_both_modes(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = default_ledger(CAMPAIGN_JOB, extra=[f'{CAMPAIGN_JOB} s1-p1-cand 0 1.0 2.0 load=0.0 0.0 0.0'])
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign,
                                                      'duplicate ledger entries for s1-p1-cand with statuses [0, 0]')

    def test_unexpected_process_name_is_refused(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = default_ledger(CAMPAIGN_JOB, extra=[f'{CAMPAIGN_JOB} s9-p9-cand 0 1.0 2.0 load=0.0 0.0 0.0'])
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign, 'names an unexpected process (s9-p9-cand)')

    def test_non_integer_status_is_refused(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = default_ledger(CAMPAIGN_JOB, extra=[f'{CAMPAIGN_JOB} s1-p2-cand ok 1.0 2.0 load=0.0 0.0 0.0'])
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign, 'has a non-integer status (ok)')

    def test_entry_for_another_job_is_refused(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = default_ledger(CAMPAIGN_JOB, extra=['OTHER-JOB s1-p3-cand 0 1.0 2.0 load=0.0 0.0 0.0'])
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign, 'is an entry for another job (OTHER-JOB)')

    def test_unrecognised_ledger_line_is_refused(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = default_ledger(CAMPAIGN_JOB, extra=['garbage'])
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign, 'has no recognised form')

    def test_missing_job_done_is_refused_in_both_modes(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=default_ledger(CAMPAIGN_JOB, job_done=False))
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign, 'ledger has no JOB DONE line for this job')

    def test_missing_ledger_file_is_refused_in_both_modes(self):
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        campaign = scratch.campaign('campaign', records_for_layout(2))
        (campaign / 'timing-exits.txt').unlink()
        self.assert_one_campaign_reason_in_both_modes(scratch, control, campaign, 'timing-exits.txt is missing (exit evidence)')

    def test_process_lines_without_trailing_fields_are_read_not_skipped(self):
        # A line of the form "<job> <name> <status>" is a complete entry. Every process must be read, so the campaign accepts.
        scratch = Scratch(self)
        control = scratch.campaign('control', aa_records(1))
        ledger = [f'job start {STAMP}'] + [f'{CAMPAIGN_JOB} {name} 0' for name in v4.process_names()] + [f'{CAMPAIGN_JOB} JOB DONE {STAMP}']
        campaign = scratch.campaign('campaign', records_for_layout(2), ledger=ledger)
        report = scratch.report('acceptance', control, campaign)
        self.assertEqual('acceptance', report['claim_status'], report['refusals'][:3])


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
        records = aa_records(1)
        records['s1-p1-cand']['measurement_mode'] = 'diagnostic-trace'
        control, campaign = scratch.eligible_pair(control_records=records)
        self.assert_refused_both(scratch, control, campaign, 'diagnostic trace')

    def test_observe_boundary_is_refused(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(control_records=aa_records(1, boundary='observe'))
        self.assert_refused_both(scratch, control, campaign, 'boundary is')

    def test_telemetry_enabled_is_refused(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(control_records=aa_records(1, telemetry='enabled'))
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

    def test_an_aa_control_with_a_different_binary_is_refused(self):
        scratch = Scratch(self)
        records = aa_records(1)
        records['s2-p2-ref']['core_sha256'] = 'a' * 64
        control, campaign = scratch.eligible_pair(control_records=records)
        self.assert_refused_both(scratch, control, campaign, 's2-p2-ref: core_sha256')

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

    def test_a_json_null_record_is_refused_not_treated_as_missing(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair()
        (campaign / 's2-p2-ref.json').write_text('null', encoding='utf-8')
        self.assert_refused_both(scratch, control, campaign, 's2-p2-ref: record is not a JSON object')


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

    def test_output_uses_lf_line_endings(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair()
        report = scratch.report('descriptive', control, campaign)
        target = scratch.root / 'derived.json'
        v4.write_fresh(target, report, (control, campaign))
        self.assertNotIn(b'\r', target.read_bytes())

    def test_output_inside_an_input_folder_is_refused(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair()
        report = scratch.report('descriptive', control, campaign)
        with self.assertRaises(v4.OutputRefused):
            v4.write_fresh(control / 'report.json', report, (control, campaign))
        self.assertFalse((control / 'report.json').exists())

    def test_command_line_returns_refusal_code_and_writes_a_report(self):
        scratch = Scratch(self)
        control, campaign = scratch.eligible_pair(control_records=aa_records(1, timing_eligible=False))
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
        self.assertEqual([], campaign_reasons_of(acceptance, 'control'))
        self.assertEqual([], campaign_reasons_of(acceptance, 'campaign'))
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
