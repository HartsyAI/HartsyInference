"""Versioned timing analyzer, v4. Fail-closed.

Two modes, one set of integrity rules.

  acceptance   A claim that may count as acceptance evidence. Every process needs all of:
               exit status 0 from the campaign's exit ledger; a parsed record in acceptance mode on the public boundary;
               the registered branch, route, arm, telemetry state, cohort and device; quality passed;
               timing_eligible true and acceptance_reviewed true; the registered fixture, tool, production-library
               (core and backend), kernel (PTX and SPIR-V), resolved-settings, knob-provenance and native-library
               identities for its role; exactly the registered number of finite, positive samples; and memory evidence:
               the campaign status records a quiet verify and a zero job exit, and no process log shows an
               out-of-memory or allocator-retry event.

  descriptive  The same integrity rules without the eligibility, review and memory-evidence rules. The numbers stay
               visible, every result is labelled descriptive, and no acceptance claim is made.

Diagnostic traces, failed quality, missing or wrong identities, malformed samples, missing exit status and an
incomplete layout are refused in both modes. Validity is derived from these checks. A successful process exit never
makes a pair valid. Outputs are write-once: the output path must not exist and must not lie inside an input folder.

The registration is an input written before the campaign ran. This module never writes a registration.
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

from statistics_protocol import analyze as frozen_analyze, percentile

ANALYZER = 'timing-analysis-v4'
REGISTRATION_SCHEMA = 'timing-registration-v4'
SEED = 274019
RESAMPLES = 20000
THRESHOLD = 1.03
SESSIONS = (1, 2, 3, 4)  # sessions 1-3 are inferential, session 4 is the confirmation session
PAIRS = (1, 2, 3, 4, 5)
ROLES = ('cand', 'ref')
CLAIMS = ('noninferiority', 'regression')

MEMORY_EVENT = re.compile(r'OOM|OutOfVram|Out of VRAM|async retry|cast-cache eviction|FreeAllDeviceMemory|'
                          r'\[WRN\] \[CudaMemory\]')
EXIT_LINE = re.compile(r'^(\S+) (s\d-p\d-(?:cand|ref)) (-?\d+) ')
REQUIRED_TOP = ('schema', 'registration_id', 'trials', 'statistic', 'seed', 'resamples', 'thresholds', 'expected', 'campaigns')
REQUIRED_EXPECTED = ('measurement_mode', 'boundary', 'branch', 'route', 'telemetry', 'cohort', 'device', 'arm',
                     'legacy_asserted_bounded', 'query_key_permissions', 'prob_value_permissions', 'fixture_sha256',
                     'tool_sha256')
REQUIRED_CAMPAIGN = ('job_id', 'candidate', 'reference')
REQUIRED_IDENTITY = ('core_sha256', 'backend_sha256', 'resolved_settings_sha256', 'knobs_sha256',
                     'native_libraries_sha256', 'ptx_sha256', 'spirv_sha256')


class RegistrationError(ValueError):
    """The registration is missing a field or does not match the frozen protocol."""


class OutputRefused(RuntimeError):
    """The output path is not a fresh file outside the input folders."""


def process_names():
    return [f's{session}-p{pair}-{role}' for session in SESSIONS for pair in PAIRS for role in ROLES]


def canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def identities_from_record(record):
    """Identity digests of one process record. A registration stores these for each role of each campaign."""
    return {
        'core_sha256': record.get('core_sha256'),
        'backend_sha256': record.get('backend_sha256'),
        'resolved_settings_sha256': canonical_sha256(record.get('resolved_settings')),
        'knobs_sha256': sha256_bytes(str(record.get('knobs')).encode('utf-8')),
        'native_libraries_sha256': canonical_sha256(record.get('native_libraries')),
        'ptx_sha256': canonical_sha256(record.get('ptx_sha256')),
        'spirv_sha256': canonical_sha256(record.get('spirv_sha256')),
    }


def load_registration(path):
    """Reads and validates a registration. Raises RegistrationError on any missing or off-protocol field."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))

    def require(condition, message):
        if not condition:
            raise RegistrationError(message)

    require(isinstance(data, dict) and data.get('schema') == REGISTRATION_SCHEMA,
            'registration schema is not timing-registration-v4')
    for key in REQUIRED_TOP:
        require(key in data, f'registration lacks {key}')
    require(data['seed'] == SEED and data['resamples'] == RESAMPLES, 'seed or resamples differ from the frozen protocol')
    require(data['thresholds'] == {'noninferiority': THRESHOLD, 'regression': THRESHOLD},
            'thresholds differ from the frozen 1.03 decision rules')
    require(data['statistic'] in ('mean', 'median'), 'statistic must be mean or median')
    require(isinstance(data['trials'], int) and not isinstance(data['trials'], bool) and data['trials'] > 0,
            'trials must be a positive integer')
    for key in REQUIRED_EXPECTED:
        require(key in data['expected'], f'expected lacks {key}')
    require(set(data['campaigns']) >= {'control', 'campaign'}, 'campaigns must include control and campaign')
    for key, campaign in data['campaigns'].items():
        for field in REQUIRED_CAMPAIGN:
            require(field in campaign, f'campaign {key} lacks {field}')
        for role in ('candidate', 'reference'):
            for field in REQUIRED_IDENTITY:
                require(field in campaign[role], f'campaign {key} {role} lacks {field}')
    return data


def read_campaign(folder, job_id):
    """Reads the exit ledger and the status file of one campaign folder. Missing files read as empty."""
    folder = Path(folder)
    exits_path = folder / 'timing-exits.txt'
    exits_text = exits_path.read_text(encoding='utf-8') if exits_path.exists() else ''
    exits = {}
    for line in exits_text.splitlines():
        match = EXIT_LINE.match(line)
        if match and match.group(1) == job_id:
            exits[match.group(2)] = int(match.group(3))
    status_path = folder / 'status.txt'
    status_text = status_path.read_text(encoding='utf-8') if status_path.exists() else ''
    status_lines = [line.strip() for line in status_text.splitlines()]
    return exits, status_lines


def same(value, expected):
    return value is expected if isinstance(expected, bool) else value == expected


def integrity_reasons(record, exit_code, campaign, role, expected, trials):
    """Checks that apply in both modes. An empty list means the process is intact for this role."""
    reasons = []
    if exit_code is None:
        reasons.append('no exit status recorded')
    elif exit_code != 0:
        reasons.append(f'exit status {exit_code}')
    if not isinstance(record, dict):
        return reasons + ['record is not a JSON object']
    for field in REQUIRED_EXPECTED:
        if field not in record:
            reasons.append(f'{field} is missing')
        elif field not in ('tool_sha256', 'fixture_sha256') and not same(record[field], expected[field]):
            reasons.append(f'{field} is {record[field]!r}, registered {expected[field]!r}')
    if 'fixture_sha256' in record and record['fixture_sha256'] != expected['fixture_sha256']:
        reasons.append('fixture is not the registered fixture')
    if 'tool_sha256' in record and record['tool_sha256'] != expected['tool_sha256']:
        reasons.append('tool is not the registered tool')
    if record.get('measurement_mode') == 'diagnostic-trace':
        reasons.append('diagnostic trace records are never timing evidence')
    if record.get('quality_passed') is not True:
        reasons.append('quality did not pass')
    for field in ('core_sha256', 'backend_sha256'):
        if field not in record:
            reasons.append(f'{field} is missing')
        elif record[field] != campaign[role][field]:
            reasons.append(f'{field} is not the registered production library for this role')
    digests = identities_from_record(record)
    for field in ('resolved_settings_sha256', 'knobs_sha256', 'native_libraries_sha256', 'ptx_sha256', 'spirv_sha256'):
        if digests[field] != campaign[role][field]:
            reasons.append(f'{field} differs from the registered identity for this role')
    samples = record.get('milliseconds')
    if not isinstance(samples, list):
        reasons.append('milliseconds is not a list')
    elif len(samples) != trials:
        reasons.append(f'{len(samples)} samples, registered {trials}')
    else:
        bad = [index for index, value in enumerate(samples)
               if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0]
        if bad:
            reasons.append(f'{len(bad)} malformed samples (first at index {bad[0]})')
    return reasons


def acceptance_reasons(record, log_text):
    """Checks that apply only to acceptance: review, eligibility and per-process memory evidence."""
    reasons = []
    if isinstance(record, dict):
        if record.get('timing_eligible') is not True:
            reasons.append('timing_eligible is not true')
        if record.get('acceptance_reviewed') is not True:
            reasons.append('acceptance_reviewed is not true')
    if log_text is None:
        reasons.append('process log is missing (memory evidence)')
    else:
        event = MEMORY_EVENT.search(log_text)
        if event:
            reasons.append(f'process log shows a memory event ({event.group(0)})')
    return reasons


def campaign_reasons(mode, status_lines):
    reasons = []
    if mode == 'acceptance':
        if 'quiet verify passed' not in status_lines:
            reasons.append('status has no quiet verify passed (cohort and memory evidence)')
        if not any(line.startswith('job rc=0') for line in status_lines):
            reasons.append('status has no job rc=0 (job exit evidence)')
    return reasons


def evaluate_campaign(folder, campaign, expected, trials, statistic, mode):
    """Evaluates every process of one campaign. Returns per-process reasons, values and the verdicts for each level."""
    folder = Path(folder)
    exits, status_lines = read_campaign(folder, campaign['job_id'])
    per_process = {}
    values = {}
    integrity_ok = {}
    acceptance_ok = {}
    for name in process_names():
        role = 'candidate' if name.endswith('-cand') else 'reference'
        record_path = folder / f'{name}.json'
        log_path = folder / f'{name}.log'
        entry = {'input_sha256': sha256_file(record_path) if record_path.exists() else None,
                 'exit_status': exits.get(name), 'integrity_reasons': [], 'acceptance_reasons': []}
        record = None
        if not record_path.exists():
            entry['integrity_reasons'].append('process record is missing')
        else:
            try:
                record = json.loads(record_path.read_text(encoding='utf-8'))
            except ValueError:
                entry['integrity_reasons'].append('process record is not valid JSON')
        if record is not None:
            entry['integrity_reasons'] += integrity_reasons(record, exits.get(name), campaign, role, expected, trials)
        log_text = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else None
        if mode == 'acceptance':
            entry['acceptance_reasons'] = acceptance_reasons(record, log_text)
        integrity_ok[name] = not entry['integrity_reasons']
        acceptance_ok[name] = integrity_ok[name] and not entry['acceptance_reasons']
        if integrity_ok[name]:
            samples = record['milliseconds']
            values[name] = statistics.fmean(samples) if statistic == 'mean' else statistics.median(samples)
        per_process[name] = entry
    return {'per_process': per_process, 'values': values, 'integrity_ok': integrity_ok, 'acceptance_ok': acceptance_ok,
            'campaign_reasons': campaign_reasons(mode, status_lines)}


def descriptive_statistics(sessions, confirmation, claim, seed=SEED, resamples=RESAMPLES):
    """Mirror of statistics_protocol.analyze without its validity gate. Used only for descriptive output.

    Arithmetic and random-number order are identical to the frozen function, so on pairs that the frozen function
    accepts the two give the same numbers. The tests check this.
    """
    def logs(pairs):
        return [math.log(pair['candidate_ms'] / pair['reference_ms']) for pair in pairs]

    groups = [logs(session) for session in sessions]
    holdout = logs(confirmation)
    rng = random.Random(seed)
    bootstrap = [math.exp(sum(sum(rng.choices(group, k=5)) / 5 for group in groups) / 3) for _ in range(resamples)]
    ratio = math.exp(sum(sum(group) / 5 for group in groups) / 3)
    confirmation_ratio = math.exp(sum(holdout) / 5)
    upper_one = percentile(bootstrap, .95)
    lower_one = percentile(bootstrap, .05)
    passed = {'noninferiority': upper_one <= THRESHOLD and confirmation_ratio <= THRESHOLD,
              'regression': lower_one > THRESHOLD and confirmation_ratio > THRESHOLD}[claim]
    return {'ratio': ratio, 'upper95_one_sided': upper_one, 'lower95_one_sided': lower_one,
            'confirmation_ratio': confirmation_ratio, 'claim': claim, 'passed': passed,
            'inferential_pairs': 15, 'confirmation_pairs': 5}


def pairs_from(values, valid=None):
    """Pairs for statistics. With valid given, each pair carries the verdict derived from its two processes."""
    def session_pairs(session):
        out = []
        for pair in PAIRS:
            cand, ref = f's{session}-p{pair}-cand', f's{session}-p{pair}-ref'
            item = {'candidate_ms': values[cand], 'reference_ms': values[ref]}
            if valid is not None:
                item['valid'] = bool(valid[cand] and valid[ref])
            out.append(item)
        return out

    return [session_pairs(1), session_pairs(2), session_pairs(3)], session_pairs(4)


def outcome_of(results):
    if results['regression']['passed']:
        return 'regression detected'
    if results['noninferiority']['passed']:
        return 'noninferiority established'
    return 'inconclusive'


def campaign_results(evaluation, mode):
    """Results for one campaign. Only called when every process passes the level this mode requires."""
    if mode == 'acceptance':
        sessions, confirmation = pairs_from(evaluation['values'], valid=evaluation['acceptance_ok'])
        results = {claim: frozen_analyze(sessions, confirmation, claim) for claim in CLAIMS}
    else:
        sessions, confirmation = pairs_from(evaluation['values'])
        results = {claim: descriptive_statistics(sessions, confirmation, claim) for claim in CLAIMS}
    return {'claims': results, 'outcome': outcome_of(results)}


def build_report(registration_path, control_dir, campaign_dir, mode):
    registration = load_registration(registration_path)
    trials = registration['trials']
    statistic = registration['statistic']
    expected = registration['expected']
    campaigns_out = {}
    evaluations = {}
    for key, folder in (('control', control_dir), ('campaign', campaign_dir)):
        campaign = registration['campaigns'][key]
        evaluations[key] = evaluate_campaign(folder, campaign, expected, trials, statistic, mode)
        folder_path = Path(folder)
        exits_path = folder_path / 'timing-exits.txt'
        status_path = folder_path / 'status.txt'
        campaigns_out[key] = {
            'folder': str(folder_path),
            'job_id': campaign['job_id'],
            'exit_ledger_sha256': sha256_file(exits_path) if exits_path.exists() else None,
            'status_sha256': sha256_file(status_path) if status_path.exists() else None,
            'campaign_reasons': evaluations[key]['campaign_reasons'],
            'processes': evaluations[key]['per_process'],
        }
    refusals = []
    for key in ('control', 'campaign'):
        if evaluations[key]['campaign_reasons']:
            refusals.append(f'{key}: ' + '; '.join(evaluations[key]['campaign_reasons']))
        for name, entry in evaluations[key]['per_process'].items():
            reasons = entry['integrity_reasons'] + (entry['acceptance_reasons'] if mode == 'acceptance' else [])
            if reasons:
                refusals.append(f'{key} {name}: ' + '; '.join(reasons))
    results = {}
    if not refusals:
        results = {key: campaign_results(evaluations[key], mode) for key in ('control', 'campaign')}
    if refusals:
        claim_status = 'refused'
    else:
        claim_status = 'acceptance' if mode == 'acceptance' else 'descriptive only'
    return {
        'analyzer': ANALYZER,
        'analyzer_sha256': sha256_file(__file__),
        'mode': mode,
        'registration_id': registration['registration_id'],
        'registration_sha256': sha256_file(registration_path),
        'claim_status': claim_status,
        'acceptance_eligible': claim_status == 'acceptance',
        'refusals': refusals,
        'campaigns': campaigns_out,
        'results': results if not refusals else None,
    }


def write_fresh(path, payload, input_folders):
    """Writes a new file. An existing file is never replaced, and the output may not lie inside an input folder."""
    target = Path(path).resolve()
    for folder in input_folders:
        folder = Path(folder).resolve()
        if target == folder or folder in target.parents:
            raise OutputRefused(f'output path lies inside input folder {folder}')
    text = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    with open(target, 'x', encoding='utf-8') as handle:
        handle.write(text)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--control-dir', required=True)
    parser.add_argument('--campaign-dir', required=True)
    parser.add_argument('--mode', choices=('acceptance', 'descriptive'), required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.registration, args.control_dir, args.campaign_dir, args.mode)
        write_fresh(args.out, report, (args.control_dir, args.campaign_dir))
    except (RegistrationError, OutputRefused, OSError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    print(f"{report['claim_status']}: {args.out}")
    return 0 if report['claim_status'] in ('acceptance', 'descriptive only') else 2


if __name__ == '__main__':
    sys.exit(main())
