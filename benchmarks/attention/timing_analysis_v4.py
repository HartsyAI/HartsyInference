"""Versioned timing analyzer, v4. Fail-closed.

Two modes, one set of integrity rules.

  acceptance   A claim that may count as acceptance evidence. Every process needs all of:
               exit status 0 from the campaign's exit ledger; a parsed record in acceptance mode on the public boundary;
               the registered branch, route, arm, telemetry state, cohort and device; quality passed;
               timing_eligible true and acceptance_reviewed true; the registered fixture, tool, production-library
               (core and backend), kernel (PTX and SPIR-V), resolved-settings, knob-provenance and native-library
               identities for its role; exactly the registered number of finite, positive samples; and no
               out-of-memory or allocator-retry event in its process log. The campaign must also show batch evidence in
               status.txt: one job line with rc 0, one watcher line with rc 0, exactly one quiet verify pass, no quiet
               verify failure, exactly one BATCH DONE, and no line of unrecognised form.

  descriptive  The same integrity rules without the eligibility, review, memory and status rules. The numbers stay
               visible, every result is labelled descriptive, and no acceptance claim is made.

Integrity rules apply in both modes. They cover the exit ledger, which must be a complete and unambiguous record of this
job: each process appears at most once with an integer status, every line has a recognised form, no line belongs to
another job or names an unexpected process, and the ledger records one JOB DONE. A diagnostic trace, failed quality, a
missing or wrong identity, malformed samples and an incomplete layout are also refused in both modes. The A/A control
must bind identical candidate and reference identities. Validity is derived from these checks. A successful process exit
never makes a pair valid. Outputs are write-once: the output path must not exist and must not lie inside an input folder.

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
# Ledger lines. Each is matched in full. A line that matches none of these is a finding, never skipped.
LEDGER_JOB_START = re.compile(r'job start(?: \S+)?')
LEDGER_JOB_DONE = re.compile(r'(\S+) JOB DONE(?: \S+)?')
LEDGER_ENTRY = re.compile(r'(\S+) (\S+) (\S+)(?: .*)?')
INTEGER = re.compile(r'-?\d+')
# Status lines written by the batch script, matched in full. The watcher and the verifier append their own output to the
# same file, so any other line is a finding, never skipped.
STATUS_JOB = re.compile(r'job rc=(-?\d+) watch rc=(-?\d+)(?: \S+)?')
STATUS_STEP_PASSED = re.compile(r'(?:lock acquired|quiet wait passed)(?: \S+)?')
STATUS_STEP_FAILED = re.compile(r'(?:lock failed|quiet wait failed|watcher not ready)(?: \S+)?')
STATUS_BATCH_DONE = re.compile(r'BATCH DONE(?: \S+)?')
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


PROCESS_NAMES = frozenset(process_names())


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
    control = data['campaigns']['control']
    require(all(control['candidate'][field] == control['reference'][field] for field in REQUIRED_IDENTITY),
            'the control campaign must bind identical candidate and reference identities (an A/A control)')
    return data


def read_ledger(folder, job_id):
    """Parses one campaign's exit ledger. Returns the statuses recorded for each process name and the ledger findings.
    Every non-blank line must have a recognised form, and nothing is skipped. Findings refuse in both modes."""
    path = Path(folder) / 'timing-exits.txt'
    if not path.exists():
        return {'statuses': {}, 'findings': ['timing-exits.txt is missing (exit evidence)']}
    findings = []
    statuses = {}
    job_starts = job_done = 0
    for number, raw in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if LEDGER_JOB_START.fullmatch(line):
            job_starts += 1
            continue
        done = LEDGER_JOB_DONE.fullmatch(line)
        if done:
            if done.group(1) == job_id:
                job_done += 1
            else:
                findings.append(f'ledger line {number} is a JOB DONE for another job ({done.group(1)})')
            continue
        entry = LEDGER_ENTRY.fullmatch(line)
        if not entry:
            findings.append(f'ledger line {number} has no recognised form')
            continue
        job, name, status = entry.groups()
        if job != job_id:
            findings.append(f'ledger line {number} is an entry for another job ({job})')
        elif name not in PROCESS_NAMES:
            findings.append(f'ledger line {number} names an unexpected process ({name})')
        elif not INTEGER.fullmatch(status):
            findings.append(f'ledger line {number} for {name} has a non-integer status ({status})')
        else:
            statuses.setdefault(name, []).append(int(status))
    if job_starts > 1:
        findings.append(f'ledger has {job_starts} job start lines')
    if job_done == 0:
        findings.append('ledger has no JOB DONE line for this job')
    elif job_done > 1:
        findings.append(f'ledger has {job_done} JOB DONE lines for this job')
    for name in sorted(statuses):
        if len(statuses[name]) > 1:
            findings.append(f'duplicate ledger entries for {name} with statuses {statuses[name]}')
    return {'statuses': statuses, 'findings': findings}


def exit_findings(statuses):
    """Process-level findings from the ledger entries recorded for one process name."""
    if len(statuses) > 1:
        return ['ledger entries are ambiguous (statuses ' + ', '.join(str(status) for status in statuses) + ')']
    if not statuses:
        return ['no exit status recorded']
    if statuses[0] != 0:
        return [f'exit status {statuses[0]}']
    return []


def read_status(folder):
    """The non-blank lines of status.txt, or None when the file is missing."""
    path = Path(folder) / 'status.txt'
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()]


def status_findings(lines):
    """Acceptance findings from status.txt. They refuse acceptance only: the descriptive numbers do not depend on the batch
    evidence. The file also holds the watcher's and the verifier's own output, so an unrecognised line is a finding."""
    if lines is None:
        return ['status.txt is missing (batch evidence)']
    findings = []
    job_rc, watch_rc, unrecognised = [], [], []
    passed = failed = done = 0
    for line in lines:
        job = STATUS_JOB.fullmatch(line)
        if job:
            job_rc.append(int(job.group(1)))
            watch_rc.append(int(job.group(2)))
        elif line == 'quiet verify passed':
            passed += 1
        elif line == 'quiet verify FAILED':
            failed += 1
        elif STATUS_BATCH_DONE.fullmatch(line):
            done += 1
        elif STATUS_STEP_PASSED.fullmatch(line):
            pass
        elif STATUS_STEP_FAILED.fullmatch(line):
            findings.append(f'batch step failed: {line}')
        else:
            unrecognised.append(line)
    if not job_rc:
        findings.append('status has no job rc line (job exit evidence)')
    elif len(job_rc) > 1:
        findings.append(f'status has {len(job_rc)} job rc lines (ambiguous job exit)')
    else:
        if job_rc[0] != 0:
            findings.append(f'job rc={job_rc[0]} (job exit evidence)')
        if watch_rc[0] != 0:
            findings.append(f'watch rc={watch_rc[0]} (watcher evidence)')
    if failed:
        findings.append('quiet verify FAILED (cohort evidence)')
    elif passed != 1:
        findings.append(f'quiet verify passed {passed} times (expected once)')
    if done != 1:
        findings.append(f'BATCH DONE recorded {done} times (expected once)')
    if unrecognised:
        findings.append(f'status has {len(unrecognised)} line(s) without a recognised form '
                        f'(first: {unrecognised[0][:60]!r})')
    return findings


def same(value, expected):
    return value is expected if isinstance(expected, bool) else value == expected


def integrity_reasons(record, campaign, role, expected, trials):
    """Checks that apply in both modes to one process record. An empty list means the record is intact for this role.
    The exit status is checked against the ledger by exit_findings."""
    if not isinstance(record, dict):
        return ['record is not a JSON object']
    reasons = []
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
    try:
        digests = identities_from_record(record)
    except (TypeError, ValueError):
        reasons.append('identity fields are not canonical JSON')
        digests = {}
    for field in ('resolved_settings_sha256', 'knobs_sha256', 'native_libraries_sha256', 'ptx_sha256', 'spirv_sha256'):
        if field in digests and digests[field] != campaign[role][field]:
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


def read_record(path):
    """Returns (record, None), or (None, reason) when the record is missing or is not valid JSON."""
    if not path.exists():
        return None, 'process record is missing'
    try:
        return json.loads(path.read_text(encoding='utf-8')), None
    except ValueError:
        return None, 'process record is not valid JSON'


def campaign_findings(mode, ledger, status_lines):
    """Campaign-level refusals for one mode. Ledger findings refuse in both modes; status findings refuse acceptance only."""
    reasons = list(ledger['findings'])
    if mode == 'acceptance':
        reasons += status_findings(status_lines)
    return reasons


def evaluate_campaign(folder, campaign, expected, trials, statistic, mode):
    """Evaluates every process of one campaign and the campaign-level findings for the requested mode."""
    folder = Path(folder)
    ledger = read_ledger(folder, campaign['job_id'])
    status_lines = read_status(folder)
    per_process = {}
    values = {}
    integrity_ok = {}
    acceptance_ok = {}
    for name in process_names():
        role = 'candidate' if name.endswith('-cand') else 'reference'
        record_path = folder / f'{name}.json'
        log_path = folder / f'{name}.log'
        statuses = ledger['statuses'].get(name, [])
        entry = {'input_sha256': sha256_file(record_path) if record_path.exists() else None,
                 'exit_status': statuses[0] if len(statuses) == 1 else None,
                 'integrity_reasons': exit_findings(statuses), 'acceptance_reasons': []}
        record, load_reason = read_record(record_path)
        if load_reason is not None:
            entry['integrity_reasons'].append(load_reason)
        else:
            entry['integrity_reasons'] += integrity_reasons(record, campaign, role, expected, trials)
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
            'campaign_reasons': campaign_findings(mode, ledger, status_lines)}


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
    with open(target, 'x', encoding='utf-8', newline='\n') as handle:
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
    except (RegistrationError, OutputRefused, OSError, ValueError, KeyError, TypeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    print(f"{report['claim_status']}: {args.out}")
    return 0 if report['claim_status'] in ('acceptance', 'descriptive only') else 2


if __name__ == '__main__':
    sys.exit(main())
