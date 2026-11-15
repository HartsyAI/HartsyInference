#!/usr/bin/env python3
"""Observe Swarm global activity without stopping services or changing backend settings."""
import argparse
import datetime
import json
import pathlib
import time
import urllib.request


def post(url, method, payload):
    request = urllib.request.Request(
        url.rstrip('/') + '/API/' + method,
        json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.load(response)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def activity(url, session):
    status = post(url, 'GetGlobalStatus', {'session_id': session}).get('status')
    required = ('live_gens', 'waiting_gens', 'loading_models')
    if not isinstance(status, dict) or any(key not in status for key in required):
        raise RuntimeError('Global status is incomplete; quiet cannot be established')
    if any(not isinstance(status[key], (int, float)) or status[key] < 0 for key in required):
        raise RuntimeError('Global status counters are invalid')
    return {key: status[key] for key in required}


def verify(records, start, end, max_gap=3):
    if not records or end < start:
        raise ValueError('Missing or invalid quiet-window interval')
    times = [row['time'] for row in records]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Quiet-window timestamps are not increasing')
    before = [i for i, timestamp in enumerate(times) if timestamp <= start]
    after = [i for i, timestamp in enumerate(times) if timestamp >= end]
    if not before or not after:
        raise ValueError('Quiet-window samples do not bracket the operation')
    window = records[before[-1]:after[0] + 1]
    if any(not row.get('idle', False) for row in window):
        raise ValueError('Swarm activity or observation failure invalidated the operation')
    if any(b['time'] - a['time'] > max_gap for a, b in zip(window, window[1:])):
        raise ValueError('Quiet-window observation has a coverage gap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('wait', 'watch'):
        child = sub.add_parser(command)
        child.add_argument('--url', default='http://127.0.0.1:7801')
        if command == 'wait':
            child.add_argument('--seconds', type=float, default=30)
            child.add_argument('--timeout', type=float, default=540)
        else:
            child.add_argument('--state', type=pathlib.Path, required=True)
            child.add_argument('--ready', type=pathlib.Path, required=True)
            child.add_argument('--stop', type=pathlib.Path, required=True)
    child = sub.add_parser('verify')
    child.add_argument('--state', type=pathlib.Path, required=True)
    child.add_argument('--start', type=float, required=True)
    child.add_argument('--end', type=float, required=True)
    args = parser.parse_args()
    if args.command == 'verify':
        verify([json.loads(line) for line in args.state.read_text().splitlines()], args.start, args.end)
        return
    session = post(args.url, 'GetNewSession', {})['session_id']
    if args.command == 'wait':
        if args.seconds <= 0 or args.timeout < args.seconds:
            raise ValueError('Timeout must cover the positive quiet interval')
        deadline, quiet = time.monotonic() + args.timeout, None
        while time.monotonic() < deadline:
            idle = not any(activity(args.url, session).values())
            quiet = (quiet if quiet is not None else time.monotonic()) if idle else None
            if quiet is not None and time.monotonic() - quiet >= args.seconds:
                return
            time.sleep(1)
        raise TimeoutError('Swarm did not become quiet; no service was stopped')
    if args.stop.exists() or args.ready.exists():
        raise ValueError('Watch ready/stop files must be fresh')
    with args.state.open('x') as output:
        while True:
            counters = activity(args.url, session)
            row = {'time': time.time(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'idle': not any(counters.values()), 'counters': counters}
            output.write(json.dumps(row) + '\n')
            output.flush()
            if not args.ready.exists():
                if not row['idle']:
                    raise RuntimeError('Swarm became active before measurement')
                args.ready.touch()
            if args.stop.exists():
                return
            time.sleep(0.5)


if __name__ == '__main__':
    main()
