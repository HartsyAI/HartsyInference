#!/usr/bin/env python3
"""Stubbed-API tests for watchdog.py. No network, no key, no pod: run with `python3 test_watchdog.py`."""
import datetime as dt, json, pathlib, tempfile, unittest

import watchdog

T0 = dt.datetime(2026, 10, 13, 12, 0, tzinfo=dt.timezone.utc)


class StubRunPod:
    """Stands in for watchdog.api. `pods` is the GET /pods listing; stop POSTs are recorded and answered with `stop_code`."""

    def __init__(self, pods, stop_code=200, list_code=200):
        self.pods = pods
        self.stop_code = stop_code
        self.list_code = list_code
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path))
        if method == "GET" and path == "/pods":
            return self.list_code, self.pods
        if method == "POST" and path.endswith("/stop"):
            return self.stop_code, {"id": path.split("/")[2]} if self.stop_code < 300 else "boom"
        raise AssertionError(f"unexpected call {method} {path}")

    def stops(self):
        return [p for m, p in self.calls if m == "POST"]


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.leases = pathlib.Path(self.dir.name) / "leases.json"
        self.logs = []

    def tearDown(self):
        self.dir.cleanup()

    def log(self, m):
        self.logs.append(m)

    def lease(self, pid, hours_ago, max_hours=6.0):
        watchdog.record_lease(pid, f"name-{pid}", max_hours, path=self.leases, now=T0 - dt.timedelta(hours=hours_ago))

    def test_record_lease_writes_start_and_max(self):
        self.lease("p1", 0, 3.5)
        rec = watchdog.load_leases(self.leases)["p1"]
        self.assertEqual(rec["max_hours"], 3.5)
        self.assertEqual(rec["started"], T0.isoformat())
        self.assertIsNone(rec["stop_requested"])

    def test_record_lease_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            watchdog.record_lease("p1", "x", 0, path=self.leases)

    def test_stops_only_running_pods_past_max(self):
        self.lease("old", 7)          # 7 h old, max 6: due
        self.lease("young", 2)        # not due
        self.lease("exited", 9)       # due, but already EXITED: left alone
        stub = StubRunPod([
            {"id": "old", "desiredStatus": "RUNNING"},
            {"id": "young", "desiredStatus": "RUNNING"},
            {"id": "exited", "desiredStatus": "EXITED"},
        ])
        stopped = watchdog.run_once(stub, self.leases, now=T0, log=self.log)
        self.assertEqual(stopped, ["old"])
        self.assertEqual(stub.stops(), ["/pods/old/stop"])
        rec = watchdog.load_leases(self.leases)
        self.assertIsNotNone(rec["old"]["stop_requested"])
        self.assertIsNone(rec["young"]["stop_requested"])

    def test_never_stops_a_pod_without_a_lease(self):
        self.lease("mine", 1)
        stub = StubRunPod([{"id": "someone-else", "desiredStatus": "RUNNING"}])
        watchdog.run_once(stub, self.leases, now=T0 + dt.timedelta(hours=50), log=self.log)
        self.assertEqual(stub.stops(), [])

    def test_no_second_stop_after_request_recorded(self):
        self.lease("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}])
        watchdog.run_once(stub, self.leases, now=T0, log=self.log)
        watchdog.run_once(stub, self.leases, now=T0 + dt.timedelta(minutes=5), log=self.log)
        self.assertEqual(stub.stops(), ["/pods/p/stop"])

    def test_failed_stop_is_retried_next_pass(self):
        self.lease("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}], stop_code=500)
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), [])
        self.assertIsNone(watchdog.load_leases(self.leases)["p"]["stop_requested"])
        stub.stop_code = 200
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), ["p"])

    def test_listing_failure_issues_no_stops(self):
        self.lease("p", 7)
        stub = StubRunPod([], list_code=503)
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), [])
        self.assertEqual(stub.stops(), [])
        self.assertIn("cannot list pods", self.logs[-1])

    def test_vanished_pod_drops_its_lease(self):
        self.lease("gone", 1)
        stub = StubRunPod([])
        watchdog.run_once(stub, self.leases, now=T0, log=self.log)
        self.assertNotIn("gone", watchdog.load_leases(self.leases))

    def test_dry_run_posts_nothing_and_keeps_leases(self):
        self.lease("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}])
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, dry_run=True, log=self.log), ["p"])
        self.assertEqual(stub.stops(), [])
        self.assertIsNone(watchdog.load_leases(self.leases)["p"]["stop_requested"])

    def test_no_leases_means_no_api_calls(self):
        stub = StubRunPod([])
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), [])
        self.assertEqual(stub.calls, [])


if __name__ == "__main__":
    unittest.main()
