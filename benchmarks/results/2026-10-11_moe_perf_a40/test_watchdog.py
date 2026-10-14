#!/usr/bin/env python3
"""Stubbed-API tests for watchdog.py. No network, no key, no pod: run with `python3 test_watchdog.py`."""
import datetime as dt, json, pathlib, tempfile, unittest, unittest.mock as um

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

    def bound(self, pid, hours_ago, max_hours=6.0, name="pod"):
        """A lease already bound to a pod id (what pod_up.py leaves behind)."""
        recs = watchdog.load_leases(self.leases) if self.leases.exists() else {}
        recs[pid] = {"name": name, "started": (T0 - dt.timedelta(hours=hours_ago)).isoformat(),
                     "max_hours": max_hours, "stop_requested": None, "pod_id": pid}
        watchdog.save_leases(recs, self.leases)

    def pending(self, pod_name, hours_ago, max_hours=6.0):
        watchdog.reserve_lease(pod_name, max_hours, path=self.leases, now=T0 - dt.timedelta(hours=hours_ago))

    # --- lease lifecycle -------------------------------------------------------------------------------------------

    def test_reserve_then_bind_moves_the_record_to_the_pod_id(self):
        self.pending("kit-abc123", 0, 3.5)
        self.assertIn(watchdog.PENDING + "kit-abc123", watchdog.load_leases(self.leases))
        watchdog.bind_lease("kit-abc123", "p1", path=self.leases)
        leases = watchdog.load_leases(self.leases)
        self.assertNotIn(watchdog.PENDING + "kit-abc123", leases)
        self.assertEqual(leases["p1"]["pod_id"], "p1")
        self.assertEqual(leases["p1"]["max_hours"], 3.5)

    def test_check_max_hours_rejects_bad_values_and_accepts_good_ones(self):
        for bad in (0, -1, float("nan"), float("inf"), "soon"):
            with self.assertRaises(ValueError):
                watchdog.check_max_hours(bad)
        self.assertEqual(watchdog.check_max_hours("2.5"), 2.5)
        with self.assertRaises(ValueError):
            watchdog.reserve_lease("x", 0, path=self.leases)
        self.assertFalse(self.leases.exists())   # nothing written for a rejected value

    def test_corrupt_lease_file_raises_and_is_not_overwritten(self):
        self.leases.write_text("{not json")
        stub = StubRunPod([])
        with self.assertRaises(ValueError):
            watchdog.run_once(stub, self.leases, now=T0, log=self.log)
        with self.assertRaises(ValueError):
            watchdog.reserve_lease("x", 6, path=self.leases)
        self.assertEqual(self.leases.read_text(), "{not json")
        self.assertEqual(stub.calls, [])

    # --- watchdog pass --------------------------------------------------------------------------------------------

    def test_stops_only_running_bound_pods_past_max(self):
        self.bound("old", 7)
        self.bound("young", 2)
        self.bound("exited", 9)
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

    def test_pending_lease_binds_by_unique_name_and_is_stopped_when_due(self):
        self.pending("kit-abc123", 7)
        stub = StubRunPod([{"id": "p9", "name": "kit-abc123", "desiredStatus": "RUNNING"}])
        stopped = watchdog.run_once(stub, self.leases, now=T0, log=self.log)
        self.assertEqual(stopped, ["p9"])
        self.assertEqual(stub.stops(), ["/pods/p9/stop"])
        self.assertNotIn(watchdog.PENDING + "kit-abc123", watchdog.load_leases(self.leases))

    def test_pending_lease_with_no_pod_is_dropped_after_max_plus_grace(self):
        self.pending("kit-zzz999", 6.5)     # not yet past max + grace: kept (the create may still be landing)
        watchdog.run_once(StubRunPod([]), self.leases, now=T0, log=self.log)
        self.assertIn(watchdog.PENDING + "kit-zzz999", watchdog.load_leases(self.leases))
        watchdog.run_once(StubRunPod([]), self.leases, now=T0 + dt.timedelta(hours=1), log=self.log)
        self.assertNotIn(watchdog.PENDING + "kit-zzz999", watchdog.load_leases(self.leases))

    def test_never_stops_a_pod_without_a_lease(self):
        self.bound("mine", 1)
        stub = StubRunPod([{"id": "someone-else", "name": "other", "desiredStatus": "RUNNING"}])
        watchdog.run_once(stub, self.leases, now=T0 + dt.timedelta(hours=50), log=self.log)
        self.assertEqual(stub.stops(), [])

    def test_stop_is_retried_each_pass_while_the_pod_stays_running(self):
        self.bound("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}])
        watchdog.run_once(stub, self.leases, now=T0, log=self.log)
        watchdog.run_once(stub, self.leases, now=T0 + dt.timedelta(minutes=5), log=self.log)
        self.assertEqual(stub.stops(), ["/pods/p/stop", "/pods/p/stop"])

    def test_pod_restarted_from_the_console_is_stopped_again(self):
        self.bound("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "EXITED"}])
        watchdog.run_once(stub, self.leases, now=T0, log=self.log)
        self.assertEqual(stub.stops(), [])
        stub.pods = [{"id": "p", "desiredStatus": "RUNNING"}]     # operator restarted it
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0 + dt.timedelta(hours=1), log=self.log), ["p"])
        self.assertEqual(stub.stops(), ["/pods/p/stop"])

    def test_exited_pod_is_logged_once_not_every_pass(self):
        self.bound("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "EXITED"}])
        for minutes in (0, 5, 10):
            watchdog.run_once(stub, self.leases, now=T0 + dt.timedelta(minutes=minutes), log=self.log)
        self.assertEqual(sum("not stopping" in m for m in self.logs), 1)

    def test_failed_stop_is_retried_next_pass(self):
        self.bound("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}], stop_code=500)
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), [])
        self.assertIsNone(watchdog.load_leases(self.leases)["p"]["stop_requested"])
        stub.stop_code = 200
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), ["p"])

    def test_listing_failure_issues_no_stops(self):
        self.bound("p", 7)
        stub = StubRunPod([], list_code=503)
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), [])
        self.assertEqual(stub.stops(), [])
        self.assertIn("cannot list pods", self.logs[-1])

    def test_vanished_pod_is_dropped_only_after_continuous_absence(self):
        self.bound("gone", 0.5)
        watchdog.run_once(StubRunPod([]), self.leases, now=T0, log=self.log)
        self.assertIn("gone", watchdog.load_leases(self.leases))      # first missing pass: kept
        watchdog.run_once(StubRunPod([]), self.leases, now=T0 + dt.timedelta(hours=0.5), log=self.log)
        self.assertIn("gone", watchdog.load_leases(self.leases))      # still inside the grace
        watchdog.run_once(StubRunPod([]), self.leases, now=T0 + dt.timedelta(hours=1.5), log=self.log)
        self.assertNotIn("gone", watchdog.load_leases(self.leases))

    def test_one_missing_listing_does_not_cost_an_old_running_pod_its_stop(self):
        # Codex P1: a pod older than the grace must not be dropped on the first pass that omits it.
        self.bound("p", 7)
        watchdog.run_once(StubRunPod([]), self.leases, now=T0, log=self.log)            # one omitted listing
        self.assertIn("p", watchdog.load_leases(self.leases))
        stub = StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}])                   # pod is listed again
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0 + dt.timedelta(minutes=5), log=self.log), ["p"])
        self.assertEqual(stub.stops(), ["/pods/p/stop"])

    def test_missing_since_resets_when_the_pod_reappears(self):
        self.bound("p", 2)
        watchdog.run_once(StubRunPod([]), self.leases, now=T0, log=self.log)
        watchdog.run_once(StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}]), self.leases, now=T0 + dt.timedelta(minutes=10), log=self.log)
        self.assertNotIn("missing_since", watchdog.load_leases(self.leases)["p"])
        watchdog.run_once(StubRunPod([]), self.leases, now=T0 + dt.timedelta(hours=2), log=self.log)
        self.assertIn("p", watchdog.load_leases(self.leases))     # a new absence starts a new grace

    def test_due_pod_that_is_not_running_is_logged_not_stopped(self):
        self.bound("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "EXITED"}])
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), [])
        self.assertEqual(stub.stops(), [])
        self.assertTrue(any("not stopping" in m for m in self.logs))

    def test_drop_pending_removes_only_that_reservation(self):
        self.pending("kit-a", 0)
        self.pending("kit-b", 0)
        watchdog.drop_pending("kit-a", path=self.leases)
        leases = watchdog.load_leases(self.leases)
        self.assertNotIn(watchdog.PENDING + "kit-a", leases)
        self.assertIn(watchdog.PENDING + "kit-b", leases)

    def test_dry_run_posts_nothing_and_keeps_leases(self):
        self.bound("p", 7)
        stub = StubRunPod([{"id": "p", "desiredStatus": "RUNNING"}])
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, dry_run=True, log=self.log), ["p"])
        self.assertEqual(stub.stops(), [])
        self.assertIsNone(watchdog.load_leases(self.leases)["p"]["stop_requested"])

    def test_no_leases_means_no_api_calls(self):
        stub = StubRunPod([])
        self.assertEqual(watchdog.run_once(stub, self.leases, now=T0, log=self.log), [])
        self.assertEqual(stub.calls, [])

    # --- transport ------------------------------------------------------------------------------------------------

    def test_curl_calls_are_time_bounded(self):
        with um.patch.object(watchdog.subprocess, "run") as run:
            run.return_value.stdout = "{}\n200"
            watchdog.api("GET", "/pods")
            watchdog.api("POST", "/pods", {"name": "x"}, timeout=watchdog.CREATE_TIMEOUT_S)
        first, second = run.call_args_list[0][0][0], run.call_args_list[1][0][0]
        self.assertEqual(first[first.index("--max-time") + 1], str(watchdog.REQUEST_TIMEOUT_S))
        self.assertEqual(second[second.index("--max-time") + 1], str(watchdog.CREATE_TIMEOUT_S))

    def test_truncated_json_body_comes_back_as_text_not_an_exception(self):
        with um.patch.object(watchdog.subprocess, "run") as run:
            run.return_value.stdout = '[{"id": "p1", "desired\n200'
            code, body = watchdog.api("GET", "/pods")
        self.assertEqual(code, 200)
        self.assertIsInstance(body, str)
        self.bound("p", 7)
        # A malformed listing is a failed pass: no stop is issued and the pass does not raise.
        self.assertEqual(watchdog.run_once(lambda m, p, b=None: (code, body), self.leases, now=T0, log=self.log), [])


if __name__ == "__main__":
    unittest.main()
