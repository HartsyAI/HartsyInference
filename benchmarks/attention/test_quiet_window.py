import unittest
import argparse
import json
import pathlib
import tempfile
from unittest.mock import patch
from quiet_window import activity, verify, watch


class QuietWindowTests(unittest.TestCase):
    def test_requires_bracketing_and_continuous_idle_coverage(self):
        verify([{'time': 1, 'idle': True}, {'time': 2, 'idle': True}], 1.1, 1.9)
        for records in ([], [{'time': 1, 'idle': True}],
                        [{'time': 1, 'idle': True}, {'time': 2, 'idle': False}],
                        [{'time': 0, 'idle': True}, {'time': 8, 'idle': True}]):
            with self.assertRaises(ValueError):
                verify(records, 1.1, 1.9)

    def test_clock_reversal_is_not_quiet_evidence(self):
        with self.assertRaises(ValueError):
            verify([{'time': 2, 'idle': True}, {'time': 1, 'idle': True}], 1, 2)

    def test_missing_or_nonfinite_global_counters_fail_closed(self):
        for value in (None, {}, {'live_gens': 0},
                      {'live_gens': float('nan'), 'waiting_gens': 0, 'loading_models': 0},
                      {'live_gens': False, 'waiting_gens': 0, 'loading_models': 0}):
            with patch('quiet_window.post', return_value={'status': value}):
                with self.assertRaises(RuntimeError): activity('http://localhost', 'session')

    def test_stop_racing_with_timestamp_still_publishes_a_final_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = argparse.Namespace(state=root/'state', ready=root/'ready', stop=root/'stop', url='unused')
            ticks = iter((2., 3.))
            def clock():
                args.stop.touch()
                return next(ticks)
            with patch('quiet_window.activity', return_value={'live_gens': 0}), \
                 patch('quiet_window.time.time', side_effect=clock), patch('quiet_window.time.sleep'):
                watch(args, 'session')
            rows = [json.loads(line) for line in args.state.read_text().splitlines()]
            self.assertEqual([2., 3.], [row['time'] for row in rows])
            verify(rows, 2.1, 2.5)


if __name__ == '__main__':
    unittest.main()
