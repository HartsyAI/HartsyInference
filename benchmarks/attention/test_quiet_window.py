import unittest
from quiet_window import verify


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


if __name__ == '__main__':
    unittest.main()
