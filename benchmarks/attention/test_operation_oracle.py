import unittest
try:
    import torch
except ImportError:
    torch = None
from operation_oracle import dense, merge, sol


@unittest.skipIf(torch is None, 'Reference tests require the locked PyTorch environment')
class OperationOracleTests(unittest.TestCase):
    def test_dense_mixed_lengths_bottom_right_and_empty_rows(self):
        q = torch.zeros(2, 2, 2, 2)
        k = torch.zeros(2, 1, 4, 2)
        v = torch.arange(16).reshape(2, 1, 4, 2).float()
        output, lse = dense(q, k, v, 1, [0, 4], True, [-2, 2])
        self.assertTrue(torch.equal(output[0], torch.zeros_like(output[0])))
        self.assertTrue(torch.isneginf(lse[0]).all())
        self.assertTrue(torch.equal(output[1, 0, 0], v[1, 0, :3].mean(0).double()))

    def test_merge_counts_prefix_once(self):
        q = torch.zeros(1, 1, 2, 2)
        k = torch.zeros_like(q)
        v = torch.tensor([[[[10., 10.], [20., 20.]]]])
        prefix = torch.full_like(q, 2.)
        lse = torch.full((1, 1, 2), 1.3862943611198906, dtype=torch.float64)
        actual = merge(prefix, lse, q, k, v, 1, [1, 3])
        self.assertAlmostEqual(float(actual[0, 0, 0, 0]), 18/5)
        self.assertAlmostEqual(float(actual[0, 0, 1, 0]), 38/6)

    def test_sol_tail_has_block_multiplicity_and_shared_denominator(self):
        q = torch.zeros(1, 1, 256, 2)
        v = torch.cat([torch.full((1, 1, 64, 2), float(n)) for n in (10, 20, 30, 40)], 2)
        with_tail = sol(q, q, v, 1, {'tail': True})
        without_tail = sol(q, q, v, 1, {'tail': False})
        self.assertTrue(torch.equal(with_tail, torch.full_like(with_tail, 25)))
        self.assertEqual(float(without_tail[0, 0, 0, 0]), 15)


if __name__ == '__main__':
    unittest.main()
