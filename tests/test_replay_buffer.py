import unittest

import numpy as np

from src.agents.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_ring_overwrite_and_sampling(self):
        buf = ReplayBuffer(10, 3, seed=0)
        for i in range(25):
            buf.add(np.full(3, i), i % 4, float(i), np.full(3, i + 1), i % 7 == 0)
        self.assertEqual(len(buf), 10)
        self.assertEqual(set(buf.s[:, 0].astype(int)), set(range(15, 25)))  # only the newest 10 remain
        s, a, r, s2, d = buf.sample(8)
        self.assertEqual((s.shape, s2.shape, a.shape), ((8, 3), (8, 3), (8,)))
        self.assertTrue(np.allclose(s2[:, 0], s[:, 0] + 1))                 # (s, s2) pairs stay aligned
        self.assertEqual(len({tuple(x) for x in s}), 8)                      # no duplicates (without replacement)
        self.assertTrue(set(d.tolist()) <= {0.0, 1.0})

    def test_sampling_is_reproducible(self):
        def draw():
            b = ReplayBuffer(50, 2, seed=1)
            for i in range(50):
                b.add([i, i], 0, 0.0, [i, i], False)
            return b.sample(16)[0]
        np.testing.assert_array_equal(draw(), draw())


if __name__ == "__main__":
    unittest.main()
