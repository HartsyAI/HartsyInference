"""CPU check of h1_step_dump.py: off by default (no files, no work); on with H1_STEP_DUMP_DIR (files written with the values given).

Run: python h1_step_dump_check.py <directory holding h1_step_dump.py>
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import h1_step_dump  # noqa: E402

os.environ.pop("H1_STEP_DUMP_DIR", None)
noise = torch.randn(1, 16, 3, 4, 5)
latents = torch.randn(1, 16, 3, 4, 5)

# off: nothing written, and the call does not fail on a tensor view
before = os.getcwd()
h1_step_dump.maybe_dump(0, torch.tensor(999.0), noise, latents)
assert h1_step_dump._CALL[0] == -1, "the counter moved while off"
assert os.getcwd() == before and not any(n.startswith('call') for n in os.listdir('.')), "files written while off"
print("PASS: off by default (no counter change, no files)")

# on: two calls of two steps each; files named by call and step; values equal the inputs
with tempfile.TemporaryDirectory() as tmp:
    os.environ["H1_STEP_DUMP_DIR"] = tmp
    h1_step_dump._CALL[0] = -1
    for call in range(2):
        for i in range(2):
            h1_step_dump.maybe_dump(i, torch.tensor(float(10 - i)), noise + i, latents + call)
    del os.environ["H1_STEP_DUMP_DIR"]
    names = sorted(os.listdir(tmp))
    expected = sorted(f"call{c:02d}_step{i:02d}_{k}.npy" for c in range(2) for i in range(2) for k in ("t", "noise", "latent"))
    assert names == expected, names
    got = np.load(os.path.join(tmp, "call01_step01_noise.npy"))
    assert np.array_equal(got, (noise + 1).numpy()), "noise values differ"
    got = np.load(os.path.join(tmp, "call01_step00_latent.npy"))
    assert np.array_equal(got, (latents + 1).numpy()), "latent values differ"
    assert float(np.load(os.path.join(tmp, "call00_step01_t.npy"))) == 9.0
    assert got.dtype == np.float32
print("PASS: on: 12 files named by call and step, values and dtype float32")
print("ALL PASS")
