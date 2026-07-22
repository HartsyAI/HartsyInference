Label fix (2026-07-22): the harness's GPU-name suffix uses `nvidia-smi -i $CVD` but CVD follows CUDA
device order, which is REVERSED vs SMI order on this box (CUDA 0 = 4090 = SMI 1). Both H0 runs executed
on their TAGGED GPU (verified: Sdpa_F16 shape-8 = 130.4 ms here = the 3060's known value; 22.5 ms in the
4090 run). Dir names corrected by stripping the wrong auto-suffix. Harness fix: map CVD through CUDA
order (or query the name via the CUDA API in-process) before naming.
