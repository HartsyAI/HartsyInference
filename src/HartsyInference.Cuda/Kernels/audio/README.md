# Audio CUDA sources

Build from the repository root with:

```bash
bash src/HartsyInference.Cuda/Kernels/audio/build.sh
```

See the [parent policy](../README.md) for compiler fallback and shipped PTX. Audio operations are wired into the CUDA backend; the old source-only/stub table was obsolete. Inspect CudaBackend/CudaKernels and the build list for current operation/dtype coverage, then validate the affected codec against its reference.
