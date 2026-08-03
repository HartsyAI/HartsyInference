using System.Threading;
using System.Threading.Tasks;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Regression coverage for two backends sharing one <see cref="FluxRope"/> instance: CFG-branch
/// parallelism (Wan/Flux, ROADMAP.md §1) calls <c>_transformer.Forward</c> concurrently on two
/// <see cref="CudaBackend"/> instances through the SAME transformer object, and a DiT block-range shard (Phase
/// 8) does the same sequentially (backend A's blocks, then backend B's). Confirms results stay numerically
/// correct in both shapes and that alternating backends doesn't crash or corrupt state. Note:
/// <c>IBackend.WanRopeInterleaved</c> re-stages cos/sin per call (not a persistent weight-cache hit), so a
/// stale cross-backend Tensor reference was never actually a wrong-device numerics bug — the real defect
/// fixed alongside these tests was a Dispose-during-concurrent-read hazard on cache rebuild (single unkeyed
/// slot calling <c>Tensor.Dispose()</c> unconditionally) plus an unsynchronized <c>_cosCache</c>/<c>_sinCache</c>
/// write race in <c>Precompute</c>; see FluxRope.cs's <c>_gpuLock</c> doc comment.</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed unsafe class FluxRopeCrossBackendTests
{
    private readonly ITestOutputHelper _output;
    public FluxRopeCrossBackendTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    /// <summary>CPU-side ground truth: the same interleaved-pair rotation FluxRope.Forward performs, computed
    /// independently so a wrong-device (or torn-cache) GPU result can't accidentally agree with itself.</summary>
    private static float[] CpuRotate(float[] src, float[] cos, float[] sin, int seqLen, int headDim)
    {
        float[] outp = (float[])src.Clone();
        int halfDim = headDim / 2;
        for (int s = 0; s < seqLen; s++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                int baseIdx = s * headDim;
                float x0 = src[baseIdx + 2 * i];
                float x1 = src[baseIdx + 2 * i + 1];
                float c = cos[s * halfDim + i];
                float sn = sin[s * halfDim + i];
                outp[baseIdx + 2 * i] = c * x0 - sn * x1;
                outp[baseIdx + 2 * i + 1] = sn * x0 + c * x1;
            }
        }
        return outp;
    }

    /// <summary>Sequential cross-backend use: build GPU rope tables for backend A, then for backend B, then back
    /// to A, on the SAME FluxRope instance (the shape of a DiT block-range shard: blocks 0..N run on A, blocks
    /// N..last run on B, one thread, one step). Asserts A's second round trip is still correct after B's build —
    /// guards against B's cache entry evicting A's (the per-backend Dictionary keeps both live simultaneously,
    /// unlike a single unkeyed slot that only ever remembers the most recent backend).</summary>
    [Fact]
    public void ApplyGpu_SequentialTwoBackends_BothGetCorrectOwnTables()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // Cross-DEVICE backends, not just cross-backend: on one shared physical GPU, a stale cross-backend
        // tensor reference is numerically harmless (same context, same pointer validity, identical content) —
        // the original bug only actually corrupts/crashes when the two backends are on different physical
        // devices (a raw device pointer from GPU 0's context is not a valid address in GPU 1's context).
        // Falls back to same-ordinal (weaker, still exercises the cache-keying logic) on a single-GPU box.
        int deviceCount = CudaContext.GetDeviceCount();
        int secondOrdinal = deviceCount >= 2 ? 1 : 0;
        _output.WriteLine($"Devices: {deviceCount}; second backend on ordinal {secondOrdinal}.");
        using CudaBackend a = new(deviceOrdinal: 0, PtxDir());
        using CudaBackend b = new(deviceOrdinal: secondOrdinal, PtxDir());

        int[] axesDim = [4, 4, 4]; // headDim = 12
        const int headDim = 12;
        const int seqLen = 3;
        const int numHeads = 2;

        FluxRope rope = new(axesDim, theta: 10000);
        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen: 0, hPacked: seqLen, wPacked: 1);
        rope.Precompute(posIds);
        posIds.Dispose();

        TensorShape qkShape = new(1, seqLen, numHeads, headDim); // pre-permute [B,S,H,D] — ApplyGpuGqa's contract
        using Tensor qA = RandomF32(qkShape, seed: 11);
        using Tensor kA = RandomF32(qkShape, seed: 12);
        float[] qASrc = ToArray(qA);

        rope.ApplyGpuGqa(a, qA, kA, numHeads, numHeads);
        float[] qAResult = ToArray(qA);

        using Tensor qB = RandomF32(qkShape, seed: 21);
        using Tensor kB = RandomF32(qkShape, seed: 22);
        float[] qBSrc = ToArray(qB);

        rope.ApplyGpuGqa(b, qB, kB, numHeads, numHeads);
        float[] qBResult = ToArray(qB);

        // Re-run A after B has built its own tables — A's slot must not have been evicted by B's build.
        using Tensor qA2 = RandomF32(qkShape, seed: 31);
        using Tensor kA2 = RandomF32(qkShape, seed: 32);
        float[] qA2Src = ToArray(qA2);
        rope.ApplyGpuGqa(a, qA2, kA2, numHeads, numHeads);
        float[] qA2Result = ToArray(qA2);

        float[] cosHost = HostCos(rope, seqLen, headDim);
        float[] sinHost = HostSin(rope, seqLen, headDim);

        AssertRotationMatches(qASrc, qAResult, cosHost, sinHost, seqLen, numHeads, headDim, "A (first)");
        AssertRotationMatches(qBSrc, qBResult, cosHost, sinHost, seqLen, numHeads, headDim, "B");
        AssertRotationMatches(qA2Src, qA2Result, cosHost, sinHost, seqLen, numHeads, headDim, "A (after B)");
    }

    /// <summary>Concurrent cross-backend use: the CFG-branch-parallelism shape — two threads, two backends,
    /// same FluxRope, same call at (roughly) the same time, repeated to widen the window for the
    /// Dispose-during-read / torn-write races the lock closes. Each backend's dictionary entry is independent,
    /// and results must stay correct across all iterations.</summary>
    [Fact]
    public void ApplyGpu_ConcurrentTwoBackends_NoCrossContamination()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        int deviceCount = CudaContext.GetDeviceCount();
        int secondOrdinal = deviceCount >= 2 ? 1 : 0;
        using CudaBackend a = new(deviceOrdinal: 0, PtxDir());
        using CudaBackend b = new(deviceOrdinal: secondOrdinal, PtxDir());

        int[] axesDim = [4, 4, 4];
        const int headDim = 12;
        const int seqLen = 3;
        const int numHeads = 2;

        FluxRope rope = new(axesDim, theta: 10000);
        Tensor posIds = FluxRope.BuildPositionIds(txtSeqLen: 0, hPacked: seqLen, wPacked: 1);
        rope.Precompute(posIds);
        posIds.Dispose();

        TensorShape qkShape = new(1, seqLen, numHeads, headDim);
        float[] cosHost = HostCos(rope, seqLen, headDim);
        float[] sinHost = HostSin(rope, seqLen, headDim);

        for (int iter = 0; iter < 8; iter++)
        {
            using Tensor qA = RandomF32(qkShape, seed: 100 + iter);
            using Tensor kA = RandomF32(qkShape, seed: 200 + iter);
            using Tensor qB = RandomF32(qkShape, seed: 300 + iter);
            using Tensor kB = RandomF32(qkShape, seed: 400 + iter);
            float[] qASrc = ToArray(qA);
            float[] qBSrc = ToArray(qB);

            Barrier barrier = new(2);
            Parallel.Invoke(
                () => { barrier.SignalAndWait(); rope.ApplyGpuGqa(a, qA, kA, numHeads, numHeads); },
                () => { barrier.SignalAndWait(); rope.ApplyGpuGqa(b, qB, kB, numHeads, numHeads); });

            AssertRotationMatches(qASrc, ToArray(qA), cosHost, sinHost, seqLen, numHeads, headDim, $"A iter {iter}");
            AssertRotationMatches(qBSrc, ToArray(qB), cosHost, sinHost, seqLen, numHeads, headDim, $"B iter {iter}");
        }
    }

    private static void AssertRotationMatches(float[] src, float[] result, float[] cos, float[] sin,
        int seqLen, int numHeads, int headDim, string label)
    {
        for (int h = 0; h < numHeads; h++)
        {
            float[] srcHead = new float[seqLen * headDim];
            float[] resultHead = new float[seqLen * headDim];
            for (int s = 0; s < seqLen; s++)
                for (int d = 0; d < headDim; d++)
                {
                    int idx = (s * numHeads + h) * headDim + d; // [1, S, H, D] layout
                    srcHead[s * headDim + d] = src[idx];
                    resultHead[s * headDim + d] = result[idx];
                }
            float[] expected = CpuRotate(srcHead, cos, sin, seqLen, headDim);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - resultHead[i]) < 1e-4f,
                    $"{label} head {h} idx {i}: expected {expected[i]}, got {resultHead[i]}");
        }
    }

    private static float[] ToArray(Tensor t)
    {
        float[] arr = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) arr[i] = p[i];
        return arr;
    }

    // Rebuilds the host-side cos/sin FluxRope.Precompute produced, via the same BuildPositionIds/theta math the
    // instance already ran — independent of the private cache this test is trying to validate.
    private static float[] HostCos(FluxRope rope, int seqLen, int headDim) => HostTable(rope, seqLen, headDim, cos: true);
    private static float[] HostSin(FluxRope rope, int seqLen, int headDim) => HostTable(rope, seqLen, headDim, cos: false);

    private static float[] HostTable(FluxRope rope, int seqLen, int headDim, bool cos)
    {
        int[] axesDim = [4, 4, 4];
        const int theta = 10000;
        int halfDim = headDim / 2;
        float[] table = new float[seqLen * halfDim];
        int freqOffset = 0;
        for (int axis = 0; axis < axesDim.Length; axis++)
        {
            int numPairs = axesDim[axis] / 2;
            for (int k = 0; k < numPairs; k++)
            {
                double omega = 1.0 / Math.Pow(theta, (double)(2 * k) / axesDim[axis]);
                for (int s = 0; s < seqLen; s++)
                {
                    // Position ids from BuildPositionIds(txtSeqLen: 0, hPacked: seqLen, wPacked: 1):
                    // axis 0 always 0, axis 1 = row = s, axis 2 = col = 0.
                    double pos = axis == 1 ? s : 0;
                    double angle = pos * omega;
                    table[s * halfDim + freqOffset + k] = (float)(cos ? Math.Cos(angle) : Math.Sin(angle));
                }
            }
            freqOffset += numPairs;
        }
        return table;
    }
}
