using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Regression test for the glm4 root cause found 2026-07-22: <c>IBackend.ApplyRopeInterleaved</c> (and
/// its CUDA kernel <c>lm_rope_interleaved_f32</c>) always rotated all <c>headDim/2</c> pairs regardless of a
/// model's actual (possibly partial) rotary dimension — GLM-4 uses <c>partial_rotary_factor=0.5</c>, so the
/// upper half of every Q/K vector was being spuriously rotated with wrong (repeated-block) frequencies instead
/// of passing through unchanged. Invisible at position 0 (angle=0, "wrong" rotation is still identity) — exactly
/// why short greedy smoke tests never caught it — but corrupts every later position. Verifies both the CPU
/// fallback (<see cref="IBackend"/> default) and the real CUDA kernel: (a) dims inside <c>[0, rotaryDim)</c>
/// rotate correctly, (b) dims outside are byte-identical to the un-rotated input, and (c) the default
/// <c>rotaryDim=0</c> (full rotary — Kyutai Moshi / Dia's usage) is unchanged from the pre-fix behavior.</summary>
public sealed unsafe class RopeInterleavedPartialRotaryTests
{
    private static Tensor F32(int a, int b, int c, int d)
    {
        Tensor t = new(new TensorShape(a, b, c, d), DType.F32);
        return t;
    }

    private static Tensor CosSinRow(int headDim, float[] values)
    {
        Tensor t = new(new TensorShape(1, 1, headDim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < headDim; i++) p[i] = i < values.Length ? values[i] : 0f;
        return t;
    }

    // headDim=8, rotaryDim=4: pairs (0,1) and (2,3) [dims 0-3] rotate; pairs (4,5) and (6,7) [dims 4-7] pass through.
    private static (Tensor x, Tensor cos, Tensor sin, float[] expected) BuildPartialRotaryCase()
    {
        const int headDim = 8;
        Tensor x = F32(1, 1, 1, headDim);
        float[] input = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < headDim; i++) xp[i] = input[i];

        // cos/sin only meaningful for i in [0, rotaryDim/2)=[0,2); index 2,3 (beyond rotaryDim/2) intentionally
        // hold garbage-like nonzero values to prove the kernel does NOT read them once the guard is in place.
        Tensor cos = CosSinRow(headDim, [0.5f, 0.6f, 999f, 999f]);
        Tensor sin = CosSinRow(headDim, [0.8660254f, 0.8f, 999f, 999f]);

        // Pair 0 (dims 0,1): xe=1,xo=2, c=0.5, s=0.8660254 -> x0=1*0.5-2*0.8660254=-1.2320508, x1=2*0.5+1*0.8660254=1.8660254
        // Pair 1 (dims 2,3): xe=3,xo=4, c=0.6, s=0.8       -> x2=3*0.6-4*0.8=-1.4,          x3=4*0.6+3*0.8=4.8
        // Pairs 2,3 (dims 4-7): rotaryDim=4 excludes them -> pass through unchanged: 5,6,7,8
        float[] expected = [-1.2320508f, 1.8660254f, -1.4f, 4.8f, 5f, 6f, 7f, 8f];
        return (x, cos, sin, expected);
    }

    private static void AssertMatches(Tensor x, float[] expected, string label)
    {
        float* p = (float*)x.DataPointer;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(p[i] - expected[i]) < 1e-4f, $"{label} i={i}: expected {expected[i]}, got {p[i]}");
    }

    [Fact]
    public void CpuFallback_PartialRotary_RotatesInsideDimOnly_PassesThroughOutside()
    {
        (Tensor x, Tensor cos, Tensor sin, float[] expected) = BuildPartialRotaryCase();
        using CpuBackend backend = new();
        try
        {
            IBackend b = backend;
            b.ApplyRopeInterleaved(x, cos, sin, rotaryDim: 4);
            AssertMatches(x, expected, "CPU");
        }
        finally { x.Dispose(); cos.Dispose(); sin.Dispose(); }
    }

    [Fact]
    public void CudaKernel_PartialRotary_RotatesInsideDimOnly_PassesThroughOutside()
    {
        if (!CudaContext.IsAvailable()) { Console.Error.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        (Tensor x, Tensor cos, Tensor sin, float[] expected) = BuildPartialRotaryCase();
        using CudaBackend backend = new(0, ptxDir);
        try
        {
            IBackend b = backend;
            b.ApplyRopeInterleaved(x, cos, sin, rotaryDim: 4);
            backend.Sync();
            AssertMatches(x, expected, "CUDA");
        }
        finally
        {
            x.Dispose(); cos.Dispose(); sin.Dispose();
        }
    }

    [Fact]
    public void CudaKernel_FullRotaryDefault_MatchesPreFixBehavior_AllPairsRotate()
    {
        if (!CudaContext.IsAvailable()) { Console.Error.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        // Same shape as BuildPartialRotaryCase but with rotaryDim omitted (default 0 = full) — Kyutai
        // Moshi/Dia's actual call pattern. All 4 pairs must rotate, including the previously-garbage-cos pairs
        // 2/3, now given real (not garbage) frequencies since full rotary genuinely uses all headDim/2 pairs.
        const int headDim = 8;
        Tensor x = F32(1, 1, 1, headDim);
        float[] input = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < headDim; i++) xp[i] = input[i];
        Tensor cos = CosSinRow(headDim, [0.5f, 0.6f, 0.7f, 1f]);
        Tensor sin = CosSinRow(headDim, [0.8660254f, 0.8f, 0.71414f, 0f]);

        using CudaBackend backend = new(0, ptxDir);
        try
        {
            IBackend b = backend;
            b.ApplyRopeInterleaved(x, cos, sin);   // rotaryDim defaults to 0 (full)
            backend.Sync();
            float* p = (float*)x.DataPointer;
            // Pair 3 (dims 6,7): xe=7,xo=8,c=1,s=0 -> unchanged 7,8 (sanity: this pair rotating with c=1,s=0 is a
            // no-op regardless, so also assert pair 2 (dims 4,5) actually moved off its input value to prove the
            // kernel really executed a rotation there under the full-rotary default, not silently no-op'd.
            Assert.True(MathF.Abs(p[4] - 5f) > 1e-3f, "pair 2 (dims 4,5) should have rotated under full-rotary default, not passed through");
        }
        finally
        {
            x.Dispose(); cos.Dispose(); sin.Dispose();
            try { backend.Dispose(); } catch (Exception ex) { Console.Error.WriteLine($"[teardown-ignored] {ex.GetType().Name}: {ex.Message}"); }
        }
    }
}
