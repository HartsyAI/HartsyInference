using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Row S.16: an F32 call with the BF16 grant runs cuDNN on BF16 copies of Q, K and V. BF16 keeps F32's range, so a V that
/// S.5's saturating F16 cast clips stays finite and accurate; the cast rounds to nearest even bit for bit; a refused BF16 signature
/// leaves the F16 path alone; and copies that do not fit leave the call to the tiled row.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaBf16CastTests
{
    private readonly ITestOutputHelper _output;
    public CudnnSdpaBf16CastTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static bool CudnnAvailable(ITestOutputHelper output)
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return false; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa) { output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return false; }
        return true;
    }

    private static Tensor Filled(Func<long, float> value, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = value(i);
        return t;
    }

    private static Tensor Uniform(int seed, params long[] dims)
    {
        Random rng = new(seed);
        return Filled(_ => (float)(rng.NextDouble() * 2 - 1), dims);
    }

    /// <summary>F32 → BF16 by the bit pattern: keep the top half, and round it up past the halfway point or on a tie to an odd
    /// keep. A NaN keeps its sign and top payload bits with the quiet bit set.</summary>
    private static ushort RoundBf16(uint bits)
    {
        if ((bits & 0x7FFF_FFFFu) > 0x7F80_0000u) return (ushort)((bits >> 16) | 0x0040u);
        uint keep = bits >> 16, rest = bits & 0xFFFFu;
        return (ushort)(rest > 0x8000u || (rest == 0x8000u && (keep & 1u) != 0) ? keep + 1 : keep);
    }

    private static float Bf16(float x) => BitConverter.UInt32BitsToSingle((uint)RoundBf16(BitConverter.SingleToUInt32Bits(x)) << 16);

    private static AttentionCall Granted(Tensor o, Tensor q, Tensor k, Tensor v, float scale) =>
        AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, scale, false) with { Numerics = AttentionNumerics.Bf16Storage };

    private static float[] Oracle(Tensor q, Tensor k, Tensor v, float scale)
    {
        using Tensor expected = new(q.Shape, DType.F32);
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(expected, q, k, v, null, scale, false));
        return AttnInvoker.ReadF32(expected);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ValuesPastTheF16Range_StayFiniteAndWithinTheBf16Bar_WhereTheSaturatingRowClips()
    {
        if (!CudnnAvailable(_output)) return;
        const int heads = 2, s = 128, d = 128;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(5);
        using Tensor q = Uniform(3, 1, heads, s, d), k = Uniform(4, 1, heads, s, d);
        // Every ninth V element is 1e6 and every ninth from the fifth -3e5, far past F16's 65504.
        using Tensor v = Filled(i => i % 9 == 0 ? 1e6f : i % 9 == 4 ? -3e5f : (float)(rng.NextDouble() * 2 - 1), 1, heads, s, d);
        using Tensor qb = Filled(i => Bf16(((float*)q.DataPointer)[i]), 1, heads, s, d);
        using Tensor kb = Filled(i => Bf16(((float*)k.DataPointer)[i]), 1, heads, s, d);
        using Tensor vb = Filled(i => Bf16(((float*)v.DataPointer)[i]), 1, heads, s, d);
        float[] stored = Oracle(qb, kb, vb, scale), exact = Oracle(q, k, v, scale);
        KnobStore.Set(EngineKnobs.SdpaCudnn, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            IBackend b = backend;
            using Tensor o = new(new TensorShape(1, heads, s, d), DType.F32), saturated = new(new TensorShape(1, heads, s, d), DType.F32);
            b.Attention(Granted(o, q, k, v, scale));
            backend.Sync();
            Assert.Equal("S.16", CudaBackend.LastAttentionRoute);
            float[] bf16 = AttnInvoker.ReadF32(o);
            Assert.True(bf16.All(float.IsFinite), "a V value past the F16 range made the BF16 row's output non-finite");
            (double rel, double cos, _) = AttnAssert.Measure(bf16, stored, d);
            Assert.True(rel <= 8e-3, $"the BF16 row is {rel:E2} from the oracle on its BF16 inputs (bar 8e-3)");

            b.ScaledDotProductAttention(saturated, q, k, v, null, scale, allowF16: true);
            backend.Sync();
            Assert.Equal("S.5", CudaBackend.LastAttentionRoute);
            float[] clipped = AttnInvoker.ReadF32(saturated);
            Assert.True(clipped.All(float.IsFinite), "the saturating row produced a non-finite output");
            double relClipped = AttnAssert.Differs(clipped, exact, d, 0.1, "after the saturating F16 cast clipped V");
            double relCaller = AttnAssert.Measure(bf16, exact, d).RelL2;
            _output.WriteLine($"S.16 relL2 {rel:E2} (min cos {cos:F7}) vs the oracle on BF16 inputs, {relCaller:E2} vs the F32 inputs; "
                + $"S.5 relL2 {relClipped:E2}");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ANaNInValue_StaysNaN()
    {
        if (!CudnnAvailable(_output)) return;
        const int heads = 1, s = 128, d = 128;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Uniform(3, 1, heads, s, d), k = Uniform(4, 1, heads, s, d);
        using Tensor v = Filled(i => i == 3 * d + 7 ? float.NaN : 0.5f, 1, heads, s, d);   // one NaN in V at key row 3
        KnobStore.Set(EngineKnobs.SdpaCudnn, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            using Tensor o = new(new TensorShape(1, heads, s, d), DType.F32);
            ((IBackend)backend).Attention(Granted(o, q, k, v, scale));
            backend.Sync();
            Assert.Equal("S.16", CudaBackend.LastAttentionRoute);
            int nan = AttnInvoker.ReadF32(o).Count(float.IsNaN);
            _output.WriteLine($"NaN outputs: {nan} of {o.ElementCount}");
            Assert.True(nan > 0, "a NaN in V did not reach the BF16 row's output: the cast replaced it with a finite value");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
        }
    }

    /// <summary>The row's F32 → BF16 kernel against the managed rounding on ties, subnormals, the top binade, ±Inf, NaN and a
    /// million random patterns, and its BF16 → F32 kernel on every BF16 pattern.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TheCasts_RoundToNearestEvenAndWidenExactly_BitForBit()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        uint[] edges =
        [
            0x3F80_8000u, 0x3F81_8000u, 0xBF80_8000u, 0xBF81_8000u,   // ties: an even keep stays, an odd keep rounds up
            0x3F80_7FFFu, 0x3F80_8001u,                               // either side of a tie
            0x0000_0001u, 0x0000_8000u, 0x0001_8000u, 0x007F_FFFFu, 0x8000_8000u, 0x807F_FFFFu,   // subnormals, a carry into the normals
            0x7F7F_7FFFu, 0x7F7F_8000u, 0x7F7F_FFFFu, 0xFF7F_FFFFu,   // the top binade: finite below the tie, infinite from it
            0x7F80_0000u, 0xFF80_0000u, 0x0000_0000u, 0x8000_0000u,   // ±Inf, ±0
            0x7F80_0001u, 0x7FC0_0000u, 0xFFFF_FFFFu, 0x7FBF_FFFFu, 0xFF80_0001u,   // NaNs, payloads in the low half only included
        ];
        Random rng = new(11);
        uint[] bits = new uint[edges.Length + (1 << 20)];
        edges.CopyTo(bits, 0);
        for (int i = edges.Length; i < bits.Length; i++) bits[i] = (uint)rng.NextInt64(0, 1L << 32);
        using Tensor f32 = new(new TensorShape(bits.Length), DType.F32), narrowed = new(new TensorShape(bits.Length), DType.BF16);
        bits.AsSpan().CopyTo(new Span<uint>(f32.DataPointer, bits.Length));
        using Tensor every = new(new TensorShape(1 << 16), DType.BF16), widened = new(new TensorShape(1 << 16), DType.F32);
        Span<ushort> patterns = new(every.DataPointer, 1 << 16);
        for (int i = 0; i < patterns.Length; i++) patterns[i] = (ushort)i;

        using CudaBackend backend = new(0, PtxDir());
        backend.CastToBf16(narrowed, f32);
        backend.CastToF32(widened, every);
        backend.Sync();

        ReadOnlySpan<ushort> got = narrowed.AsReadOnlySpan<ushort>();
        List<string> wrong = [];
        for (int i = 0; i < bits.Length; i++)
            if (got[i] != RoundBf16(bits[i]))
                wrong.Add($"0x{bits[i]:X8} -> 0x{got[i]:X4}, expected 0x{RoundBf16(bits[i]):X4}");
        Assert.True(wrong.Count == 0, $"{wrong.Count} of {bits.Length} narrowed patterns differ, first: {string.Join("; ", wrong.Take(5))}");
        ReadOnlySpan<uint> wide = widened.AsReadOnlySpan<uint>();
        for (int i = 0; i < wide.Length; i++)
            Assert.True(wide[i] == (uint)i << 16, $"BF16 0x{i:X4} widened to 0x{wide[i]:X8}");
    }

    /// <summary>A structural refusal of every BF16 cast plan gives up the BF16 signature only: the F16 plans at the same head dim and
    /// the saturating F32 row keep cuDNN.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ARefusedBf16Signature_LeavesTheF16D128PathEligible()
    {
        if (!CudnnAvailable(_output)) return;
        const int heads = 4, d = 128;
        float scale = 1f / MathF.Sqrt(d);
        KnobStore.Set(EngineKnobs.SdpaCudnn, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            IBackend b = backend;
            int injected = 0;
            backend.TestCudnnSdpaFaultInjector = signature =>
            {
                if (signature.Dtype != CudnnSdpa.SdpaDtype.BF16) return null;
                injected++;
                return new CudnnStatusException(3000, "injected structural failure");   // NOT_SUPPORTED: permanent for the shape
            };
            string Run(DType dtype, int s, bool allowF16, bool grant)
            {
                Tensor Operand(int seed)
                {
                    Tensor f32 = Uniform(seed, 1, heads, s, d);
                    if (dtype == DType.F32) return f32;
                    try { return f32.CastTo(dtype); } finally { f32.Dispose(); }
                }
                using Tensor q = Operand(1), k = Operand(2), v = Operand(3), o = new(new TensorShape(1, heads, s, d), dtype);
                AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, scale, allowF16);
                b.Attention(grant ? call with { Numerics = AttentionNumerics.Bf16Storage } : call);
                backend.Sync();
                return CudaBackend.LastAttentionRoute!;
            }

            for (int i = 0; i < CudnnSdpaHealth.RefusalsToDisable; i++)
                Assert.Equal("S.14", Run(DType.F32, 64 + 8 * i, allowF16: false, grant: true));
            Assert.Equal(CudnnSdpaHealth.RefusalsToDisable, injected);
            Assert.Equal("S.14", Run(DType.F32, 256, allowF16: false, grant: true));
            Assert.Equal(CudnnSdpaHealth.RefusalsToDisable, injected);
            Assert.Equal("S.3", Run(DType.F16, 256, allowF16: true, grant: false));
            Assert.Equal("S.5", Run(DType.F32, 256, allowF16: true, grant: false));
            Assert.True(backend.CudnnSdpaEngaged);
            Assert.Contains(backend.CudnnSdpaDiagnostics, kv => kv.Key.Dtype == CudnnSdpa.SdpaDtype.BF16 && kv.Value.Permanent);
            Assert.DoesNotContain(backend.CudnnSdpaDiagnostics, kv => kv.Key.Dtype == CudnnSdpa.SdpaDtype.F16);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
        }
    }

    /// <summary>With free VRAM too small for the BF16 copies the row declines and the memory-bounded tiled row serves the call.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void CopiesThatDoNotFit_FallThroughToTheTiledRow()
    {
        if (!CudnnAvailable(_output)) return;
        const int heads = 4, s = 256, d = 128;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Uniform(1, 1, heads, s, d), k = Uniform(2, 1, heads, s, d), v = Uniform(3, 1, heads, s, d);
        float[] expected = Oracle(q, k, v, scale);
        KnobStore.Set(EngineKnobs.SdpaCudnn, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            IBackend b = backend;
            using Tensor o = new(new TensorShape(1, heads, s, d), DType.F32);
            // A budget the size of the four BF16 copies: a row may use half, which neither the copies nor the F32 scores fit.
            backend.TestFreeVramOverride = () => (nuint)(sizeof(ushort) * 4L * heads * s * d);
            b.Attention(Granted(o, q, k, v, scale));
            backend.Sync();
            Assert.Equal("S.9", CudaBackend.LastAttentionRoute);
            AttnAssert.Agrees(AttnInvoker.ReadF32(o), expected, d, AttnTolerance.For(AttnAchieved.Tf32), "on the tiled row");

            backend.TestFreeVramOverride = null;
            b.Attention(Granted(o, q, k, v, scale));
            backend.Sync();
            Assert.Equal("S.16", CudaBackend.LastAttentionRoute);
            (double rel, _, _) = AttnAssert.Agrees(AttnInvoker.ReadF32(o), expected, d, AttnTolerance.For(AttnAchieved.Bf16Io), "on the BF16 row");
            _output.WriteLine($"S.16 relL2 {rel:E2} vs the F32 inputs");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
        }
    }
}
