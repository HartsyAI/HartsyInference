using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The graph-decode fusions of Kernels/lm/lm_decode_fused.cu against the launch chains they replace: each must match bit for bit,
/// so every test runs the op with its knob off (the unfused composition) and on (the fused kernel) and compares raw floats and bytes.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaDecodeFusionTests
{
    private readonly ITestOutputHelper _output;
    public CudaDecodeFusionTests(ITestOutputHelper output) => _output = output;

    private readonly record struct Q8(byte[] Q, float[] D, float[] S);
    private sealed record NormRun(float[] Norm, float[] Resid, Q8 Sc);
    private sealed record ScatterRun(float[] Q, byte[] K, byte[] V);

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor Random(Random rng, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)((rng.NextDouble() - 0.5) * 2.0);
        return t;
    }

    private static float[] Floats(Tensor t)
    {
        float[] r = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    /// <summary>The Q8_1 sidecar registered on <paramref name="t"/> (k = <paramref name="k"/>), read back as (q, scale, sum).</summary>
    private static Q8 Sidecar(Tensor t, int k)
    {
        Assert.True(GpuTransferHelper.TryGetSidecar(t, k, out ulong xq, out ulong xd, out ulong xs), "the op must publish a Q8_1 sidecar");
        byte[] q = new byte[k];
        float[] d = new float[k / 32], s = new float[k / 32];
        fixed (byte* pq = q) CudaMemory.CopyDeviceToHost(pq, xq, (nuint)k);
        fixed (float* pd = d) CudaMemory.CopyDeviceToHost(pd, xd, (nuint)(d.Length * sizeof(float)));
        fixed (float* ps = s) CudaMemory.CopyDeviceToHost(ps, xs, (nuint)(s.Length * sizeof(float)));
        return new Q8(q, d, s);
    }

    /// <summary>quantize_activation_q8_1_f32 on the host: per 32-block scale amax/127, round half to even, clamp, int sum.</summary>
    private static Q8 HostQ8(float[] x)
    {
        byte[] q = new byte[x.Length];
        float[] d = new float[x.Length / 32], s = new float[x.Length / 32];
        for (int b = 0; b < d.Length; b++)
        {
            float amax = 0f;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(x[b * 32 + i]));
            float scale = amax / 127f, inv = scale > 0f ? 1f / scale : 0f;
            int sum = 0;
            for (int i = 0; i < 32; i++)
            {
                int v = Math.Clamp((int)MathF.Round(x[b * 32 + i] * inv, MidpointRounding.ToEven), -127, 127);
                q[b * 32 + i] = (byte)(sbyte)v;
                sum += v;
            }
            d[b] = scale;
            s[b] = sum;
        }
        return new Q8(q, d, s);
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(512)]
    [InlineData(768)]
    [InlineData(2048)]
    [InlineData(2560)]
    [InlineData(3072)]
    [InlineData(5120)]
    [InlineData(8192)]
    public void WideRmsNormQ8_IsBitIdenticalToTheReferenceKernels(int dim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(dim);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor weight = Random(rng, dim);
        using Tensor a = Random(rng, 1, 1, dim), b = Random(rng, 1, 1, dim);
        for (int i = 0; i < dim; i += 97) ((float*)a.DataPointer)[i] *= 40f;   // a few large values stretch the scales

        NormRun Run(bool add, bool reference, bool wide)
        {
            cuda.Kernels!.ForceReferenceNorm = reference;
            KnobStore.Set(EngineKnobs.NormWide, wide);
            using Tensor norm = new(new TensorShape(1, 1, dim), DType.F32);
            using Tensor resid = new(new TensorShape(1, 1, dim), DType.F32);
            if (add) cuda.AddRmsNormEmitQ8(resid, norm, a, b, weight, 1e-6f);
            else cuda.RmsNormEmitQ8(norm, a, weight, 1e-6f);
            cuda.Sync();
            Q8 sc = Sidecar(norm, dim);   // before any host read: reading the tensor back releases its device copy and sidecar
            return new NormRun(Floats(norm), add ? Floats(resid) : [], sc);
        }
        try
        {
            foreach (bool add in new[] { false, true })
            {
                NormRun reference = Run(add, reference: true, wide: false);
                NormRun fast = Run(add, reference: false, wide: false);
                NormRun wide = Run(add, reference: false, wide: true);
                foreach (NormRun other in new[] { fast, wide })
                {
                    Assert.Equal(reference.Norm, other.Norm);
                    Assert.Equal(reference.Resid, other.Resid);
                    Assert.Equal(reference.Sc.Q, other.Sc.Q);
                    Assert.Equal(reference.Sc.D, other.Sc.D);
                    Assert.Equal(reference.Sc.S, other.Sc.S);
                }
            }
        }
        finally
        {
            cuda.Kernels!.ForceReferenceNorm = false;
            KnobStore.Clear(EngineKnobs.NormWide);
        }
        _output.WriteLine($"wide RMSNorm Q8 == reference bit for bit (dim {dim})");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(2048, 8, false, false)]   // OLMoE
    [InlineData(2048, 8, true, true)]     // gated shared expert (Qwen2-MoE shape)
    [InlineData(2560, 6, true, false)]
    [InlineData(4096, 4, false, false)]
    public void MoeCombineAddRmsNormQ8_IsBitIdenticalToTheComposition(int dim, int topk, bool withShared, bool withGate)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(dim + topk);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor weight = Random(rng, dim);
        using Tensor a = Random(rng, 1, 1, dim);
        using Tensor slots = Random(rng, 1, topk, dim);
        using Tensor topkWeight = Random(rng, 1, topk);
        using Tensor? shared = withShared ? Random(rng, 1, 1, dim) : null;
        using Tensor? gate = withGate ? Random(rng, 1, 1, 1) : null;

        NormRun Run(bool fused)
        {
            KnobStore.Set(EngineKnobs.DecodeResidualFold, fused);
            using Tensor norm = new(new TensorShape(1, 1, dim), DType.F32);
            using Tensor resid = new(new TensorShape(1, 1, dim), DType.F32);
            cuda.MoeCombineAddRmsNormEmitQ8(resid, norm, a, slots, topkWeight, shared, gate, topk, weight, 1e-6f);
            cuda.Sync();
            Q8 sc = Sidecar(norm, dim);
            return new NormRun(Floats(norm), Floats(resid), sc);
        }
        try
        {
            NormRun composed = Run(fused: false);
            NormRun fused = Run(fused: true);
            Assert.Equal(composed.Resid, fused.Resid);
            Assert.Equal(composed.Norm, fused.Norm);
            Assert.Equal(composed.Sc.Q, fused.Sc.Q);
            Assert.Equal(composed.Sc.D, fused.Sc.D);
            Assert.Equal(composed.Sc.S, fused.Sc.S);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.DecodeResidualFold);
        }
        _output.WriteLine($"MoE combine + add + RMSNorm Q8 == composition (dim {dim}, topk {topk}, shared {withShared}, gate {withGate})");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false, false, false)]   // [q|k|v] in one buffer, F32 cache (OLMoE layers whose q, k and v share a type)
    [InlineData(true, false, false)]    // [q|k] + separate v (mixed-type layers)
    [InlineData(false, true, false)]    // F16 cache
    [InlineData(true, false, true)]     // interleaved rope
    public void QkNormFullRopeScatter_IsBitIdenticalToTheComposition(bool separateV, bool f16Cache, bool interleaved)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int hq = 16, hkv = 4, d = 128, maxSeq = 64, pos = 23;
        Random rng = new(separateV ? 11 : 12);
        using CudaBackend cuda = new(0, PtxDir());
        ulong posBuf = CudaMemory.Allocate(2 * sizeof(int));
        int* hostPos = stackalloc int[2] { pos + 1, pos };
        CudaMemory.CopyHostToDevice(posBuf, hostPos, 2 * sizeof(int));
        using Tensor cos = Random(rng, maxSeq, d), sin = Random(rng, maxSeq, d);
        using Tensor qk = Random(rng, 1, 1, (hq + hkv * (separateV ? 1 : 2)) * d);
        using Tensor v = Random(rng, 1, hkv, 1, d);
        using Tensor qNorm = Random(rng, hq * d), kNorm = Random(rng, hkv * d);
        DType cacheType = f16Cache ? DType.F16 : DType.F32;
        int elem = f16Cache ? 2 : 4;

        ScatterRun Run(bool fused)
        {
            KnobStore.Set(EngineKnobs.QknormFullScatter, fused);
            using Tensor kc = new(new TensorShape(1, hkv, maxSeq, d), cacheType), vc = new(new TensorShape(1, hkv, maxSeq, d), cacheType);
            using Tensor qo = new(new TensorShape(1, hq, 1, d), DType.F32);
            cuda.QkNormFullRopeScatterDecodeStep(qo, kc, vc, qk, separateV ? v : null, qNorm, kNorm, 1e-6f, cos, sin, hq, hkv, d, d,
                interleaved, posBuf);
            cuda.Sync();
            byte[] kb = new byte[kc.ElementCount * elem], vb = new byte[vc.ElementCount * elem];
            new ReadOnlySpan<byte>((void*)kc.DataPointer, kb.Length).CopyTo(kb);
            new ReadOnlySpan<byte>((void*)vc.DataPointer, vb.Length).CopyTo(vb);
            return new ScatterRun(Floats(qo), kb, vb);
        }
        try
        {
            ScatterRun composed = Run(fused: false);
            ScatterRun fused = Run(fused: true);
            Assert.Equal(composed.Q, fused.Q);
            Assert.Equal(composed.K, fused.K);
            Assert.Equal(composed.V, fused.V);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.QknormFullScatter);
            CudaMemory.Free(posBuf);
        }
        _output.WriteLine($"full-width QK-norm scatter == composition (separateV {separateV}, f16 {f16Cache}, interleaved {interleaved})");
    }

    [Fact]
    public void QkNormFullFusedEligible_RejectsBuffersTheKernelWouldOverrun()
    {
        const int hq = 16, hkv = 4, d = 128;
        Random rng = new(5);
        using Tensor qk = Random(rng, 1, 1, (hq + hkv * 2) * d), qkShort = Random(rng, 1, 1, (hq + hkv) * d);
        using Tensor v = Random(rng, 1, hkv, 1, d), vShort = Random(rng, 1, 1, 1, d);
        using Tensor qNorm = Random(rng, hq * d), kNorm = Random(rng, hkv * d), qNormShort = Random(rng, d);
        using Tensor qOut = new(new TensorShape(1, hq, 1, d), DType.F32);
        using Tensor qOutHalf = new(new TensorShape(1, hq, 1, d), DType.F16);
        using Tensor qOutSmall = new(new TensorShape(1, hq - 1, 1, d), DType.F32);
        try
        {
            KnobStore.Set(EngineKnobs.QknormFullScatter, true);
            Assert.True(CudaBackend.QkNormFullFusedEligible(qOut, qk, null, qNorm, kNorm, hq, hkv, d, 1));
            Assert.True(CudaBackend.QkNormFullFusedEligible(qOut, qkShort, v, qNorm, kNorm, hq, hkv, d, 1));
            Assert.False(CudaBackend.QkNormFullFusedEligible(qOutHalf, qk, null, qNorm, kNorm, hq, hkv, d, 1));      // F16 output: kernel writes floats
            Assert.False(CudaBackend.QkNormFullFusedEligible(qOutSmall, qk, null, qNorm, kNorm, hq, hkv, d, 1));     // too small
            Assert.False(CudaBackend.QkNormFullFusedEligible(qOut, qkShort, null, qNorm, kNorm, hq, hkv, d, 1));     // no room for v
            Assert.False(CudaBackend.QkNormFullFusedEligible(qOut, qkShort, vShort, qNorm, kNorm, hq, hkv, d, 1));   // separate v too small
            Assert.False(CudaBackend.QkNormFullFusedEligible(qOut, qk, null, qNormShort, kNorm, hq, hkv, d, 1));     // norm weight too narrow
            Assert.False(CudaBackend.QkNormFullFusedEligible(qOut, qk, null, qNorm, kNorm, hq, hkv, d, 0));          // no device position
            KnobStore.Set(EngineKnobs.QknormFullScatter, false);
            Assert.False(CudaBackend.QkNormFullFusedEligible(qOut, qk, null, qNorm, kNorm, hq, hkv, d, 1));
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.QknormFullScatter);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(32, 8, 128, 300, 1024)]   // Qwen3-4B decode shape, grouped-query kernel
    [InlineData(16, 16, 128, 40, 4096)]   // OLMoE, MHA
    [InlineData(8, 2, 64, 77, 512)]
    public void AttentionCombineQ8_OutputUnchangedAndSidecarMatchesQuantize(int hq, int hkv, int d, int kvLen, int capacity)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(hq * d + kvLen);
        using CudaBackend cuda = new(0, PtxDir());
        ulong posBuf = CudaMemory.Allocate(2 * sizeof(int));
        int* hostPos = stackalloc int[2] { kvLen, kvLen - 1 };
        CudaMemory.CopyHostToDevice(posBuf, hostPos, 2 * sizeof(int));
        using Tensor q = Random(rng, 1, hq, 1, d);
        using Tensor k = Random(rng, 1, hkv, capacity, d), v = Random(rng, 1, hkv, capacity, d);
        float scale = 1f / MathF.Sqrt(d);

        float[] Run(bool q8, out Q8? sidecar)
        {
            KnobStore.Set(EngineKnobs.AttnCombineQ8, q8);
            using Tensor o = new(new TensorShape(1, hq, 1, d), DType.F32);
            cuda.FlashAttentionDev(o, q, k, v, 0, hq / hkv, causal: true, 0, scale, posBuf);
            cuda.Sync();
            sidecar = q8 ? Sidecar(o, hq * d) : null;
            if (!q8) Assert.False(GpuTransferHelper.TryGetSidecar(o, hq * d, out _, out _, out _));
            return Floats(o);
        }
        try
        {
            float[] plain = Run(q8: false, out _);
            float[] withQ8 = Run(q8: true, out Q8? sc);
            Assert.Equal(plain, withQ8);
            Q8 expected = HostQ8(withQ8);
            Assert.Equal(expected.Q, sc!.Value.Q);
            Assert.Equal(expected.D, sc.Value.D);
            Assert.Equal(expected.S, sc.Value.S);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.AttnCombineQ8);
            CudaMemory.Free(posBuf);
        }
        _output.WriteLine($"attention combine Q8: output unchanged, sidecar == quantize (hq {hq}, hkv {hkv}, d {d}, kvLen {kvLen})");
    }
}
