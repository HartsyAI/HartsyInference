using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The probed cuDNN variants through the production entry points, each against a float64 reference and with its route asserted.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaVariantTests
{
    internal const double F16Tolerance = 3e-3, Bf16Tolerance = 8e-3;   // ATTENTION_CONTRACT §8

    private readonly ITestOutputHelper _output;
    public CudnnSdpaVariantTests(ITestOutputHelper output) => _output = output;

    internal static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir)
            ? dir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private bool Skip()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return true; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return true; }
        return false;
    }

    // ── host helpers: head-major is [b, h, s, d], token-major [b, s, h, d] ──

    internal static float[] Gauss(Random rng, long n, float sigma = 1f)
    {
        float[] x = new float[n];
        for (long i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            x[i] = sigma * (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return x;
    }

    internal static Tensor Make(float[] values, TensorShape shape, DType dtype)
    {
        Tensor f32 = new(shape, DType.F32);
        values.AsSpan().CopyTo(new Span<float>(f32.DataPointer, values.Length));
        if (dtype == DType.F32) return f32;
        try { return f32.CastTo(dtype); }
        finally { f32.Dispose(); }
    }

    internal static float[] Read(Tensor t)
    {
        using Tensor f32 = t.CastTo(DType.F32);
        return new Span<float>(f32.DataPointer, (int)f32.ElementCount).ToArray();
    }

    /// <summary>What a <paramref name="dtype"/> tensor holds for <paramref name="values"/>, so the reference sees what the kernel read.</summary>
    internal static float[] Quantize(float[] values, DType dtype)
    {
        using Tensor t = Make(values, new TensorShape(values.Length), dtype);
        return Read(t);
    }

    internal static float[] Transpose(float[] x, int b, int h, int s, int d)   // [b,h,s,d] <-> [b,s,h,d]: swaps the two middle axes
    {
        float[] y = new float[x.Length];
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
                for (int si = 0; si < s; si++)
                    Array.Copy(x, (((long)bi * h + hi) * s + si) * d, y, (((long)bi * s + si) * h + hi) * d, d);
        return y;
    }

    /// <summary>float64 softmax(scale·QKᵀ + bias)·V over head-major operands; the bias <c>[biasB, biasH, biasSq, skv]</c> broadcasts extents of 1.</summary>
    internal static float[] Reference(float[] q, float[] k, float[] v, float[]? bias, int b, int hq, int hkv, int sq, int skv, int d,
        float scale, int biasB = 1, int biasH = 1, int biasSq = 1)
    {
        float[] o = new float[(long)b * hq * sq * d];
        int group = hq / hkv;
        Parallel.For(0, b * hq * sq, () => new double[skv], (item, _, p) =>
        {
            int i = item % sq, h = item / sq % hq, bi = item / (sq * hq), kh = h / group;
            long qBase = (((long)bi * hq + h) * sq + i) * d;
            double max = double.NegativeInfinity;
            for (int j = 0; j < skv; j++)
            {
                long kBase = (((long)bi * hkv + kh) * skv + j) * d;
                double dot = 0;
                for (int e = 0; e < d; e++) dot += (double)q[qBase + e] * k[kBase + e];
                double sc = dot * scale;
                if (bias is not null)
                    sc += bias[((((long)(biasB == 1 ? 0 : bi) * biasH + (biasH == 1 ? 0 : h)) * biasSq) + (biasSq == 1 ? 0 : i)) * skv + j];
                p[j] = sc;
                if (sc > max) max = sc;
            }
            double sum = 0;
            for (int j = 0; j < skv; j++) { p[j] = Math.Exp(p[j] - max); sum += p[j]; }
            for (int e = 0; e < d; e++)
            {
                double acc = 0;
                for (int j = 0; j < skv; j++) acc += p[j] * v[(((long)bi * hkv + kh) * skv + j) * d + e];
                o[qBase + e] = (float)(acc / sum);
            }
            return p;
        }, _ => { });
        return o;
    }

    internal static double RelL2(float[] got, float[] want)
    {
        double err = 0, norm = 0;
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]), $"non-finite output at {i}: {got[i]}");
            double diff = got[i] - want[i];
            err += diff * diff;
            norm += (double)want[i] * want[i];
        }
        return Math.Sqrt(err) / Math.Max(Math.Sqrt(norm), 1e-12);
    }

    internal static Tensor? BiasTensor(string kind, Random rng, int b, int hq, int sq, int skv, out int biasB, out int biasH, out int biasSq,
        out float[]? values)
    {
        (biasB, biasH, biasSq, int[] shape) = kind switch
        {
            "none" => (1, 1, 1, Array.Empty<int>()),
            "shared" => (1, 1, sq, new[] { 1, 1, sq, skv }),
            "key" => (1, 1, 1, new[] { 1, 1, 1, skv }),
            "perhead" => (1, hq, sq, new[] { 1, hq, sq, skv }),
            "rank3perhead" => (1, hq, sq, new[] { hq, sq, skv }),
            "perbatch" => (b, 1, sq, new[] { b, 1, sq, skv }),
            "batchhead" => (b, hq, sq, new[] { b, hq, sq, skv }),
            "batchkey" => (b, 1, 1, new[] { b, 1, 1, skv }),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        if (kind == "none") { values = null; return null; }
        values = Gauss(rng, shape.Aggregate(1L, (a, x) => a * x));
        return Make(values, new TensorShape([.. shape.Select(x => (long)x)]), DType.F32);
    }

    // ── grouped-query K/V ──

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(48, 12, 128, 128, "none")]
    [InlineData(48, 12, 128, 128, "key")]
    [InlineData(48, 12, 128, 128, "shared")]
    [InlineData(48, 12, 128, 128, "perhead")]
    [InlineData(16, 8, 256, 128, "none")]
    [InlineData(16, 8, 256, 64, "none")]
    public void GroupedTokenMajor_MatchesF32Reference(int hq, int hkv, int s, int d, string biasKind)
    {
        if (Skip()) return;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(hq * 1000 + d);
        float[] qHm = Quantize(Gauss(rng, (long)hq * s * d), DType.F16);
        float[] kHm = Quantize(Gauss(rng, (long)hkv * s * d), DType.F16);
        float[] vHm = Quantize(Gauss(rng, (long)hkv * s * d), DType.F16);
        using Tensor q = Make(Transpose(qHm, 1, hq, s, d), new TensorShape(1, s, hq, d), DType.F16);
        using Tensor k = Make(Transpose(kHm, 1, hkv, s, d), new TensorShape(1, s, hkv, d), DType.F16);
        using Tensor v = Make(Transpose(vHm, 1, hkv, s, d), new TensorShape(1, s, hkv, d), DType.F16);
        using Tensor? bias = BiasTensor(biasKind, rng, 1, hq, s, s, out int bB, out int bH, out int bS, out float[]? biasValues);
        using Tensor o = new(new TensorShape(1, s, hq * d), DType.F16);   // Krea2's rank-3 output

        using CudaBackend backend = new(0, PtxDir());
        ((IBackend)backend).ScaledDotProductAttentionTokenMajor(o, q, k, v, bias, hq, hkv, d, scale, allowF16: true);
        backend.Sync();

        float[] got = Transpose(Read(o), 1, s, hq, d);   // token-major back to head-major
        float[] want = Reference(qHm, kHm, vHm, biasValues, 1, hq, hkv, s, s, d, scale, bB, bH, bS);
        double rel = RelL2(got, want);
        _output.WriteLine($"{hq}/{hkv} d{d} s{s} bias={biasKind}: route {CudaBackend.LastAttentionRoute} relL2 {rel:E2}");
        Assert.Equal("G.1", CudaBackend.LastAttentionRoute);
        Assert.True(backend.CudnnSdpaEngaged, "the grouped call did not run on cuDNN");
        Assert.True(rel <= F16Tolerance, $"grouped token-major SDPA {hq}/{hkv} d{d} bias={biasKind} relL2 {rel:E2}");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(16, 8, 256, 128)]
    [InlineData(48, 12, 128, 128)]
    public void CausalPrefillGqa_NeedsNoWidening_MatchesF32Reference(int hq, int hkv, int s, int d)
    {
        if (Skip()) return;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(hq + s);
        using Tensor q = Make(Gauss(rng, (long)hq * s * d), new TensorShape(1, hq, s, d), DType.F32);
        using Tensor k = Make(Gauss(rng, (long)hkv * s * d), new TensorShape(1, hkv, s, d), DType.F32);
        using Tensor v = Make(Gauss(rng, (long)hkv * s * d), new TensorShape(1, hkv, s, d), DType.F32);
        using Tensor o = new(new TensorShape(1, hq, s, d), DType.F32);
        using Tensor want = new(new TensorShape(1, hq, s, d), DType.F32);
        AttentionReference.FlashAttention(want, q, k, v, s, hq / hkv, causal: true, qOffset: 0, scale);

        using CudaBackend backend = new(0, PtxDir());
        ((IBackend)backend).FlashAttention(o, q, k, v, s, hq / hkv, causal: true, 0, scale);
        backend.Sync();

        double rel = RelL2(Read(o), Read(want));
        _output.WriteLine($"{hq}/{hkv} causal d{d} s{s}: route {CudaBackend.LastAttentionRoute} relL2 {rel:E2}");
        Assert.Equal("F.2", CudaBackend.LastAttentionRoute);
        Assert.True(rel <= 5e-3, $"cuDNN causal prefill {hq}/{hkv} relL2 {rel:E2} (the F32 inputs narrow to F16 on the way in)");
    }

    // ── bf16, opted in through the call's numerics ──

    private static readonly AttentionNumerics GrantsBf16 = new() { QueryKey = AttentionFormats.Bf16, ProbValue = AttentionFormats.Bf16 };

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false, 8, 8, 256, 128, "none", "S.3")]
    [InlineData(true, 8, 8, 256, 128, "none", "M.2")]
    [InlineData(false, 8, 8, 256, 64, "shared", "S.3")]
    [InlineData(false, 8, 8, 256, 64, "perhead", "S.3")]
    [InlineData(true, 48, 12, 128, 128, "none", "G.1")]
    public void Bf16_WithTheGrant_RunsOnCudnnAndMatchesTheReference(bool tokenMajor, int hq, int hkv, int s, int d, string biasKind, string route)
    {
        if (Skip()) return;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(hq * 7 + d);
        float[] qHm = Quantize(Gauss(rng, (long)hq * s * d), DType.BF16);
        float[] kHm = Quantize(Gauss(rng, (long)hkv * s * d), DType.BF16);
        float[] vHm = Quantize(Gauss(rng, (long)hkv * s * d), DType.BF16);
        TensorShape qs = tokenMajor ? new(1, s, hq, d) : new(1, hq, s, d), ks = tokenMajor ? new(1, s, hkv, d) : new(1, hkv, s, d);
        using Tensor q = Make(tokenMajor ? Transpose(qHm, 1, hq, s, d) : qHm, qs, DType.BF16);
        using Tensor k = Make(tokenMajor ? Transpose(kHm, 1, hkv, s, d) : kHm, ks, DType.BF16);
        using Tensor v = Make(tokenMajor ? Transpose(vHm, 1, hkv, s, d) : vHm, ks, DType.BF16);
        using Tensor? bias = BiasTensor(biasKind, rng, 1, hq, s, s, out int bB, out int bH, out int bS, out float[]? biasValues);
        using Tensor o = new(qs, DType.BF16);
        float[] want = Reference(qHm, kHm, vHm, biasValues, 1, hq, hkv, s, s, d, scale, bB, bH, bS);

        using CudaBackend backend = new(0, PtxDir());
        AttentionCall granted = tokenMajor
            ? AttentionCall.FromTokenMajor(o, q, k, v, bias, hq, hkv, d, scale, false) with { Numerics = GrantsBf16 }
            : AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, scale, false) with { Numerics = GrantsBf16 };
        Assert.Equal(route == "G.1" ? "cudnn.tokenmajor.gqa" : route == "M.2" ? "cudnn.tokenmajor" : "cudnn.f16", backend.PlanAttention(in granted).RowId);
        ((IBackend)backend).Attention(in granted);
        backend.Sync();

        float[] got = Read(o);
        if (tokenMajor) got = Transpose(got, 1, s, hq, d);
        double rel = RelL2(got, want);
        _output.WriteLine($"bf16 {(tokenMajor ? "token" : "head")}-major {hq}/{hkv} d{d} bias={biasKind}: route {CudaBackend.LastAttentionRoute} relL2 {rel:E2}");
        Assert.Equal(route, CudaBackend.LastAttentionRoute);
        Assert.True(backend.CudnnSdpaEngaged);
        Assert.True(rel <= Bf16Tolerance, $"bf16 cuDNN SDPA relL2 {rel:E2}");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false, "S.12")]
    [InlineData(true, "M.3/S.12")]
    public void Bf16_WithoutTheGrant_KeepsItsExistingRoute(bool tokenMajor, string route)
    {
        if (Skip()) return;
        const int h = 4, s = 128, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(11);
        TensorShape shape = tokenMajor ? new(1, s, h, d) : new(1, h, s, d);
        using Tensor q = Make(Gauss(rng, (long)h * s * d), shape, DType.BF16);
        using Tensor k = Make(Gauss(rng, (long)h * s * d), shape, DType.BF16);
        using Tensor v = Make(Gauss(rng, (long)h * s * d), shape, DType.BF16);
        using Tensor o = new(shape, DType.BF16);
        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;
        if (tokenMajor) b.ScaledDotProductAttentionTokenMajor(o, q, k, v, null, h, d, scale, allowF16: true);
        else b.ScaledDotProductAttention(o, q, k, v, null, scale, allowF16: true);
        backend.Sync();
        Assert.Equal(route, CudaBackend.LastAttentionRoute);
        Assert.False(backend.CudnnSdpaEngaged, "a bf16 call without the grant ran on cuDNN");
    }

    // ── per-head and per-batch bias ──

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("perhead")]
    [InlineData("rank3perhead")]
    [InlineData("perbatch")]
    [InlineData("batchhead")]
    [InlineData("key")]
    [InlineData("batchkey")]
    public void Bias_MatchesF32Reference(string biasKind)
    {
        if (Skip()) return;
        const int b = 2, h = 8, s = 128, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(5);
        float[] qv = Quantize(Gauss(rng, (long)b * h * s * d), DType.F16);
        float[] kv = Quantize(Gauss(rng, (long)b * h * s * d), DType.F16);
        float[] vv = Quantize(Gauss(rng, (long)b * h * s * d), DType.F16);
        TensorShape shape = new(b, h, s, d);
        using Tensor q = Make(qv, shape, DType.F16);
        using Tensor k = Make(kv, shape, DType.F16);
        using Tensor v = Make(vv, shape, DType.F16);
        using Tensor? bias = BiasTensor(biasKind, rng, b, h, s, s, out int bB, out int bH, out int bS, out float[]? biasValues);
        using Tensor o = new(shape, DType.F16);

        using CudaBackend backend = new(0, PtxDir());
        ((IBackend)backend).ScaledDotProductAttention(o, q, k, v, bias, scale, allowF16: true);
        backend.Sync();

        double rel = RelL2(Read(o), Reference(qv, kv, vv, biasValues, b, h, h, s, s, d, scale, bB, bH, bS));
        _output.WriteLine($"bias={biasKind}: route {CudaBackend.LastAttentionRoute} relL2 {rel:E2}");
        Assert.Equal("S.3", CudaBackend.LastAttentionRoute);
        Assert.True(rel <= F16Tolerance, $"cuDNN SDPA with a {biasKind} bias relL2 {rel:E2}");
    }

    // ── planning only ──

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Planner_SendsGroupedBf16AndPerHeadCallsToCudnnOnlyInsidePermissions()
    {
        if (Skip()) return;
        using CudaBackend backend = new(0, PtxDir());
        const int s = 16, d = 128;

        // Left undisposed: a plan reads only shapes, and nothing here touches the device.
        AttentionCall TokenMajor(DType dt, int hq, int hkv)
        {
            Tensor q = new(new TensorShape(1, s, hq, d), dt), k = new(new TensorShape(1, s, hkv, d), dt);
            Tensor v = new(new TensorShape(1, s, hkv, d), dt), o = new(new TensorShape(1, s, hq, d), dt);
            return AttentionCall.FromTokenMajor(o, q, k, v, null, hq, hkv, d, 0.125f, true);
        }

        AttentionCall HeadMajor(DType dt, int h, long biasHeads)
        {
            Tensor q = new(new TensorShape(1, h, s, d), dt), k = new(new TensorShape(1, h, s, d), dt);
            Tensor v = new(new TensorShape(1, h, s, d), dt), o = new(new TensorShape(1, h, s, d), dt);
            Tensor? bias = biasHeads == 0 ? null : new Tensor(new TensorShape(1, biasHeads, s, s), DType.F32);
            return AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, 0.125f, true);
        }

        AttentionCall grouped = TokenMajor(DType.F16, 48, 12);
        Assert.Equal("cudnn.tokenmajor.gqa", backend.PlanAttention(in grouped).RowId);
        AttentionCall equalHeads = TokenMajor(DType.F16, 24, 24) with { Route = AttentionRoute.TokenMajorGqa };
        Assert.Equal("permute.widen.dense", backend.PlanAttention(in equalHeads).RowId);
        AttentionCall f32Grouped = TokenMajor(DType.F32, 48, 12);
        Assert.Equal("permute.widen.dense", backend.PlanAttention(in f32Grouped).RowId);

        AttentionCall perHead = HeadMajor(DType.F16, 8, biasHeads: 8);
        Assert.Equal("cudnn.f16", backend.PlanAttention(in perHead).RowId);

        AttentionCall bf16 = HeadMajor(DType.BF16, 8, biasHeads: 0);
        // allowF16 is true in HeadMajor, so this call carries the legacy F16 permission and narrows inside the materialized
        // route, as it did before the split. The split names that row materialized.f16; the computation is the same.
        Assert.Equal("materialized.f16", backend.PlanAttention(in bf16).RowId);
        AttentionCall bf16Granted = bf16 with { Numerics = GrantsBf16 };
        Assert.Equal("cudnn.f16", backend.PlanAttention(in bf16Granted).RowId);
    }

    // ── by-value scale ──

    private static ulong Upload(Half[] host)
    {
        ulong dptr = CudaMemory.Allocate((nuint)(host.Length * sizeof(ushort)));
        fixed (Half* p = host) CudaMemory.CopyHostToDevice(dptr, p, (nuint)(host.Length * sizeof(ushort)));
        return dptr;
    }

    private static Half[] Download(ulong dptr, int count)
    {
        Half[] host = new Half[count];
        fixed (Half* p = host) CudaMemory.CopyDeviceToHost(p, dptr, (nuint)(count * sizeof(ushort)));
        return host;
    }

    /// <summary>Three scales enqueue back to back with no sync, so each launch must capture its scale at enqueue; a fresh plan must match the reused one bit for bit.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ByValueScale_OnePlanServesEveryScale_BitwiseEqualToAFreshPlan()
    {
        if (Skip()) return;
        const int h = 8, s = 256, d = 64, n = h * s * d;
        float[] scales = [1f / MathF.Sqrt(d), 0.05f, 0.3f];
        Random rng = new(15);
        float[] qv = Quantize(Gauss(rng, n), DType.F16), kv = Quantize(Gauss(rng, n), DType.F16), vv = Quantize(Gauss(rng, n), DType.F16);
        CudnnSdpa.View view = CudnnSdpa.View.HeadMajor(1, h, s, d);
        CudnnSdpa.PlanKey key = new(CudnnSdpa.SdpaDtype.F16, d, view, view, view, view);

        using CudaBackend backend = new(0, PtxDir());
        using CudnnSdpa sdpa = new(backend.Stream.Handle);
        List<ulong> owned = new();
        try
        {
            ulong dq = Upload(qv.Select(x => (Half)x).ToArray()), dk = Upload(kv.Select(x => (Half)x).ToArray()), dv = Upload(vv.Select(x => (Half)x).ToArray());
            owned.AddRange([dq, dk, dv]);
            ulong[] outs = [.. scales.Append(scales[0]).Select(_ => CudaMemory.Allocate((nuint)(n * sizeof(ushort))))];
            owned.AddRange(outs);
            for (int i = 0; i < outs.Length; i++)
                sdpa.Execute(dq, dk, dv, outs[i], key, scales[i % scales.Length]);   // scale A, B, C, then A again on the same plan
            backend.Sync();
            Assert.Equal(1, sdpa.PlanCount);

            Half[][] got = [.. outs.Select(p => Download(p, n))];
            Assert.True(got[0].AsSpan().SequenceEqual(got[3]), "scale A through a plan that ran scales B and C in between changed bits");
            Assert.False(got[0].AsSpan().SequenceEqual(got[1]), "two different scales gave the same output");
            for (int i = 0; i < scales.Length; i++)
            {
                double rel = RelL2([.. got[i].Select(x => (float)x)], Reference(qv, kv, vv, null, 1, h, h, s, s, d, scales[i]));
                _output.WriteLine($"scale {scales[i]}: relL2 {rel:E2}");
                Assert.True(rel <= F16Tolerance, $"scale {scales[i]} relL2 {rel:E2}");
            }

            using CudnnSdpa fresh = new(backend.Stream.Handle);
            ulong freshOut = CudaMemory.Allocate((nuint)(n * sizeof(ushort)));
            owned.Add(freshOut);
            fresh.Execute(dq, dk, dv, freshOut, key, scales[0]);
            backend.Sync();
            Assert.True(got[0].AsSpan().SequenceEqual(Download(freshOut, n)), "a fresh plan and a reused one disagree for the same scale");
        }
        finally
        {
            foreach (ulong p in owned) CudaMemory.Free(p);
        }
    }
}
