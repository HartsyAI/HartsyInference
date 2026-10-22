using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.CudnnSdpaVariantTests;

namespace HartsyInference.Cuda.Tests;

/// <summary>The strided cuDNN rows through <c>IBackend.Attention</c>, each against the F32 reference at the F16 tolerance of the other cuDNN tests.
/// Token-major is [b, s, h, d] and head-major [b, h, s, d]; the reference always reads head-major values.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaStridedRowsTests
{
    private readonly ITestOutputHelper _output;
    public CudnnSdpaStridedRowsTests(ITestOutputHelper output) => _output = output;

    private bool Skip()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return true; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return true; }
        return false;
    }

    private static Tensor F16(float[] headMajor, int b, int h, int s, int d, bool tokenMajor) =>
        tokenMajor ? Make(Transpose(headMajor, b, h, s, d), new TensorShape(b, s, h, d), DType.F16) : Make(headMajor, new TensorShape(b, h, s, d), DType.F16);

    /// <summary>The attention output as head-major values.</summary>
    private static float[] HeadMajor(Tensor o, int b, int h, int s, int d, bool tokenMajor) =>
        tokenMajor ? Transpose(Read(o), b, s, h, d) : Read(o);

    private double Check(string what, float[] got, float[] want)
    {
        double rel = RelL2(got, want);
        _output.WriteLine($"{what}: route {CudaBackend.LastAttentionRoute} relL2 {rel:E2}");
        Assert.True(rel <= F16Tolerance, $"{what}: relL2 {rel:E2}");
        return rel;
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("perbatch")]
    [InlineData("batchhead")]
    [InlineData("batchkey")]
    [InlineData("key")]
    [InlineData("perhead")]
    public void TokenMajorB2_WithABias_MatchesF32Reference_OnCudnn(string biasKind)
    {
        if (Skip()) return;
        const int b = 2, h = 8, s = 256, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(21);
        float[] q = Quantize(Gauss(rng, (long)b * h * s * d), DType.F16), k = Quantize(Gauss(rng, (long)b * h * s * d), DType.F16);
        float[] v = Quantize(Gauss(rng, (long)b * h * s * d), DType.F16);
        using Tensor qt = F16(q, b, h, s, d, true), kt = F16(k, b, h, s, d, true), vt = F16(v, b, h, s, d, true);
        using Tensor? bias = BiasTensor(biasKind, rng, b, h, s, s, out int bB, out int bH, out int bS, out float[]? biasValues);
        using Tensor o = new(new TensorShape(b, s, h, d), DType.F16);
        AttentionCall call = AttentionCall.FromTokenMajor(o, qt, kt, vt, bias, h, h, d, scale, allowF16: true);
        Assert.NotNull(call.LegacyGap());   // B = 2: no legacy entry carries it

        using CudaBackend backend = new(0, PtxDir());
        ((IBackend)backend).Attention(in call);
        backend.Sync();

        Assert.Equal("M.2", CudaBackend.LastAttentionRoute);
        Assert.True(backend.CudnnSdpaEngaged, "the batched token-major call did not run on cuDNN");
        Check($"token-major b{b} h{h} s{s} d{d} bias={biasKind}", HeadMajor(o, b, h, s, d, true), Reference(q, k, v, biasValues, b, h, h, s, s, d, scale, bB, bH, bS));
    }

    /// <summary>The shape classes of the sites that moved onto the token-major entry: M.2 must give the bits the head-major row
    /// they used gave (F32 calls cast whole tensors on both rows), at sequence lengths off the kernel's tile size too, and past
    /// Int32.MaxValue scores: SD1.5 at 2048² and Wan 14B's 720p length.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("zimage-1024", 30, 128, 4128, true, "S.3")]
    [InlineData("sdxl-b1", 10, 64, 4096, true, "S.3")]
    [InlineData("sd15-d40", 8, 40, 4096, false, "S.5")]
    [InlineData("sd15-d80", 8, 80, 1024, false, "S.5")]
    [InlineData("sd15-d160", 8, 160, 256, false, "S.5")]
    [InlineData("krea2-fusion-12", 20, 128, 12, false, "S.5")]
    [InlineData("krea2-fusion-117", 20, 128, 117, false, "S.5")]
    [InlineData("krea2-fusion-333", 20, 128, 333, false, "S.5")]
    [InlineData("sd15-2048", 8, 40, 65536, false, "S.5")]
    [InlineData("rows-75600", 8, 128, 75600, false, "S.5")]
    public void TokenMajor_IsBitwiseEqualToTheHeadMajorRow_AtTheFlippedSitesShapes(string name, int h, int d, int s, bool f16,
        string headMajorRoute)
    {
        if (Skip()) return;
        DType dtype = f16 ? DType.F16 : DType.F32;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(s * 31 + h * 7 + d);
        float[] q = Gauss(rng, (long)h * s * d), k = Gauss(rng, (long)h * s * d), v = Gauss(rng, (long)h * s * d);
        TensorShape headMajor = new(1, h, s, d), tokenMajor = new(1, s, h, d);
        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;

        using Tensor qh = Make(q, headMajor, dtype), kh = Make(k, headMajor, dtype), vh = Make(v, headMajor, dtype);
        using Tensor oh = new(headMajor, dtype);
        b.ScaledDotProductAttention(oh, qh, kh, vh, null, scale, allowF16: true);
        backend.Sync();
        Assert.Equal(headMajorRoute, CudaBackend.LastAttentionRoute);

        using Tensor qt = Make(Transpose(q, 1, h, s, d), tokenMajor, dtype), kt = Make(Transpose(k, 1, h, s, d), tokenMajor, dtype);
        using Tensor vt = Make(Transpose(v, 1, h, s, d), tokenMajor, dtype), ot = new(new TensorShape(s, h * d), dtype);
        b.ScaledDotProductAttentionTokenMajor(ot, qt, kt, vt, null, h, d, scale, allowF16: true);
        backend.Sync();
        Assert.Equal("M.2", CudaBackend.LastAttentionRoute);

        float[] fromHeadMajor = Read(oh), fromTokenMajor = Transpose(Read(ot), 1, s, h, d);
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        _output.WriteLine($"{name}: {headMajorRoute} vs M.2, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal(0, differing);
    }

    /// <summary>The single-stream blocks' attention through <see cref="AttentionLayer.Attend"/>: the token-major config writes the
    /// caller's [1, S, H·D] through M.2 with the bits the head-major config's permutes and row give, with Chroma's [1, 1, S, S] padding
    /// mask too and F32 cast whole on both; at Flux's 1536² length the Sage guard keeps the head-major Sage row.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("flux1-1024", 24, 4608, true, false, "S.3", "M.2")]
    [InlineData("flux1-1024-f32", 24, 4608, false, false, "S.5", "M.2")]
    [InlineData("flux1-1536", 24, 9728, true, false, "S.1", "S.1")]
    [InlineData("chroma-1024", 24, 4107, true, false, "S.3", "M.2")]
    [InlineData("chroma-1024-masked", 24, 4107, true, true, "S.3", "M.2")]
    [InlineData("chroma-1024-masked-f32", 24, 4107, false, true, "S.5", "M.2")]
    [InlineData("flux2-1024", 48, 4608, true, false, "S.3", "M.2")]
    [InlineData("hunyuanimage-f32", 28, 4173, false, false, "S.5", "M.2")]
    public void SingleStreamAttend_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int s, bool f16, bool masked,
        string headMajorRoute, string tokenMajorRoute)
    {
        if (Skip()) return;
        const int d = 128, txt = 11, keptTxt = 4;
        DType dtype = f16 ? DType.F16 : DType.F32;
        Random rng = new(s * 31 + h);
        float[] q = Gauss(rng, (long)s * h * d), k = Gauss(rng, (long)s * h * d), v = Gauss(rng, (long)s * h * d);
        float[]? bias = masked ? PaddingMask(s, txt, keptTxt) : null;
        TensorShape operand = new(1, s, h, d), flat = new(1, s, h * d);
        using CudaBackend backend = new(0, PtxDir());

        float[] Run(AttentionRoute route)
        {
            AttentionLayer layer = new(new AttentionLayerConfig
            {
                Heads = h, KvHeads = h, HeadDim = d, AllowF16 = true, TokenMajorRoute = route, HeadMajorForSage = true,
            });
            using Tensor? mask = bias is null ? null : Make(bias, new TensorShape(1, 1, s, s), DType.F32);
            using Tensor o = layer.Attend(backend, Make(q, operand, dtype), Make(k, operand, dtype), Make(v, operand, dtype), flat,
                new AttentionCallContext { Bias = mask });
            backend.Sync();
            Assert.Equal(flat, o.Shape);
            return Read(o);
        }

        float[] fromHeadMajor = Run(AttentionRoute.Dense);
        Assert.Equal(headMajorRoute, CudaBackend.LastAttentionRoute);
        float[] fromTokenMajor = Run(AttentionRoute.TokenMajor);
        Assert.Equal(tokenMajorRoute, CudaBackend.LastAttentionRoute);
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        _output.WriteLine($"{name}: {headMajorRoute} vs {tokenMajorRoute}, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal(0, differing);
    }

    /// <summary>The double-stream blocks' joint call through <see cref="AttentionLayer.AttendFlat"/>: Q, K, V and the output are the
    /// concatenated [1, S, H·D] rows, which the token-major config attends in place through M.2 with the bits the head-major config's
    /// permutes and row give: F32 cast whole on both, Chroma's padding mask, and a context-parallel rank's local queries over the
    /// whole sequence's keys. At Flux's 1536² length the Sage guard keeps the head-major Sage row.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("flux1-1024", 24, 128, 4608, 4608, true, false, "S.3", "M.2")]
    [InlineData("flux1-1024-f32", 24, 128, 4608, 4608, false, false, "S.5", "M.2")]
    [InlineData("flux1-1536", 24, 128, 9728, 9728, true, false, "S.1", "S.1")]
    [InlineData("chroma-1024-masked", 24, 128, 4107, 4107, true, true, "S.3", "M.2")]
    [InlineData("flux2-1024", 48, 128, 4608, 4608, true, false, "S.3", "M.2")]
    [InlineData("qwenimage-1024", 24, 128, 4115, 4115, false, false, "S.5", "M.2")]
    [InlineData("qwenimage-1024-f16", 24, 128, 4115, 4115, true, false, "S.3", "M.2")]
    [InlineData("qwenimage-cp-rank", 24, 128, 2067, 4115, false, false, "S.5", "M.2")]
    [InlineData("hunyuanimage-f32", 28, 128, 1101, 1101, false, false, "S.5", "M.2")]
    [InlineData("auraflow-f32", 12, 256, 4352, 4352, false, false, "S.5", "M.2")]
    [InlineData("hidream-f32", 20, 128, 4352, 4352, false, false, "S.5", "M.2")]
    public void JointAttendFlat_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int d, int sq, int skv, bool f16, bool masked,
        string headMajorRoute, string tokenMajorRoute)
    {
        if (Skip()) return;
        const int txt = 11, keptTxt = 4;
        DType dtype = f16 ? DType.F16 : DType.F32;
        Random rng = new(sq * 31 + skv * 7 + h);
        float[] q = Gauss(rng, (long)sq * h * d), k = Gauss(rng, (long)skv * h * d), v = Gauss(rng, (long)skv * h * d);
        float[]? bias = masked ? PaddingMask(sq, txt, keptTxt) : null;
        TensorShape rows = new(1, sq, h * d), kvRows = new(1, skv, h * d);
        using CudaBackend backend = new(0, PtxDir());

        float[] Run(AttentionRoute route)
        {
            AttentionLayer layer = new(new AttentionLayerConfig
            {
                Heads = h, KvHeads = h, HeadDim = d, AllowF16 = true, TokenMajorRoute = route, HeadMajorForSage = true,
            });
            using Tensor? mask = bias is null ? null : Make(bias, new TensorShape(1, 1, sq, skv), DType.F32);
            using Tensor o = layer.AttendFlat(backend, Make(q, rows, dtype), Make(k, kvRows, dtype), Make(v, kvRows, dtype), rows,
                new AttentionCallContext { Bias = mask });
            backend.Sync();
            Assert.Equal(rows, o.Shape);
            return Read(o);
        }

        float[] fromHeadMajor = Run(AttentionRoute.Dense);
        Assert.Equal(headMajorRoute, CudaBackend.LastAttentionRoute);
        float[] fromTokenMajor = Run(AttentionRoute.TokenMajor);
        Assert.Equal(tokenMajorRoute, CudaBackend.LastAttentionRoute);
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        _output.WriteLine($"{name}: {headMajorRoute} vs {tokenMajorRoute}, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal(0, differing);
    }

    /// <summary>The full-width RMS self-attention of Wan and HiDream's single-stream blocks through <see cref="AttentionLayer.Forward"/>,
    /// projections, norm, rotation and output projection included: the token-major config attends the rows in place through M.2 with
    /// the bits the head-major config's permutes and S.5 give, F32 cast whole on both. Wan 2.2 5B at 49 and 121 frames (F32 never takes
    /// Sage, at any length), Wan 2.1 14B, Matrix-Game 2's block mask, a context-parallel rank's local queries over the exchanged keys,
    /// and HiDream's [1, S, hidden] rows.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("wan5b-49f", 24, 5070, 0, false, false)]
    [InlineData("wan5b-121f", 24, 12090, 0, false, false)]
    [InlineData("wan21-14b", 40, 4480, 0, false, false)]
    [InlineData("matrixgame2-masked", 12, 2640, 0, true, false)]
    [InlineData("wan5b-cp-rank", 24, 2730, 5070, false, false)]
    [InlineData("hidream-single", 20, 4352, 0, false, true)]
    public void FullWidthRmsForward_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int s, int exchanged, bool masked, bool rank3) =>
        AssertFullWidthRmsRoutesBitwise(name, h, s, exchanged, masked, rank3, DType.F32);

    /// <summary>Wan's F16 branches (TI2V-5B and Wan 2.1 14B: one timestep group, no mask, no exchange): F16 rows and weights through the
    /// F16 twin, which refuses INT8, take S.3 head-major and M.2 token-major with the same bits, at 121 frames past Sage's minimum
    /// length too.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("wan5b-49f", 24, 5070)]
    [InlineData("wan5b-121f", 24, 12090)]
    [InlineData("wan21-14b", 40, 4480)]
    public void F16FullWidthRmsForward_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int s) =>
        AssertFullWidthRmsRoutesBitwise(name, h, s, 0, false, false, DType.F16);

    private void AssertFullWidthRmsRoutesBitwise(string name, int h, int s, int exchanged, bool masked, bool rank3, DType dtype)
    {
        if (Skip()) return;
        const int d = 128, framesOfMask = 3;
        int inner = h * d, skv = exchanged > 0 ? exchanged : s;
        Random rng = new(s * 31 + h);
        float[] x = Gauss(rng, (long)s * inner), angles = Gauss(rng, (long)s * d, 3f);
        float[][] w = [Gauss(rng, (long)inner * inner, 0.02f), Gauss(rng, (long)inner * inner, 0.02f), Gauss(rng, (long)inner * inner, 0.02f),
            Gauss(rng, (long)inner * inner, 0.02f)];
        float[] qNorm = Gauss(rng, inner, 0.1f), kNorm = Gauss(rng, inner, 0.1f);
        float[]? kFull = exchanged > 0 ? Gauss(rng, (long)skv * inner) : null, vFull = exchanged > 0 ? Gauss(rng, (long)skv * inner) : null;
        float[]? bias = masked ? BlockCausalMask(s, framesOfMask) : null;
        TensorShape rows = rank3 ? new(1, s, inner) : new(s, inner);
        using CudaBackend backend = new(0, PtxDir());

        float[] Run(AttentionRoute route)
        {
            using Tensor wq = Make(w[0], new TensorShape(inner, inner), dtype), wk = Make(w[1], new TensorShape(inner, inner), dtype);
            using Tensor wv = Make(w[2], new TensorShape(inner, inner), dtype), wo = Make(w[3], new TensorShape(inner, inner), dtype);
            using Tensor nq = Make(qNorm, new TensorShape(inner), DType.F32), nk = Make(kNorm, new TensorShape(inner), DType.F32);
            using Tensor cos = Make([.. angles.Select(MathF.Cos)], new TensorShape(s, d), DType.F32);
            using Tensor sin = Make([.. angles.Select(MathF.Sin)], new TensorShape(s, d), DType.F32);
            using Tensor? mask = bias is null ? null : Make(bias, new TensorShape(1, 1, s, s), DType.F32);
            AttentionLayer layer = new(new AttentionLayerConfig
            {
                Heads = h, KvHeads = h, HeadDim = d, QkNorm = AttentionQkNorm.FullWidthRms, ComputeDType = dtype, AllowF16 = true,
                TokenMajorRoute = route, HeadMajorForSage = route != AttentionRoute.Dense,
                RefusedFormats = dtype == DType.F16 ? AttentionFormats.Int8 : AttentionFormats.None,
            }, new AttentionLayerWeights
            {
                Query = wq, Key = wk, Value = wv, Output = wo, QueryNorm = nq, KeyNorm = nk, QueryNormEps = 1e-6f, KeyNormEps = 1e-6f,
            });
            QkRotation rotation = new(QkRotationKind.InterleavedPairs, cos, sin, s);
            using Tensor input = Make(x, rows, dtype);
            using Tensor o = layer.Forward(backend, input, input, new AttentionCallContext
            {
                QueryRotation = rotation, KeyRotation = rotation, Bias = mask,
                KvRewrite = kFull is null ? null : new Exchanged(kFull, vFull!, new TensorShape(skv, inner)),
            });
            backend.Sync();
            Assert.Equal(rows, o.Shape);
            return Read(o);
        }

        string headMajorRoute = dtype == DType.F16 ? "S.3" : "S.5";
        float[] fromHeadMajor = Run(AttentionRoute.Dense);
        Assert.Equal(headMajorRoute, CudaBackend.LastAttentionRoute);
        float[] fromTokenMajor = Run(AttentionRoute.TokenMajor);
        Assert.Equal("M.2", CudaBackend.LastAttentionRoute);
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        _output.WriteLine($"{name} {dtype}: {headMajorRoute} vs M.2 over {s} query and {skv} key rows, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal(0, differing);
    }

    /// <summary>SDXL's CFG batch of two through <see cref="AttentionLayer.Forward"/>, projections and output projection included: once
    /// CUDA plans the batch on M.2, the token-major config attends both sequences in place with the bits the head-major config's
    /// permutes and S.5 give, F32 cast whole on both, at the UNet's 1024² levels and sdxl-wide's 1280×720 first level.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("sdxl-64x64", 10, 4096)]
    [InlineData("sdxl-32x32", 20, 1024)]
    [InlineData("sdxl-wide-80x45", 10, 3600)]
    public void BatchedSelfAttention_TokenMajorIsBitwiseTheHeadMajorRoute(string name, int h, int s)
    {
        if (Skip()) return;
        const int batch = 2, d = 64;
        int channels = h * d;
        Random rng = new(s * 37 + h);
        float[] x = Gauss(rng, (long)batch * s * channels);
        float[][] w = [Gauss(rng, (long)channels * channels, 0.03f), Gauss(rng, (long)channels * channels, 0.03f),
            Gauss(rng, (long)channels * channels, 0.03f), Gauss(rng, (long)channels * channels, 0.03f)];
        float[] outBias = Gauss(rng, channels, 0.1f);
        TensorShape rows = new(batch, s, channels), square = new(channels, channels);
        using CudaBackend backend = new(0, PtxDir());

        float[] Run(AttentionRoute route)
        {
            using Tensor wq = Make(w[0], square, DType.F32), wk = Make(w[1], square, DType.F32), wv = Make(w[2], square, DType.F32);
            using Tensor wo = Make(w[3], square, DType.F32), bo = Make(outBias, new TensorShape(channels), DType.F32);
            AttentionLayer layer = new(new AttentionLayerConfig
            {
                Heads = h, KvHeads = h, HeadDim = d, TokenMajorRoute = route, HeadMajorForSage = true, TokenMajorBatches = true,
                AllowF16 = true,
            }, new AttentionLayerWeights { Query = wq, Key = wk, Value = wv, Output = wo, OutputBias = bo });
            using Tensor input = Make(x, rows, DType.F32);
            using Tensor o = layer.Forward(backend, input, input, default);
            backend.Sync();
            Assert.Equal(rows, o.Shape);
            return Read(o);
        }

        float[] fromHeadMajor = Run(AttentionRoute.Dense);
        Assert.Equal("S.5", CudaBackend.LastAttentionRoute);
        float[] fromTokenMajor = Run(AttentionRoute.TokenMajor);
        Assert.Equal("M.2", CudaBackend.LastAttentionRoute);
        int differing = 0;
        for (int i = 0; i < fromHeadMajor.Length; i++)
            if (BitConverter.SingleToInt32Bits(fromHeadMajor[i]) != BitConverter.SingleToInt32Bits(fromTokenMajor[i])) differing++;
        _output.WriteLine($"{name}: S.5 vs M.2 over {batch} sequences of {s}, {differing} of {fromHeadMajor.Length} elements differ");
        Assert.Equal(0, differing);
    }

    /// <summary>A context-parallel exchange: the rank's K and V traded for the whole sequence's rows, as host tensors.</summary>
    private sealed class Exchanged(float[] k, float[] v, TensorShape shape) : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor key, Tensor value)
        {
            key.Dispose();
            value.Dispose();
            return (Make(k, shape, DType.F32), Make(v, shape, DType.F32));
        }
    }

    /// <summary>Matrix-Game 2's additive block-causal mask over <paramref name="frames"/> equal frames: a query sees its own frame and the
    /// earlier ones.</summary>
    private static float[] BlockCausalMask(int s, int frames)
    {
        int perFrame = s / frames;
        float[] mask = new float[(long)s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                mask[(long)i * s + j] = j / perFrame <= i / perFrame ? 0f : -1e9f;
        return mask;
    }

    /// <summary>Chroma's additive mask over a [text; image] sequence: a pair is kept only when both tokens are, and the text past
    /// <paramref name="kept"/> is padding.</summary>
    private static float[] PaddingMask(int s, int txt, int kept)
    {
        float[] mask = new float[(long)s * s];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                mask[(long)i * s + j] = (i >= kept && i < txt) || (j >= kept && j < txt) ? -1e30f : 0f;
        return mask;
    }

    /// <summary>Q, K and V are <see cref="AttentionOperand.Packed"/> column views of one [b·s, 3·h·d] F16 tensor. Asserted: the
    /// reference tolerance and the cuDNN route (no permute, no staging). Reported, not asserted: whether the output equals bit for
    /// bit the same call over plain copies of the columns, whose plan has other strides and so may pick another engine.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void FusedQkvColumnViews_MatchTheReference_AndAreComparedBitwiseToPlainCopies(int b)
    {
        if (Skip()) return;
        const int h = 8, s = 256, d = 64, hd = h * d;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(31 + b);
        float[] qkv = Quantize(Gauss(rng, (long)b * s * 3 * hd), DType.F16);
        float[][] part = [new float[b * s * hd], new float[b * s * hd], new float[b * s * hd]];   // token-major [b, s, h·d] copies of the columns
        for (int row = 0; row < b * s; row++)
            for (int p = 0; p < 3; p++)
                Array.Copy(qkv, (long)row * 3 * hd + p * hd, part[p], (long)row * hd, hd);
        float[] Hm(float[] tokenMajor) => Transpose(tokenMajor, b, s, h, d);   // [b, s, h, d] -> [b, h, s, d]
        float[] want = Reference(Hm(part[0]), Hm(part[1]), Hm(part[2]), null, b, h, h, s, s, d, scale);

        using CudaBackend backend = new(0, PtxDir());
        using Tensor packed = Make(qkv, new TensorShape((long)b * s, 3 * hd), DType.F16);
        using Tensor oPacked = new(new TensorShape(b, s, h, d), DType.F16), oPlain = new(new TensorShape(b, s, h, d), DType.F16);
        AttentionCall viaViews = new()
        {
            Query = AttentionOperand.Packed(packed, s, 3 * hd, 0, h, d, b), Key = AttentionOperand.Packed(packed, s, 3 * hd, hd, h, d, b),
            Value = AttentionOperand.Packed(packed, s, 3 * hd, 2 * hd, h, d, b), Output = AttentionOperand.TokenMajor(oPacked, h, d),
            Scale = scale, Route = AttentionRoute.TokenMajor, Numerics = AttentionNumerics.FromLegacy(true),
        };
        Assert.True(CudnnSdpaRows.ViewsFit(in viaViews));
        ((IBackend)backend).Attention(in viaViews);
        backend.Sync();
        Assert.Equal("M.2", CudaBackend.LastAttentionRoute);
        long engagedAfterViews = backend.CudnnSdpaExecutionCount;
        Assert.True(backend.CudnnSdpaEngaged, "the packed call did not run on cuDNN");

        using Tensor qc = Make(part[0], new TensorShape(b, s, h, d), DType.F16), kc = Make(part[1], new TensorShape(b, s, h, d), DType.F16);
        using Tensor vc = Make(part[2], new TensorShape(b, s, h, d), DType.F16);
        AttentionCall viaCopies = AttentionCall.FromTokenMajor(oPlain, qc, kc, vc, null, h, h, d, scale, allowF16: true);
        ((IBackend)backend).Attention(in viaCopies);
        backend.Sync();
        Assert.Equal("M.2", CudaBackend.LastAttentionRoute);
        Assert.Equal(engagedAfterViews + 1, backend.CudnnSdpaExecutionCount);

        Check($"packed qkv views b{b}", HeadMajor(oPacked, b, h, s, d, true), want);
        Check($"plain copies b{b}", HeadMajor(oPlain, b, h, s, d, true), want);
        ReadOnlySpan<ushort> a = oPacked.AsReadOnlySpan<ushort>(), c = oPlain.AsReadOnlySpan<ushort>();
        int differing = 0;
        float maxDiff = 0f;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != c[i])
            {
                differing++;
                maxDiff = MathF.Max(maxDiff, MathF.Abs((float)BitConverter.UInt16BitsToHalf(a[i]) - (float)BitConverter.UInt16BitsToHalf(c[i])));
            }
        _output.WriteLine($"packed views vs plain copies, b{b}: bitwise {(differing == 0 ? "EQUAL" : $"DIFFERENT ({differing} of {a.Length} elements, max |diff| {maxDiff:E2})")}");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false, "none")]
    [InlineData(false, "key")]
    [InlineData(false, "perbatch")]
    [InlineData(true, "none")]
    public void GroupedTokenMajor48x12B2_MatchesF32Reference_WithNoWidening(bool fused, string biasKind)
    {
        if (Skip()) return;
        const int b = 2, hq = 48, hkv = 12, s = 128, d = 128;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(41);
        float[] q = Quantize(Gauss(rng, (long)b * hq * s * d), DType.F16), k = Quantize(Gauss(rng, (long)b * hkv * s * d), DType.F16);
        float[] v = Quantize(Gauss(rng, (long)b * hkv * s * d), DType.F16);
        using Tensor? bias = BiasTensor(biasKind, rng, b, hq, s, s, out int bB, out int bH, out int bS, out float[]? biasValues);
        using Tensor o = new(new TensorShape(b, s, hq, d), DType.F16);
        using CudaBackend backend = new(0, PtxDir());

        AttentionCall call;
        Tensor? rows = null, qt = null, kt = null, vt = null;
        try
        {
            if (fused)
            {
                // One [b·s, (hq + 2·hkv)·d] projection output, the way Krea2's fused QKV lays it out.
                int width = (hq + 2 * hkv) * d;
                float[] qTm = Transpose(q, b, hq, s, d), kTm = Transpose(k, b, hkv, s, d), vTm = Transpose(v, b, hkv, s, d);
                float[] all = new float[(long)b * s * width];
                for (int row = 0; row < b * s; row++)
                {
                    Array.Copy(qTm, (long)row * hq * d, all, (long)row * width, hq * d);
                    Array.Copy(kTm, (long)row * hkv * d, all, (long)row * width + hq * d, hkv * d);
                    Array.Copy(vTm, (long)row * hkv * d, all, (long)row * width + (hq + hkv) * d, hkv * d);
                }
                rows = Make(all, new TensorShape((long)b * s, width), DType.F16);
                call = new()
                {
                    Query = AttentionOperand.Packed(rows, s, width, 0, hq, d, b), Key = AttentionOperand.Packed(rows, s, width, hq * d, hkv, d, b),
                    Value = AttentionOperand.Packed(rows, s, width, (hq + hkv) * d, hkv, d, b), Output = AttentionOperand.TokenMajor(o, hq, d),
                    Scale = scale, Mask = AttentionMask.Additive(bias), Numerics = AttentionNumerics.FromLegacy(true), Route = AttentionRoute.TokenMajorGqa,
                };
            }
            else
            {
                qt = F16(q, b, hq, s, d, true); kt = F16(k, b, hkv, s, d, true); vt = F16(v, b, hkv, s, d, true);
                call = AttentionCall.FromTokenMajor(o, qt, kt, vt, bias, hq, hkv, d, scale, allowF16: true);
            }
            Assert.Equal(AttentionRoute.TokenMajorGqa, call.Route);
            ((IBackend)backend).Attention(in call);
            backend.Sync();
        }
        finally
        {
            rows?.Dispose(); qt?.Dispose(); kt?.Dispose(); vt?.Dispose();
        }

        Assert.Equal("G.1", CudaBackend.LastAttentionRoute);
        Assert.True(backend.CudnnSdpaEngaged, "the grouped call did not run on cuDNN");
        Check($"grouped {hq}/{hkv} b{b} fused={fused} bias={biasKind}", HeadMajor(o, b, hq, s, d, true), Reference(q, k, v, biasValues, b, hq, hkv, s, s, d, scale, bB, bH, bS));
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void MixedLayouts_QueryTokenMajor_KeyValueHeadMajor_MatchTheReference()
    {
        if (Skip()) return;
        const int h = 8, s = 256, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(51);
        float[] q = Quantize(Gauss(rng, (long)h * s * d), DType.F16), k = Quantize(Gauss(rng, (long)h * s * d), DType.F16);
        float[] v = Quantize(Gauss(rng, (long)h * s * d), DType.F16);
        using Tensor qt = F16(q, 1, h, s, d, true), kt = F16(k, 1, h, s, d, false), vt = F16(v, 1, h, s, d, false);
        using Tensor o = new(new TensorShape(1, s, h, d), DType.F16);
        AttentionCall call = new()
        {
            Query = AttentionOperand.TokenMajor(qt, h, d), Key = AttentionOperand.HeadMajor(kt), Value = AttentionOperand.HeadMajor(vt),
            Output = AttentionOperand.TokenMajor(o, h, d), Scale = scale, Route = AttentionRoute.Dense, Numerics = AttentionNumerics.FromLegacy(true),
        };

        using CudaBackend backend = new(0, PtxDir());
        ((IBackend)backend).Attention(in call);
        backend.Sync();

        Assert.Equal("S.3", CudaBackend.LastAttentionRoute);
        Assert.True(backend.CudnnSdpaEngaged, "the mixed-layout call did not run on cuDNN");
        Check("Q token-major, K/V head-major", HeadMajor(o, 1, h, s, d, true), Reference(q, k, v, null, 1, h, h, s, s, d, scale));
    }

    /// <summary>The plan is keyed by strides, not offsets: the same [s, 3·h·d] buffer read with the three parts in two column orders runs on one plan.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ColumnOffsetsShareOnePlan_AndEachRoleOrderMatchesItsReference()
    {
        if (Skip()) return;
        const int h = 4, s = 128, d = 64, hd = h * d;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(61);
        float[] qkv = Quantize(Gauss(rng, (long)s * 3 * hd), DType.F16);
        float[][] part = [new float[s * hd], new float[s * hd], new float[s * hd]];
        for (int row = 0; row < s; row++)
            for (int p = 0; p < 3; p++)
                Array.Copy(qkv, (long)row * 3 * hd + p * hd, part[p], (long)row * hd, hd);

        using CudaBackend backend = new(0, PtxDir());
        using CudnnSdpa sdpa = new(backend.Stream.Handle);
        ulong dev = CudaMemory.Allocate((nuint)(qkv.Length * sizeof(ushort)));
        ulong[] outs = [CudaMemory.Allocate((nuint)(s * hd * sizeof(ushort))), CudaMemory.Allocate((nuint)(s * hd * sizeof(ushort)))];
        try
        {
            Half[] host = [.. qkv.Select(x => (Half)x)];
            fixed (Half* p = host) CudaMemory.CopyHostToDevice(dev, p, (nuint)(host.Length * sizeof(ushort)));
            CudnnSdpa.View view = new(1, h, s, (long)s * 3 * hd, d, 3 * hd);
            CudnnSdpa.View outView = CudnnSdpa.View.TokenMajor(1, h, s, d);
            CudnnSdpa.PlanKey key = new(CudnnSdpa.SdpaDtype.F16, d, view, view, view, outView);
            int[][] order = [[0, 1, 2], [2, 0, 1]];   // which column part is Q, K, V
            for (int i = 0; i < order.Length; i++)
            {
                int[] r = order[i];
                sdpa.Execute(dev, dev, dev, outs[i], key, scale, 0, new CudnnSdpa.Offsets(r[0] * hd, r[1] * hd, r[2] * hd, 0));
            }
            backend.Sync();
            Assert.Equal(1, sdpa.PlanCount);
            for (int i = 0; i < order.Length; i++)
            {
                Half[] got = new Half[s * hd];
                fixed (Half* p = got) CudaMemory.CopyDeviceToHost(p, outs[i], (nuint)(got.Length * sizeof(ushort)));
                int[] r = order[i];
                float[] want = Reference(Transpose(part[r[0]], 1, s, h, d), Transpose(part[r[1]], 1, s, h, d), Transpose(part[r[2]], 1, s, h, d), null, 1, h, h, s, s, d, scale);
                double rel = RelL2(Transpose([.. got.Select(x => (float)x)], 1, s, h, d), want);
                _output.WriteLine($"roles {string.Join("", r)}: relL2 {rel:E2}");
                Assert.True(rel <= F16Tolerance, $"column roles {string.Join("", r)}: relL2 {rel:E2}");
            }
        }
        finally
        {
            CudaMemory.Free(dev);
            foreach (ulong p in outs) CudaMemory.Free(p);
        }
    }
}
