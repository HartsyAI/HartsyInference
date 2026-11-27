using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;
using Xunit;

namespace HartsyInference.Vision.Dinov3.Tests;

/// <summary>The timm-layout DINOv3 backbone Depth Pro 2 embeds: no q/k/v bias, no final norm, taps at raw block outputs, and RoPE coordinates divided by the longer
/// grid side. Synthetic weights only; real-weight parity against timm 1.0.27 lands with the head that consumes these taps.</summary>
public sealed unsafe class Dinov3TimmLayoutTests
{
    private static readonly IBackend Backend = new CpuBackend();

    private static Dinov2Preset TimmPreset(bool gated) => Dinov3Fixture.TinyPreset(imageSize: 64) with
    {
        GatedMlp = gated, AttentionQkvBias = false, HasFinalNorm = false, RopeNormalizesByLongSide = true, RoundRopePeriodsToBf16 = false,
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TimmLayoutThroughTheAdapter_MatchesTheHuggingFaceLayoutWithZeroBiases(bool gated)
    {
        Dinov2Preset timmPreset = TimmPreset(gated);
        Dictionary<string, Tensor> timm = TimmWeights(timmPreset, seed: 11, prefix: "encoder.");
        Dictionary<string, Tensor> adapted = Dinov3TimmKeys.ToHuggingFace(timm, "encoder.", timmPreset);
        Dictionary<string, Tensor> hf = HuggingFaceCopy(timm, "encoder.", timmPreset);
        try
        {
            Dinov3VisionEncoder viaAdapter = new(timmPreset);
            viaAdapter.LoadWeights(adapted);
            // The independent copy keeps a zero q and v bias per layer, which must give the same numbers as no bias at all.
            Dinov3VisionEncoder viaCopy = new(timmPreset with { AttentionQkvBias = true });
            viaCopy.LoadWeights(hf);
            using Tensor pixels = Pixels(1, 32, 64, seed: 12);

            using Tensor a = viaAdapter.Encode(Backend, pixels);
            using Tensor b = viaCopy.Encode(Backend, pixels);

            Assert.Equal(new TensorShape(1, 1 + 4 + 2 * 4, 128), a.Shape);
            ReadOnlySpan<float> x = a.AsReadOnlySpan<float>(), y = b.AsReadOnlySpan<float>();
            Assert.True(MaxAbsDiff(x, y) < 1e-5, $"max |diff| {MaxAbsDiff(x, y)}");
            Assert.True(MaxAbs(x) > 1e-3, "the output is all zero, so the comparison proves nothing");
        }
        finally
        {
            Dinov3Fixture.Dispose(hf);
            Dispose(timm);
        }
    }

    [Theory]
    [InlineData("blocks.0.attn.qkv.weight", 384, 128)]
    [InlineData("blocks.1.mlp.fc1.weight", 256, 128)]
    public void TimmAdapter_RefusesAQuantizedWeight_BeforeASliceCouldDropItsScales(string key, long rows, long columns)
    {
        Dinov2Preset preset = TimmPreset(gated: false);
        Dictionary<string, Tensor> timm = TimmWeights(preset, seed: 14, prefix: "");
        try
        {
            timm[key].Dispose();
            timm[key] = new Tensor(new TensorShape(rows, columns), DType.F8E4M3) { QuantInfo = new QuantWeightInfo { Format = "fp8_block" } };

            NotSupportedException error = Assert.Throws<NotSupportedException>(() => Dinov3TimmKeys.ToHuggingFace(timm, "", preset));

            Assert.Contains($"'{key}'", error.Message);
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Fact]
    public void TimmQkvBias_KeepsTheQueryAndValueSlicesAndDropsTheKeyBias()
    {
        Dinov2Preset preset = TimmPreset(gated: false) with { AttentionQkvBias = true };
        Dictionary<string, Tensor> timm = TimmWeights(preset, seed: 13, prefix: "");
        try
        {
            Dictionary<string, Tensor> hf = Dinov3TimmKeys.ToHuggingFace(timm, "", preset);

            ReadOnlySpan<float> bias = timm["blocks.1.attn.qkv.bias"].AsReadOnlySpan<float>();
            Assert.Equal(bias[..128].ToArray(), hf["layer.1.attention.q_proj.bias"].AsReadOnlySpan<float>().ToArray());
            Assert.Equal(bias[256..].ToArray(), hf["layer.1.attention.v_proj.bias"].AsReadOnlySpan<float>().ToArray());
            Assert.False(hf.ContainsKey("layer.1.attention.k_proj.bias"));
            ReadOnlySpan<float> w = timm["blocks.0.attn.qkv.weight"].AsReadOnlySpan<float>();
            Assert.Equal(w[(128 * 128)..(256 * 128)].ToArray(), hf["layer.0.attention.k_proj.weight"].AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TimmPath_RefusesABiasWhenThePresetHasNone(bool separateBiases)
    {
        Dinov2Preset biased = TimmPreset(gated: false) with { AttentionQkvBias = true };
        Dinov2Preset biasFree = TimmPreset(gated: false);
        Dictionary<string, Tensor> timm = TimmWeights(biased, seed: 23, prefix: "encoder.");
        try
        {
            if (separateBiases)
            {
                for (int i = 0; i < biased.NumLayers; i++)
                {
                    timm.Remove($"encoder.blocks.{i}.attn.qkv.bias", out Tensor? fused);
                    fused!.Dispose();
                    timm[$"encoder.blocks.{i}.attn.q_bias"] = new Tensor(new TensorShape(128), DType.F32);
                    timm[$"encoder.blocks.{i}.attn.v_bias"] = new Tensor(new TensorShape(128), DType.F32);
                }
            }

            NotSupportedException error = Assert.Throws<NotSupportedException>(() => Dinov3TimmKeys.ToHuggingFace(timm, "encoder.", biasFree));
            Assert.Contains("AttentionQkvBias", error.Message);
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Fact]
    public void TimmSeparateQAndVBiases_AreRenamedWithoutSlicing()
    {
        Dinov2Preset preset = TimmPreset(gated: false) with { AttentionQkvBias = true };
        Dictionary<string, Tensor> timm = TimmWeights(preset, seed: 24, prefix: "");
        try
        {
            for (int i = 0; i < preset.NumLayers; i++)
            {
                timm.Remove($"blocks.{i}.attn.qkv.bias", out Tensor? fused);
                fused!.Dispose();
                Tensor q = new(new TensorShape(128), DType.F32), v = new(new TensorShape(128), DType.F32);
                q.AsSpan<float>().Fill(0.25f + i);
                v.AsSpan<float>().Fill(-0.5f - i);
                timm[$"blocks.{i}.attn.q_bias"] = q;
                timm[$"blocks.{i}.attn.v_bias"] = v;
            }

            Dictionary<string, Tensor> hf = Dinov3TimmKeys.ToHuggingFace(timm, "", preset);

            Assert.Same(timm["blocks.1.attn.q_bias"], hf["layer.1.attention.q_proj.bias"]);
            Assert.Same(timm["blocks.1.attn.v_bias"], hf["layer.1.attention.v_proj.bias"]);
            Assert.False(hf.ContainsKey("layer.1.attention.k_proj.bias"));
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Fact]
    public void Adapter_NamesAMissingOrMisshapenTensor()
    {
        Dinov2Preset preset = TimmPreset(gated: true);
        Dictionary<string, Tensor> timm = TimmWeights(preset, seed: 14, prefix: "encoder.");
        try
        {
            Tensor gate = timm["encoder.blocks.0.mlp.fc1_g.weight"];
            timm.Remove("encoder.blocks.0.mlp.fc1_g.weight");
            InvalidOperationException missing = Assert.Throws<InvalidOperationException>(() => Dinov3TimmKeys.ToHuggingFace(timm, "encoder.", preset));
            Assert.Contains("encoder.blocks.0.mlp.fc1_g.weight", missing.Message);
            timm["encoder.blocks.0.mlp.fc1_g.weight"] = gate;

            Tensor qkv = timm["encoder.blocks.1.attn.qkv.weight"];
            timm["encoder.blocks.1.attn.qkv.weight"] = new Tensor(new TensorShape(2 * 128, 128), DType.F32);
            qkv.Dispose();
            InvalidOperationException shape = Assert.Throws<InvalidOperationException>(() => Dinov3TimmKeys.ToHuggingFace(timm, "encoder.", preset));
            Assert.Contains("encoder.blocks.1.attn.qkv.weight", shape.Message);
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Fact]
    public void ABiasFreePresetRefusesACheckpointThatCarriesAQueryBias()
    {
        Dinov2Preset preset = TimmPreset(gated: false);
        Dictionary<string, Tensor> hf = Dinov3Fixture.Tiny(preset, seed: 15);
        try
        {
            Dinov3VisionEncoder encoder = new(preset);
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => encoder.LoadWeights(hf));
            Assert.Contains("AttentionQkvBias", error.Message);
        }
        finally
        {
            Dinov3Fixture.Dispose(hf);
        }
    }

    [Fact]
    public void WithoutAFinalNorm_EncodeReturnsTheLastBlockUnchanged()
    {
        Dinov2Preset preset = TimmPreset(gated: false);
        Dictionary<string, Tensor> timm = TimmWeights(preset, seed: 16, prefix: "");
        Dictionary<string, Tensor> hf = Dinov3TimmKeys.ToHuggingFace(timm, "", preset);
        try
        {
            Assert.False(hf.ContainsKey("norm.weight"));
            Dinov3VisionEncoder encoder = new(preset);
            encoder.LoadWeights(hf);
            using Tensor pixels = Pixels(1, 32, 64, seed: 17);

            using Tensor encoded = encoder.Encode(Backend, pixels);
            using Tensor lastBlock = encoder.EncodeBlocks(Backend, pixels, lastBlock: 1);

            Assert.Equal(lastBlock.AsReadOnlySpan<float>().ToArray(), encoded.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Fact]
    public void EncodeIntermediates_ReturnsTheRawBlockOutputsInTheOrderAsked()
    {
        Dinov2Preset preset = TimmPreset(gated: false) with { NumLayers = 3 };
        Dictionary<string, Tensor> timm = TimmWeights(preset, seed: 18, prefix: "");
        Dictionary<string, Tensor> hf = Dinov3TimmKeys.ToHuggingFace(timm, "", preset);
        try
        {
            Dinov3VisionEncoder encoder = new(preset);
            encoder.LoadWeights(hf);
            using Tensor pixels = Pixels(1, 32, 64, seed: 19);

            Tensor[] taps = encoder.EncodeIntermediates(Backend, pixels, [2, 0]);
            try
            {
                Assert.Equal(2, taps.Length);
                for (int i = 0; i < taps.Length; i++)
                {
                    using Tensor expected = encoder.EncodeBlocks(Backend, pixels, lastBlock: i == 0 ? 2 : 0);
                    Assert.Equal(expected.AsReadOnlySpan<float>().ToArray(), taps[i].AsReadOnlySpan<float>().ToArray());
                }
            }
            finally
            {
                foreach (Tensor t in taps) t.Dispose();
            }

            Assert.Throws<ArgumentException>(() => encoder.EncodeIntermediates(Backend, pixels, [1, 1]));
            Assert.Throws<ArgumentOutOfRangeException>(() => encoder.EncodeIntermediates(Backend, pixels, [3]));
            Assert.Throws<ArgumentOutOfRangeException>(() => encoder.EncodeIntermediates(Backend, pixels, [-1]));
            Assert.Throws<ArgumentException>(() => encoder.EncodeIntermediates(Backend, pixels, []));
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Fact]
    public void PrefixResetBlocks_KeepTheIncomingPrefixAndFeedItToTheNextBlock()
    {
        Dinov2Preset preset = TimmPreset(gated: false) with { NumLayers = 3, ImageSize = 32 };
        Dictionary<string, Tensor> timm = TimmWeights(preset, seed: 21, prefix: "");
        Dictionary<string, Tensor> hf = Dinov3TimmKeys.ToHuggingFace(timm, "", preset);
        try
        {
            Dinov3VisionEncoder encoder = new(preset);
            encoder.LoadWeights(hf);
            using Tensor pixels = Pixels(2, 32, 64, seed: 22);
            int prefix = 1 + preset.NumRegisterTokens, hidden = preset.HiddenSize, seq = (int)(prefix + 2L * 4);
            int prefixWidth = prefix * hidden;

            using Tensor embed = encoder.EncodeBlocks(Backend, pixels, lastBlock: -1);
            Tensor[] plain = encoder.EncodeIntermediates(Backend, pixels, [0, 1]);
            Tensor[] reset = encoder.EncodeIntermediates(Backend, pixels, [0, 1], prefixResetBlocks: [0]);
            try
            {
                for (int b = 0; b < 2; b++)
                {
                    int row = b * seq * hidden, rowEnd = (b + 1) * seq * hidden;
                    // Block 0's prefix is the embeddings' prefix again; its patch tokens are the plain block output.
                    Assert.Equal(embed.AsReadOnlySpan<float>().Slice(row, prefixWidth).ToArray(), reset[0].AsReadOnlySpan<float>().Slice(row, prefixWidth).ToArray());
                    Assert.Equal(plain[0].AsReadOnlySpan<float>()[(row + prefixWidth)..rowEnd].ToArray(),
                        reset[0].AsReadOnlySpan<float>()[(row + prefixWidth)..rowEnd].ToArray());
                }

                Assert.True(MaxAbsDiff(plain[0].AsReadOnlySpan<float>(), reset[0].AsReadOnlySpan<float>()) > 0, "the reset changed nothing");
                Assert.True(MaxAbsDiff(plain[1].AsReadOnlySpan<float>(), reset[1].AsReadOnlySpan<float>()) > 1e-6, "block 1 did not see the reset prefix");
            }
            finally
            {
                foreach (Tensor t in plain) t.Dispose();
                foreach (Tensor t in reset) t.Dispose();
            }

            Assert.Throws<ArgumentOutOfRangeException>(() => encoder.EncodeIntermediates(Backend, pixels, [0], prefixResetBlocks: [3]));
        }
        finally
        {
            Dispose(timm);
        }
    }

    [Fact]
    public void LongSideRope_IsThePerAxisRopeOnASquareGrid()
    {
        (Tensor cosA, Tensor sinA) = Dinov3RopeTables.Build(1, 5, 4, 4, 64, roundPeriodsToBf16: false);
        (Tensor cosB, Tensor sinB) = Dinov3RopeTables.Build(1, 5, 4, 4, 64, roundPeriodsToBf16: false, normalizeByLongSide: true);
        try
        {
            Assert.Equal(cosA.AsReadOnlySpan<float>().ToArray(), cosB.AsReadOnlySpan<float>().ToArray());
            Assert.Equal(sinA.AsReadOnlySpan<float>().ToArray(), sinB.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            cosA.Dispose(); sinA.Dispose(); cosB.Dispose(); sinB.Dispose();
        }
    }

    [Fact]
    public void LongSideRope_DividesBothAxesByTheLongerSide()
    {
        const int prefix = 5, gridH = 2, gridW = 4, headDim = 64;
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, prefix, gridH, gridW, headDim, roundPeriodsToBf16: false, normalizeByLongSide: true);
        try
        {
            float[] periods = Dinov3RopeTables.Periods(headDim, roundToBf16: false);
            // Patch (row 1, column 3) of a 2x4 grid: both coordinates use the length 4, so the row is 2*(1.5/4)-1 = -0.25 (per-axis would give +0.5) and the column 0.75.
            int token = prefix + 1 * gridW + 3;
            ReadOnlySpan<float> c = cos.AsReadOnlySpan<float>(), s = sin.AsReadOnlySpan<float>();
            for (int p = 0; p < headDim / 2; p++)
            {
                double coord = p < headDim / 4 ? -0.25 : 0.75;
                double angle = 6.2831855 * coord / periods[p % (headDim / 4)];
                Assert.Equal(Math.Cos(angle), c[token * headDim + p], 4);
                Assert.Equal(Math.Sin(angle), s[token * headDim + p], 4);
                Assert.Equal(c[token * headDim + p], c[token * headDim + p + headDim / 2]);
            }
        }
        finally
        {
            cos.Dispose(); sin.Dispose();
        }
    }

    [Fact]
    public void LongSideRope_ChangesANonSquareImageAndNotASquareOne()
    {
        Dinov2Preset on = TimmPreset(gated: false), off = on with { RopeNormalizesByLongSide = false };
        Dictionary<string, Tensor> timm = TimmWeights(on, seed: 20, prefix: "");
        Dictionary<string, Tensor> hf = Dinov3TimmKeys.ToHuggingFace(timm, "", on);
        try
        {
            Dinov3VisionEncoder encOn = new(on), encOff = new(off);
            encOn.LoadWeights(hf);
            encOff.LoadWeights(hf);
            using Tensor wide = Pixels(1, 32, 64, seed: 21);
            using Tensor square = Pixels(1, 64, 64, seed: 22);

            using Tensor wideOn = encOn.Encode(Backend, wide);
            using Tensor wideOff = encOff.Encode(Backend, wide);
            using Tensor squareOn = encOn.Encode(Backend, square);
            using Tensor squareOff = encOff.Encode(Backend, square);

            Assert.True(MaxAbsDiff(wideOn.AsReadOnlySpan<float>(), wideOff.AsReadOnlySpan<float>()) > 1e-4);
            Assert.Equal(squareOn.AsReadOnlySpan<float>().ToArray(), squareOff.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            Dispose(timm);
        }
    }

    // A complete timm-layout checkpoint for a preset (fused qkv without bias unless the preset has one, gamma_1/2, fc1 or fc1_g/fc1_x, reg_token, no final norm unless
    // the preset has one). Norm weights are 1 and biases 0 as in the HF fixture; the rest is seeded noise.
    private static Dictionary<string, Tensor> TimmWeights(Dinov2Preset preset, int seed, string prefix)
    {
        Random rng = new(seed);
        int h = preset.HiddenSize, inter = preset.IntermediateSize, p = preset.PatchSize;
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        void Fill(string key, float scale, params long[] shape)
        {
            Tensor t = new(new TensorShape(shape), DType.F32);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
            w[prefix + key] = t;
        }
        void Ones(string key, long n)
        {
            Tensor t = new(new TensorShape(n), DType.F32);
            t.AsSpan<float>().Fill(1f);
            w[prefix + key] = t;
        }
        void Zeros(string key, long n) => w[prefix + key] = new Tensor(new TensorShape(n), DType.F32);

        Fill("patch_embed.proj.weight", 0.05f, h, 3, p, p);
        Fill("patch_embed.proj.bias", 0.05f, h);
        Fill("cls_token", 0.5f, 1, 1, h);
        Fill("reg_token", 0.5f, 1, preset.NumRegisterTokens, h);
        for (int i = 0; i < preset.NumLayers; i++)
        {
            string b = $"blocks.{i}.";
            Ones(b + "norm1.weight", h); Zeros(b + "norm1.bias", h);
            Fill(b + "attn.qkv.weight", 0.05f, 3 * h, h);
            if (preset.AttentionQkvBias) Fill(b + "attn.qkv.bias", 0.05f, 3 * h);
            Fill(b + "attn.proj.weight", 0.05f, h, h); Fill(b + "attn.proj.bias", 0.05f, h);
            Fill(b + "gamma_1", 0.1f, h);
            Ones(b + "norm2.weight", h); Zeros(b + "norm2.bias", h);
            if (preset.GatedMlp)
            {
                Fill(b + "mlp.fc1_g.weight", 0.05f, inter, h); Fill(b + "mlp.fc1_g.bias", 0.05f, inter);
                Fill(b + "mlp.fc1_x.weight", 0.05f, inter, h); Fill(b + "mlp.fc1_x.bias", 0.05f, inter);
            }
            else
            {
                Fill(b + "mlp.fc1.weight", 0.05f, inter, h); Fill(b + "mlp.fc1.bias", 0.05f, inter);
            }

            Fill(b + "mlp.fc2.weight", 0.05f, h, inter); Fill(b + "mlp.fc2.bias", 0.05f, h);
            Fill(b + "gamma_2", 0.1f, h);
        }

        if (preset.HasFinalNorm) { Ones("norm.weight", h); Zeros("norm.bias", h); }
        return w;
    }

    // The same weights renamed by hand and copied (no views, no adapter) into the HF layout, with an explicit zero q and v bias per layer: the reference the adapter is checked against.
    private static Dictionary<string, Tensor> HuggingFaceCopy(IReadOnlyDictionary<string, Tensor> timm, string prefix, Dinov2Preset preset)
    {
        int h = preset.HiddenSize;
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        Tensor Copy(string from, string to, long rowOffset = 0, long rows = -1)
        {
            Tensor src = timm[prefix + from];
            long total = src.Shape[0];
            long n = rows < 0 ? total : rows;
            long rowElems = src.ElementCount / total;
            long[] dims = [.. Enumerable.Range(0, src.Shape.Rank).Select(i => src.Shape[i])];
            dims[0] = n;
            Tensor t = new(new TensorShape(dims), DType.F32);
            src.AsReadOnlySpan<float>().Slice((int)(rowOffset * rowElems), (int)(n * rowElems)).CopyTo(t.AsSpan<float>());
            w[to] = t;
            return t;
        }

        Copy("patch_embed.proj.weight", "embeddings.patch_embeddings.weight");
        Copy("patch_embed.proj.bias", "embeddings.patch_embeddings.bias");
        Copy("cls_token", "embeddings.cls_token");
        Copy("reg_token", "embeddings.register_tokens");
        for (int i = 0; i < preset.NumLayers; i++)
        {
            string b = $"blocks.{i}.", l = $"layer.{i}.";
            Copy(b + "norm1.weight", l + "norm1.weight"); Copy(b + "norm1.bias", l + "norm1.bias");
            Copy(b + "norm2.weight", l + "norm2.weight"); Copy(b + "norm2.bias", l + "norm2.bias");
            Copy(b + "attn.qkv.weight", l + "attention.q_proj.weight", 0, h);
            Copy(b + "attn.qkv.weight", l + "attention.k_proj.weight", h, h);
            Copy(b + "attn.qkv.weight", l + "attention.v_proj.weight", 2L * h, h);
            w[l + "attention.q_proj.bias"] = new Tensor(new TensorShape(h), DType.F32);
            w[l + "attention.v_proj.bias"] = new Tensor(new TensorShape(h), DType.F32);
            Copy(b + "attn.proj.weight", l + "attention.o_proj.weight"); Copy(b + "attn.proj.bias", l + "attention.o_proj.bias");
            Copy(b + "gamma_1", l + "layer_scale1.lambda1"); Copy(b + "gamma_2", l + "layer_scale2.lambda1");
            if (preset.GatedMlp)
            {
                Copy(b + "mlp.fc1_g.weight", l + "mlp.gate_proj.weight"); Copy(b + "mlp.fc1_g.bias", l + "mlp.gate_proj.bias");
                Copy(b + "mlp.fc1_x.weight", l + "mlp.up_proj.weight"); Copy(b + "mlp.fc1_x.bias", l + "mlp.up_proj.bias");
            }
            else
            {
                Copy(b + "mlp.fc1.weight", l + "mlp.up_proj.weight"); Copy(b + "mlp.fc1.bias", l + "mlp.up_proj.bias");
            }

            Copy(b + "mlp.fc2.weight", l + "mlp.down_proj.weight"); Copy(b + "mlp.fc2.bias", l + "mlp.down_proj.bias");
        }

        return w;
    }

    private static Tensor Pixels(int batch, int height, int width, int seed)
    {
        Tensor t = new(new TensorShape(batch, 3, height, width), DType.F32);
        Random rng = new(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static double MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static double MaxAbs(ReadOnlySpan<float> a)
    {
        double m = 0;
        foreach (float v in a) m = Math.Max(m, Math.Abs(v));
        return m;
    }

    private static void Dispose(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values) t.Dispose();
    }
}
