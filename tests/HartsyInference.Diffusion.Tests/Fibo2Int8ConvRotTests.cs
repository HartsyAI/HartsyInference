using HartsyInference.Engine;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Comfy-Org Fibo 2.0 int8 ConvRot builds on the CPU: a quantized fused weight row-splits into valid quantized weights
/// (the per-row scale slices with the rows, and ConvRot acts on the input dimension a row split leaves alone), a widened int8 checkpoint
/// matches the dense run within quantization error, and the dense builds still load. Synthetic tensors only; the real files are not read.</summary>
public sealed unsafe class Fibo2Int8ConvRotTests
{
    private const int Dim = 64;
    private const int CapFeat = 96;
    private const int Ffn = 96;

    private static readonly string FixturePath = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Fibo2", "fibo2_tiny.safetensors");

    /// <summary>Row-splitting the fused QKV, to_kv and w13 of a packed weight dequantizes to exactly the matching rows of the packed
    /// fused weight's own dequant: the split is lossless, not merely close.</summary>
    [Fact]
    public void RowSplit_OfPackedFusedWeights_DequantizesToTheFusedRows()
    {
        List<Tensor> owned = [];
        try
        {
            Tensor qkv = Pack(Random(3 * Dim, Dim, 1), 64, owned);
            (Tensor q, Tensor k, Tensor v) = ZImageBlock.SplitQkv(qkv, Dim);
            owned.AddRange([q, k, v]);
            AssertRowsMatch(qkv, [q, k, v], [Dim, Dim, Dim]);

            Tensor toKv = Pack(Random(2 * Dim, CapFeat, 2), 16, owned);
            Tensor[] keyValue = CheckpointConvertUtils.SplitRows(toKv, [Dim, Dim], ["to_kv.k.weight", "to_kv.v.weight"]);
            owned.AddRange(keyValue);
            AssertRowsMatch(toKv, keyValue, [Dim, Dim]);

            Tensor w13 = Pack(Random(2 * Ffn, Dim, 3), 64, owned);
            Tensor[] gateLinear = CheckpointConvertUtils.SplitRows(w13, [Ffn, Ffn], ["w1.weight", "w3.weight"]);
            owned.AddRange(gateLinear);
            AssertRowsMatch(w13, gateLinear, [Ffn, Ffn]);
        }
        finally
        {
            foreach (Tensor tensor in owned)
                tensor.Dispose();
        }
    }

    /// <summary>A per-tensor scale (one element) covers every row, so each split shares it rather than slicing a scale that is not there.</summary>
    [Fact]
    public void RowSplit_OfAPerTensorScale_SharesTheScalar()
    {
        List<Tensor> owned = [];
        try
        {
            Tensor fused = Pack(Random(3 * Dim, Dim, 4), 64, owned);
            Tensor scalar = new(new TensorShape(1), DType.F32);
            scalar.AsSpan<float>()[0] = 0.01f;
            owned.Add(scalar);
            fused.QuantInfo = fused.QuantInfo! with { RowScale = scalar };

            (Tensor q, Tensor k, Tensor v) = ZImageBlock.SplitQkv(fused, Dim);
            owned.AddRange([q, k, v]);
            Assert.Same(scalar, q.QuantInfo!.RowScale);
            Assert.Same(scalar, v.QuantInfo!.RowScale);
            AssertRowsMatch(fused, [q, k, v], [Dim, Dim, Dim]);
        }
        finally
        {
            foreach (Tensor tensor in owned)
                tensor.Dispose();
        }
    }

    /// <summary>The whole denoiser with every block, refiner and Perceiver linear packed int8 ConvRot (as the 215 in the real file), widened
    /// for the CPU, tracks the dense run within quantization error.</summary>
    [Fact]
    public void Forward_WithPackedLinearsWidenedForTheCpu_TracksTheDenseRun()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(FixturePath);
        Dictionary<string, Tensor> all = loader.GetAllTensors();
        Dictionary<string, Tensor> dense = new(StringComparer.Ordinal);
        foreach ((string key, Tensor value) in all)
        {
            if (key.StartsWith("weights.", StringComparison.Ordinal))
                dense[key["weights.".Length..]] = value;
        }
        Fibo2Config config = Fibo2Config.FromWeights(dense, TinyTemplate());
        float sigma = all["inputs.sigma"].AsReadOnlySpan<float>()[0];
        CpuBackend backend = new();

        using Fibo2Transformer reference = new(config);
        reference.LoadWeights(dense);
        using Tensor expected = reference.Forward(backend, all["inputs.latent"], sigma, all["inputs.context"]);

        List<IDisposable> owned = [];
        try
        {
            Dictionary<string, Tensor> packed = new(dense, StringComparer.Ordinal);
            int quantized = 0;
            foreach ((string key, Tensor weight) in dense)
            {
                if (!IsQuantizedLinear(key))
                    continue;
                Tensor pack = Pack(weight, LargestGroup((int)weight.Shape[1]), owned);
                packed[key] = pack;
                quantized++;
            }
            // 2 blocks + 1 refiner: qkv, out, w13, w2 each, plus the injection block's cross attention (to_q, to_kv, to_out);
            // 1 Perceiver layer: cross + self attention (3 each) and w13, w2.
            Assert.Equal(3 * 4 + 3 + 3 + 3 + 2, quantized);

            Dictionary<string, Tensor> prepared = Fibo2WeightPreparation.HalfPrecisionWeights(packed, owned);
            Assert.True(Fibo2WeightPreparation.HasPackedWeights(prepared));
            Fibo2WeightPreparation.WidenUnsupported(prepared, [backend], DType.F32, owned);
            Assert.False(Fibo2WeightPreparation.HasPackedWeights(prepared));

            using Fibo2Transformer widened = new(config);
            widened.LoadWeights(prepared);
            using Tensor actual = widened.Forward(backend, all["inputs.latent"], sigma, all["inputs.context"]);

            Assert.Equal(expected.Shape, actual.Shape);
            ReadOnlySpan<float> want = expected.AsReadOnlySpan<float>();
            ReadOnlySpan<float> got = actual.AsReadOnlySpan<float>();
            float scale = 0f;
            float worst = 0f;
            for (int i = 0; i < want.Length; i++)
            {
                scale = MathF.Max(scale, MathF.Abs(want[i]));
                worst = MathF.Max(worst, MathF.Abs(want[i] - got[i]));
            }
            Assert.True(scale > 0f);
            Assert.True(worst <= QuantizedTolerance * scale,
                $"int8 ConvRot velocity max |Δ| = {worst} exceeds {QuantizedTolerance} of its scale {scale}");
            Assert.NotEqual(0f, worst);
        }
        finally
        {
            for (int i = owned.Count - 1; i >= 0; i--)
                owned[i].Dispose();
        }
    }

    /// <summary>Per-row int8 rounds each weight by up to half of 1/127 of its row's range, and that compounds through five blocks and the
    /// Perceiver; a missed scale or an unrotated ConvRot weight misses by the full magnitude.</summary>
    private const float QuantizedTolerance = 0.05f;

    /// <summary>A dense build is unchanged: BF16 casts to F16, F32 and F16 pass through, nothing is packed.</summary>
    [Fact]
    public void HalfPrecisionWeights_DenseBuild_CastsBf16AndPassesTheRest()
    {
        List<IDisposable> owned = [];
        try
        {
            Tensor bf16 = Random(4, 8, 5).CastTo(DType.BF16);
            owned.Add(bf16);
            Tensor f32 = Random(2, 2, 6);
            owned.Add(f32);
            Dictionary<string, Tensor> prepared = Fibo2WeightPreparation.HalfPrecisionWeights(
                new Dictionary<string, Tensor> { ["a.weight"] = bf16, ["b.weight"] = f32 }, owned);

            Assert.Equal(DType.F16, prepared["a.weight"].DType);
            Assert.Same(f32, prepared["b.weight"]);
            Assert.False(Fibo2WeightPreparation.HasPackedWeights(prepared));
        }
        finally
        {
            foreach (IDisposable item in owned)
                item.Dispose();
        }
    }

    [Fact]
    public void HalfPrecisionWeights_DenseFixtureCheckpoint_LoadsEveryTensor()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(FixturePath);
        Dictionary<string, Tensor> weights = [];
        foreach ((string key, Tensor value) in loader.GetAllTensors())
        {
            if (key.StartsWith("weights.", StringComparison.Ordinal))
                weights[key["weights.".Length..]] = value;
        }
        List<IDisposable> owned = [];
        try
        {
            Dictionary<string, Tensor> prepared = Fibo2WeightPreparation.HalfPrecisionWeights(weights, owned);
            Assert.Equal(weights.Count, prepared.Count);
            Assert.False(Fibo2WeightPreparation.HasPackedWeights(prepared));
            Fibo2Config.FromWeights(prepared, TinyTemplate());
        }
        finally
        {
            foreach (IDisposable item in owned)
                item.Dispose();
        }
    }

    /// <summary>An int8 matrix without its scale, or any other quantized dtype, refuses rather than loading as garbage.</summary>
    [Fact]
    public void HalfPrecisionWeights_RefusesAnInt8WeightWithoutItsScale()
    {
        using Tensor bare = new(new TensorShape(4, 4), DType.I8);
        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            Fibo2WeightPreparation.HalfPrecisionWeights(new Dictionary<string, Tensor> { ["x.weight"] = bare }, []));
        Assert.Contains("x.weight", error.Message);
        Assert.Contains("int8", error.Message);
    }

    [Theory]
    [InlineData("fibo2-dev-int8", "diffusion_models/fibo2_dev_int8_convrot.safetensors", "063dd397395809fd22234c6bd32932cfcb22619f4d2282bdbf1dc58c406fe325")]
    [InlineData("fibo2-edit-turbo-int8", "diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors", "1f6fa7c499191f405467123f655f11762661eaa5bd479d1da8f161398ef72c3b")]
    public void Catalog_PinsTheInt8Builds(string id, string repoPath, string sha256)
    {
        CatalogEntry entry = ModelCatalog.Find(id)!;
        Assert.NotNull(entry);
        ModelAsset transformer = Assert.Single(entry.Assets, asset => asset.Role == "transformer");
        Assert.Equal("Comfy-Org/Fibo-2.0", transformer.Repo);
        Assert.Equal(repoPath, transformer.RepoPath);
        Assert.Equal(sha256, transformer.Sha256);
        ModelAsset encoder = Assert.Single(entry.Assets, asset => asset.Role == "text encoder");
        Assert.Equal("text_encoders/qwen3vl_4b_int8_convrot.safetensors", encoder.RepoPath);
        Assert.Equal("32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c", encoder.Sha256);
        Assert.True(new Fibo2Recipe().Matches(id));
    }

    [Theory]
    [InlineData("fibo2_dev_int8_convrot.safetensors", "dev")]
    [InlineData("fibo2_edit_turbo_int8_convrot.safetensors", "edit-turbo")]
    public void Variants_ResolveTheInt8FileNames(string file, string variantId)
    {
        CheckpointProbe probe = new(new HashSet<string>(StringComparer.Ordinal), new Dictionary<string, TensorShape>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), [Path.GetFileNameWithoutExtension(file)]);
        Assert.Equal(variantId, ModelVariantResolver.Classify(Fibo2Variants.Catalog, probe, []).Variant.Id);
    }

    private static void AssertRowsMatch(Tensor fused, Tensor[] pieces, int[] rows)
    {
        using Tensor whole = Int8ConvRotCodec.DequantToBf16(fused, fused.QuantInfo!.RowScale!, fused.QuantInfo.ConvRotGroupSize);
        using Tensor wholeF32 = whole.CastTo(DType.F32);
        ReadOnlySpan<float> all = wholeF32.AsReadOnlySpan<float>();
        long offset = 0;
        for (int i = 0; i < pieces.Length; i++)
        {
            Tensor piece = pieces[i];
            Assert.Equal(DType.I8, piece.DType);
            QuantWeightInfo info = piece.QuantInfo!;
            Assert.Equal(fused.QuantInfo!.ConvRotGroupSize, info.ConvRotGroupSize);
            using Tensor dequant = Int8ConvRotCodec.DequantToBf16(piece, info.RowScale!, info.ConvRotGroupSize);
            using Tensor dequantF32 = dequant.CastTo(DType.F32);
            ReadOnlySpan<float> got = dequantF32.AsReadOnlySpan<float>();
            Assert.Equal((long)rows[i] * fused.Shape[1], got.Length);
            Assert.True(all.Slice((int)(offset * fused.Shape[1]), got.Length).SequenceEqual(got), $"piece {i} differs from the fused rows");
            offset += rows[i];
        }
    }

    /// <summary>Packs a copy of <paramref name="source"/> as int8_tensorwise with ConvRot <paramref name="group"/>; the packed weight and its
    /// scale join <paramref name="owned"/>.</summary>
    private static Tensor Pack(Tensor source, int group, List<Tensor> owned)
    {
        (Tensor weight, Tensor scale) = Quantize(source, group);
        owned.Add(weight);
        owned.Add(scale);
        return weight;
    }

    private static Tensor Pack(Tensor source, int group, List<IDisposable> owned)
    {
        (Tensor weight, Tensor scale) = Quantize(source, group);
        owned.Add(weight);
        owned.Add(scale);
        return weight;
    }

    private static (Tensor Weight, Tensor Scale) Quantize(Tensor source, int group)
    {
        using Tensor copy = new(source.Shape, DType.F32);
        source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        (Tensor weight, Tensor scale) = Int8ConvRotCodec.QuantizeFromF32(copy, group);
        weight.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, ConvRotGroupSize = group };
        return (weight, scale);
    }

    private static Tensor Random(int rows, int columns, int seed)
    {
        Random rng = new(seed);
        Tensor tensor = new(new TensorShape(rows, columns), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return tensor;
    }

    private static int LargestGroup(int inFeatures)
    {
        int group = 256;
        while (group > 4 && inFeatures % group != 0)
            group /= 4;
        return group;
    }

    /// <summary>The linears the real file packs: block, refiner and Perceiver attention, cross-attention and FFN. Embedders, adaLN and the
    /// final layer stay dense.</summary>
    private static bool IsQuantizedLinear(string key)
    {
        if (!key.EndsWith(".weight", StringComparison.Ordinal) || key.Contains("adaLN", StringComparison.Ordinal))
            return false;
        string[] packed = ["attention.qkv", "attention.out", "to_q", "to_kv", "to_out", "w13", "w2"];
        return packed.Any(name => key.Contains("." + name + ".", StringComparison.Ordinal));
    }

    private static Fibo2Config TinyTemplate() => Fibo2Config.Checkpoint with
    {
        InChannels = 32,
        PatchSize = 2,
        AxesDims = [8, 12, 12],
        RopeTheta = 256f,
        NormEps = 1e-5f,
        TimestepScale = 1000f,
        MaxGist = 8,
        MinGist = 4,
        GistStep = 2,
        GistMinTextLen = 16,
        GistMaxTextLen = 64,
    };
}
