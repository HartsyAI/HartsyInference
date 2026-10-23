using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 LoRA support on the tiny fixture: a synthetic adapter in each spelling (the ComfyUI-native one on the fused
/// modules with <c>.diff</c>/<c>.diff_b</c> deltas, and the diffusers/PEFT one on the split projections) must merge to exactly
/// <c>base + strength·(alpha/rank)·up·down</c>, landing in the right rows of the fused <c>qkv</c>, <c>to_kv</c> and <c>w13</c>,
/// with F16 base storage as the recipe holds it. No public Fibo 2.0 trainer output was available, so these files follow the key
/// layout of Comfy-Org's published turbo LoRA header and of diffusers' <c>transformer_bria_fibo2.py</c> module names.</summary>
public sealed unsafe class Fibo2LoraTests : IDisposable
{
    private const int Dim = 64;
    private static readonly string FixturePath = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Fibo2", "fibo2_tiny.safetensors");

    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"fibo2_lora_tests_{Guid.NewGuid():N}");
    private readonly List<Tensor> _owned = [];
    private readonly SafeTensorsLoader _loader = new();
    private readonly Dictionary<string, Tensor> _all;
    private readonly Random _rng = new(1234);

    public Fibo2LoraTests()
    {
        Directory.CreateDirectory(_dir);
        _loader.Load(FixturePath);
        _all = _loader.GetAllTensors();
    }

    public void Dispose()
    {
        foreach (Tensor tensor in _owned)
            tensor.Dispose();
        _loader.Dispose();
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    [Fact]
    public void ComfyNativeFile_MergesFusedAndFullWeightTargets()
    {
        Dictionary<string, Tensor> tensors = [];
        Dictionary<string, float> alphas = [];
        AddLora(tensors, "diffusion_model.layers.0.attention.qkv", 3 * Dim, Dim, rank: 3, alpha: 6f);
        AddLora(tensors, "diffusion_model.layers.0.feed_forward.w13", 192, Dim, rank: 2, alpha: null);
        AddLora(tensors, "diffusion_model.layers.0.feed_forward.w2", Dim, 96, rank: 2, alpha: 1f);
        AddLora(tensors, "diffusion_model.layers.1.cross_attention.to_kv", 128, 96, rank: 4, alpha: 4f);
        AddLora(tensors, "diffusion_model.perceiver.layers.0.ffn.w13", 192, Dim, rank: 2, alpha: null);
        AddLora(tensors, "diffusion_model.perceiver.layers.0.self_attention.to_kv", 128, Dim, rank: 2, alpha: null);
        AddLora(tensors, "diffusion_model.x_embedder", Dim, 128, rank: 2, alpha: null);
        tensors["diffusion_model.layers.0.adaLN_modulation.0.diff_b"] = Random(256);
        tensors["diffusion_model.layers.0.attention_norm1.diff"] = Random(Dim);
        string path = Save(tensors, "native");

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.Fibo2, file.Format);
        Assert.Equal(7, file.Layers.Count);
        Assert.Equal(2, file.FullWeightDiffs.Count);

        Dictionary<string, Tensor> weights = BaseWeights(half: false);
        Dictionary<string, Tensor> untouched = new(weights);
        const float strength = 0.7f;
        using MergedLoraStack? stack = RecipeLoraMerge.Apply(
            [new Engine.Features.LoraResolver.LoraSpec { FilePath = path, ModelStrength = strength, TencStrength = strength }],
            new CpuBackend(), new LoraMergeTargets { Transformer = weights }, "Fibo2LoraTests");
        Assert.NotNull(stack);

        foreach (string module in new[]
        {
            "layers.0.attention.qkv", "layers.0.feed_forward.w13", "layers.0.feed_forward.w2", "layers.1.cross_attention.to_kv",
            "perceiver.layers.0.ffn.w13", "perceiver.layers.0.self_attention.to_kv", "x_embedder",
        })
        {
            AssertWeight(weights[module + ".weight"], untouched[module + ".weight"], tensors, "diffusion_model." + module, strength, 1e-5f);
        }
        AssertAdded(weights["layers.0.adaLN_modulation.0.bias"], untouched["layers.0.adaLN_modulation.0.bias"],
            tensors["diffusion_model.layers.0.adaLN_modulation.0.diff_b"], strength, 1e-5f);
        AssertAdded(weights["layers.0.attention_norm1.weight"], untouched["layers.0.attention_norm1.weight"],
            tensors["diffusion_model.layers.0.attention_norm1.diff"], strength, 1e-5f);
        // Everything the file does not name is the same tensor, not a copy.
        Assert.Same(untouched["layers.0.attention.out.weight"], weights["layers.0.attention.out.weight"]);
        Assert.Same(untouched["layers.0.attention_norm2.weight"], weights["layers.0.attention_norm2.weight"]);
    }

    [Fact]
    public void DiffusersFile_SplitProjectionsLandInTheirFusedRows()
    {
        Dictionary<string, Tensor> tensors = [];
        AddLora(tensors, "transformer.layers.0.attention.to_q", Dim, Dim, rank: 2, alpha: 2f);
        AddLora(tensors, "transformer.layers.0.attention.to_k", Dim, Dim, rank: 2, alpha: 4f);
        AddLora(tensors, "transformer.layers.0.attention.to_v", Dim, Dim, rank: 3, alpha: 3f);
        AddLora(tensors, "transformer.layers.0.attention.to_out.0", Dim, Dim, rank: 2, alpha: null);
        AddLora(tensors, "transformer.layers.0.feed_forward.w1", 96, Dim, rank: 2, alpha: null);
        AddLora(tensors, "transformer.layers.0.feed_forward.w3", 96, Dim, rank: 2, alpha: 1f);
        AddLora(tensors, "transformer.layers.0.feed_forward.w2", Dim, 96, rank: 2, alpha: null);
        AddLora(tensors, "transformer.layers.1.cross_attention.to_q", Dim, Dim, rank: 2, alpha: null);
        AddLora(tensors, "transformer.layers.1.cross_attention.to_k", Dim, 96, rank: 2, alpha: null);
        AddLora(tensors, "transformer.layers.1.cross_attention.to_v", Dim, 96, rank: 2, alpha: 8f);
        AddLora(tensors, "transformer.layers.1.cross_attention.to_out.0", Dim, Dim, rank: 2, alpha: null);
        AddLora(tensors, "transformer.perceiver.layers.0.self_attention.to_v", Dim, Dim, rank: 2, alpha: null);
        string path = Save(tensors, "diffusers");

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.Fibo2, file.Format);
        Assert.Contains(file.Layers, l => l.TargetKey == "layers.0.attention.out.weight");
        Assert.Contains(file.Layers, l => l.TargetKey == "layers.1.cross_attention.to_out.weight");

        Dictionary<string, Tensor> weights = BaseWeights(half: false);
        Dictionary<string, Tensor> untouched = new(weights);
        using MergedLoraStack stack = new();
        stack.AddFromPath(path, 1.0f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend());
        // Distinct weights modified: qkv, out, w13, w2, cross q/to_kv/out, perceiver self to_kv.
        Assert.Equal(8, merged);

        (string Fused, string Source, int Row)[] slices =
        [
            ("layers.0.attention.qkv", "layers.0.attention.to_q", 0),
            ("layers.0.attention.qkv", "layers.0.attention.to_k", Dim),
            ("layers.0.attention.qkv", "layers.0.attention.to_v", 2 * Dim),
            ("layers.0.feed_forward.w13", "layers.0.feed_forward.w1", 0),
            ("layers.0.feed_forward.w13", "layers.0.feed_forward.w3", 96),
            ("layers.1.cross_attention.to_kv", "layers.1.cross_attention.to_k", 0),
            ("layers.1.cross_attention.to_kv", "layers.1.cross_attention.to_v", Dim),
            ("perceiver.layers.0.self_attention.to_kv", "perceiver.layers.0.self_attention.to_v", Dim),
        ];
        foreach (IGrouping<string, (string Fused, string Source, int Row)> group in slices.GroupBy(x => x.Fused))
        {
            AssertParts(weights[group.Key + ".weight"], untouched[group.Key + ".weight"], tensors,
                [.. group.Select(x => ("transformer." + x.Source, x.Row, 1f))], 1e-5f);
        }
        AssertWeight(weights["layers.0.attention.out.weight"], untouched["layers.0.attention.out.weight"], tensors,
            "transformer.layers.0.attention.to_out.0", 1f, 1e-5f);
        AssertWeight(weights["layers.1.cross_attention.to_out.weight"], untouched["layers.1.cross_attention.to_out.weight"], tensors,
            "transformer.layers.1.cross_attention.to_out.0", 1f, 1e-5f);
        AssertWeight(weights["layers.1.cross_attention.to_q.weight"], untouched["layers.1.cross_attention.to_q.weight"], tensors,
            "transformer.layers.1.cross_attention.to_q", 1f, 1e-5f);
        // AssertParts also proves the rows no adapter named (perceiver to_kv's K half) kept their base values.
    }

    [Fact]
    public void HalfPrecisionBase_IsMergedInF32AndRoundedOnce()
    {
        Dictionary<string, Tensor> tensors = [];
        AddLora(tensors, "diffusion_model.layers.0.attention.qkv", 3 * Dim, Dim, rank: 3, alpha: 3f);
        AddLora(tensors, "diffusion_model.layers.0.feed_forward.w13", 192, Dim, rank: 2, alpha: null);
        tensors["diffusion_model.layers.0.adaLN_modulation.0.diff_b"] = Random(256);
        string path = Save(tensors, "half");

        Dictionary<string, Tensor> weights = BaseWeights(half: true);
        Dictionary<string, Tensor> untouched = new(weights);
        Assert.Equal(DType.F16, weights["layers.0.attention.qkv.weight"].DType);
        using MergedLoraStack stack = new();
        stack.AddFromPath(path, 1.0f);
        Assert.Equal(3, stack.ApplyTo(weights, LoraTarget.Transformer, new CpuBackend()));

        foreach (string module in new[] { "layers.0.attention.qkv", "layers.0.feed_forward.w13" })
        {
            Tensor merged = weights[module + ".weight"];
            Assert.Equal(DType.F16, merged.DType);
            // F16 rounds by ~5e-4 relative, so the bound is a few ulps; a merge that dropped the delta would miss by far more.
            AssertWeight(merged, untouched[module + ".weight"], tensors, "diffusion_model." + module, 1f, 2e-3f);
        }
        AssertAdded(weights["layers.0.adaLN_modulation.0.bias"], untouched["layers.0.adaLN_modulation.0.bias"],
            tensors["diffusion_model.layers.0.adaLN_modulation.0.diff_b"], 1f, 2e-3f);
    }

    [Fact]
    public void MergedWeights_DriveTheDenoiserLikeTheHandMergedOnes()
    {
        Dictionary<string, Tensor> tensors = [];
        AddLora(tensors, "diffusion_model.layers.0.attention.qkv", 3 * Dim, Dim, rank: 3, alpha: 6f);
        AddLora(tensors, "diffusion_model.layers.0.feed_forward.w13", 192, Dim, rank: 2, alpha: null);
        AddLora(tensors, "diffusion_model.layers.1.cross_attention.to_kv", 128, 96, rank: 2, alpha: null);
        string path = Save(tensors, "forward");

        Dictionary<string, Tensor> baseWeights = BaseWeights(half: false);
        Dictionary<string, Tensor> merged = new(baseWeights);
        using MergedLoraStack stack = new();
        stack.AddFromPath(path, 1.0f);
        CpuBackend backend = new();
        stack.ApplyTo(merged, LoraTarget.Transformer, backend);

        Dictionary<string, Tensor> expected = new(baseWeights);
        foreach (string module in new[] { "layers.0.attention.qkv", "layers.0.feed_forward.w13", "layers.1.cross_attention.to_kv" })
            expected[module + ".weight"] = HandMerged(baseWeights[module + ".weight"], tensors, "diffusion_model." + module);

        Fibo2Config config = Fibo2Config.FromWeights(baseWeights, TinyTemplate());
        float sigma = ((float*)_all["inputs.sigma"].DataPointer)[0];
        float[] velocityBase = Forward(config, baseWeights, backend, sigma);
        float[] velocityMerged = Forward(config, merged, backend, sigma);
        float[] velocityExpected = Forward(config, expected, backend, sigma);

        float scale = velocityExpected.Max(MathF.Abs);
        float lora = 0f, drift = 0f;
        for (int i = 0; i < velocityExpected.Length; i++)
        {
            lora = MathF.Max(lora, MathF.Abs(velocityExpected[i] - velocityBase[i]));
            drift = MathF.Max(drift, MathF.Abs(velocityExpected[i] - velocityMerged[i]));
        }
        Assert.True(lora > 1e-3f * scale, "the adapter must move the velocity, or this test proves nothing");
        Assert.True(drift <= 1e-4f * scale, $"merged vs hand-merged velocity drifts by {drift} (scale {scale})");
    }

    [Fact]
    public void LoraThatMatchesNothing_IsRefused()
    {
        Dictionary<string, Tensor> tensors = [];
        AddLora(tensors, "diffusion_model.layers.0.feed_forward.w13", 192, Dim, rank: 2, alpha: null);
        string path = Save(tensors, "nomatch");
        Dictionary<string, Tensor> empty = [];
        Assert.Throws<NotSupportedException>(() => RecipeLoraMerge.Apply(
            [new Engine.Features.LoraResolver.LoraSpec { FilePath = path, ModelStrength = 1f, TencStrength = 1f }],
            new CpuBackend(), new LoraMergeTargets { Transformer = empty }, "Fibo2LoraTests"));
    }

    [Theory]
    [InlineData("layers.0.attention.to_out.0", ".weight", "layers.0.attention.out.weight")]
    [InlineData("layers.3.cross_attention.to_out.0", ".weight", "layers.3.cross_attention.to_out.weight")]
    [InlineData("perceiver.layers.1.self_attention.to_out.0", ".weight", "perceiver.layers.1.self_attention.to_out.weight")]
    [InlineData("layers.0.attention.norm_q", ".weight", "layers.0.attention.q_norm.weight")]
    [InlineData("layers.0.adaLN_modulation.0", ".bias", "layers.0.adaLN_modulation.0.bias")]
    [InlineData("layers.0.attention.qkv", ".weight", "layers.0.attention.qkv.weight")]
    public void MapBodyToCanonical_ProducesCheckpointKeys(string body, string suffix, string expected) =>
        Assert.Equal(expected, Fibo2LoraMapper.MapBodyToCanonical(body, suffix));

    [Fact]
    public void Detector_DoesNotClaimNeighbouringFamilies()
    {
        Assert.Equal(LoraFormat.ComfyZImageDit, LoraFormatDetector.Detect(Descriptors(
            "diffusion_model.layers.0.attention.to_q.lora_A.weight", "diffusion_model.layers.0.attention.to_q.lora_B.weight",
            "diffusion_model.layers.0.feed_forward.w1.lora_A.weight", "diffusion_model.layers.0.feed_forward.w1.lora_B.weight")));
        Assert.Equal(LoraFormat.Fibo2, LoraFormatDetector.Detect(Descriptors(
            "diffusion_model.layers.0.feed_forward.w13.lora_down.weight", "diffusion_model.layers.0.feed_forward.w13.lora_up.weight")));
        Assert.Equal(LoraFormat.Fibo2, LoraFormatDetector.Detect(Descriptors(
            "transformer.layers.6.cross_attention.to_q.lora_A.weight", "transformer.layers.6.cross_attention.to_q.lora_B.weight")));
    }

    // ---- helpers -------------------------------------------------------------------------------------------------------------

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> d = [];
        foreach (string key in keys)
            d[key] = new SafeTensorDescriptor { Name = key, DType = DType.F32, Shape = new TensorShape(4, 4), DataOffset = 0, ByteLength = 64 };
        return d;
    }

    /// <summary>The fixture's weights, optionally cast to F16 as <c>Fibo2Recipe</c> holds them.</summary>
    private Dictionary<string, Tensor> BaseWeights(bool half)
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach ((string key, Tensor value) in _all)
        {
            if (!key.StartsWith("weights.", StringComparison.Ordinal))
                continue;
            Tensor tensor = value;
            if (half && key.EndsWith(".weight", StringComparison.Ordinal) || half && key.EndsWith(".bias", StringComparison.Ordinal))
            {
                tensor = value.CastTo(DType.F16);
                _owned.Add(tensor);
            }
            weights[key["weights.".Length..]] = tensor;
        }
        return weights;
    }

    private Tensor Random(params int[] shape)
    {
        Tensor tensor = new(new TensorShape(shape.Select(s => (long)s).ToArray()), DType.F32);
        _owned.Add(tensor);
        float* p = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++)
            p[i] = (float)(_rng.NextDouble() * 0.2 - 0.1);
        return tensor;
    }

    private void AddLora(Dictionary<string, Tensor> tensors, string module, int outFeatures, int inFeatures, int rank, float? alpha)
    {
        bool peft = module.StartsWith("transformer.", StringComparison.Ordinal);
        tensors[module + (peft ? ".lora_A.weight" : ".lora_down.weight")] = Random(rank, inFeatures);
        tensors[module + (peft ? ".lora_B.weight" : ".lora_up.weight")] = Random(outFeatures, rank);
        if (alpha is float a)
        {
            Tensor scalar = new(new TensorShape(1), DType.F32);
            _owned.Add(scalar);
            *(float*)scalar.DataPointer = a;
            tensors[module + ".alpha"] = scalar;
        }
    }

    private string Save(Dictionary<string, Tensor> tensors, string name)
    {
        string path = Path.Combine(_dir, name + ".safetensors");
        SafeTensorsWriter.Save(path, tensors);
        return path;
    }

    private static float[] Read(Tensor tensor)
    {
        using Tensor f32 = tensor.CastTo(DType.F32);
        return new ReadOnlySpan<float>((void*)f32.DataPointer, (int)f32.ElementCount).ToArray();
    }

    /// <summary>The adapter's delta <c>(alpha/rank)·up·down</c> for <paramref name="module"/>, [out, in], in double precision.</summary>
    private static double[,] Delta(Dictionary<string, Tensor> tensors, string module)
    {
        bool peft = module.StartsWith("transformer.", StringComparison.Ordinal);
        float[] down = Read(tensors[module + (peft ? ".lora_A.weight" : ".lora_down.weight")]);
        float[] up = Read(tensors[module + (peft ? ".lora_B.weight" : ".lora_up.weight")]);
        Tensor downT = tensors[module + (peft ? ".lora_A.weight" : ".lora_down.weight")];
        int rank = (int)downT.Shape[0], inF = (int)downT.Shape[1], outF = up.Length / rank;
        double scale = tensors.TryGetValue(module + ".alpha", out Tensor? alpha) ? Read(alpha)[0] / (double)rank : 1.0;
        double[,] delta = new double[outF, inF];
        for (int o = 0; o < outF; o++)
            for (int i = 0; i < inF; i++)
            {
                double sum = 0;
                for (int r = 0; r < rank; r++)
                    sum += up[o * rank + r] * (double)down[r * inF + i];
                delta[o, i] = scale * sum;
            }
        return delta;
    }

    /// <summary>Asserts <paramref name="merged"/> equals <paramref name="baseWeight"/> plus every contribution's delta in the rows starting at its
    /// first row, and equals the base everywhere else.</summary>
    private static void AssertParts(Tensor merged, Tensor baseWeight, Dictionary<string, Tensor> tensors,
        (string Module, int FirstRow, float Strength)[] parts, float tolerance)
    {
        int rows = (int)baseWeight.Shape[0], cols = (int)baseWeight.Shape[1];
        float[] baseData = Read(baseWeight), mergedData = Read(merged);
        double[,] expected = new double[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                expected[r, c] = baseData[r * cols + c];
        foreach ((string module, int firstRow, float strength) in parts)
        {
            double[,] delta = Delta(tensors, module);
            Assert.True(firstRow + delta.GetLength(0) <= rows, "delta does not fit the weight");
            for (int r = 0; r < delta.GetLength(0); r++)
                for (int c = 0; c < cols; c++)
                    expected[firstRow + r, c] += strength * delta[r, c];
        }
        float scale = Math.Max(1f, baseData.Max(MathF.Abs));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.True(Math.Abs(expected[r, c] - mergedData[r * cols + c]) <= tolerance * scale,
                    $"[{r},{c}] expected {expected[r, c]} got {mergedData[r * cols + c]}");
    }

    private static void AssertWeight(Tensor merged, Tensor baseWeight, Dictionary<string, Tensor> tensors, string module, float strength,
        float tolerance) => AssertParts(merged, baseWeight, tensors, [(module, 0, strength)], tolerance);

    private static void AssertAdded(Tensor merged, Tensor baseWeight, Tensor diff, float strength, float tolerance)
    {
        float[] b = Read(baseWeight), m = Read(merged), d = Read(diff);
        float scale = Math.Max(1f, b.Max(MathF.Abs));
        for (int i = 0; i < b.Length; i++)
            Assert.True(Math.Abs(b[i] + strength * d[i] - m[i]) <= tolerance * scale, $"[{i}] {b[i]} + {strength}*{d[i]} != {m[i]}");
    }

    private Tensor HandMerged(Tensor baseWeight, Dictionary<string, Tensor> tensors, string module)
    {
        double[,] delta = Delta(tensors, module);
        Tensor result = baseWeight.CastTo(DType.F32);
        _owned.Add(result);
        float* p = (float*)result.DataPointer;
        int cols = (int)result.Shape[1];
        for (int r = 0; r < delta.GetLength(0); r++)
            for (int c = 0; c < cols; c++)
                p[r * cols + c] += (float)delta[r, c];
        return result;
    }

    private float[] Forward(Fibo2Config config, Dictionary<string, Tensor> weights, CpuBackend backend, float sigma)
    {
        using Fibo2Transformer transformer = new(config);
        transformer.LoadWeights(weights);
        using Tensor velocity = transformer.Forward(backend, _all["inputs.latent"], sigma, _all["inputs.context"]);
        return new ReadOnlySpan<float>((void*)velocity.DataPointer, (int)velocity.ElementCount).ToArray();
    }

    /// <summary>The fixture's dimensions, as in the parity test (tests/python-reference/fibo2_reference.py TINY).</summary>
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
