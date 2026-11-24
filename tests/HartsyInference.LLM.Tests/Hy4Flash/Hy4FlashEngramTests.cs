using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The Engram layer against the Hugging Face classes of transformers PR 50903 at eb1d5db, from the committed tiny-config dumps (float32, CPU): the default and fp8 embedding
/// configs, the BF16 storage of the released small tensors, and the fp8 round trip inside the layer.</summary>
public sealed class Hy4FlashEngramTests(ITestOutputHelper output)
{
    private const float F32Eps = 1.1920929e-7f;

    private static string DataDir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data"))) dir = Path.GetDirectoryName(dir);
            return dir is null ? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data not found.") : Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data");
        }
    }

    private static int[] Ids(string name)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, name)));
        return [.. document.RootElement.GetProperty("input_ids").EnumerateArray().Select(static e => e.GetInt32())];
    }

    // The Engram weights of layer 1 (config id 2) as the checkpoint stores them, every tensor through `read`.
    private static Hy4FlashEngramWeights Weights(Hy4FlashConfig config, Func<string, Tensor> read)
    {
        int hc = config.HcMult;
        string e = "model.layers.1.engram_layer.";
        return new Hy4FlashEngramWeights
        {
            PerHeadNorms = [.. Enumerable.Range(0, config.EngramHashHeads).Select(i => read($"{e}lookup_per_head_norms.{i}.weight"))],
            ValueProj = read(e + "value_proj.weight"),
            ValueBias = read(e + "value_proj.bias"),
            KeyProjs = [.. Enumerable.Range(0, hc).Select(i => read($"{e}key_projs.{i}.weight"))],
            KeyBiases = [.. Enumerable.Range(0, hc).Select(i => read($"{e}key_projs.{i}.bias"))],
            NormKey = [.. Enumerable.Range(0, hc).Select(i => read($"{e}norm_key.{i}.weight"))],
            NormQuery = [.. Enumerable.Range(0, hc).Select(i => read($"{e}norm_query.{i}.weight"))],
            UnifiedPreGamma = read(e + "unified_pre_gamma"),
            PerHcScales = [.. Enumerable.Range(0, hc).Select(i => read($"{e}unified_per_hc_scales.{i}"))],
        };
    }

    private static Hy4FlashEngramLayer Layer(Hy4FlashConfig config, Hy4FlashEngramWeights weights, Tensor table, int[] lookup) =>
        new(new CpuBackend(), config, 2, weights, new Hy4FlashRowTable(table), lookup, F32Eps);

    private static float MaxAbs(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        return worst;
    }

    private static float MaxRelative(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        float scale = 1e-30f;
        foreach (float v in expected) scale = MathF.Max(scale, MathF.Abs(v));
        return MaxAbs(expected, actual) / scale;
    }

    // BF16 storage of an F32 tensor: every value rounded to BF16 (what the released checkpoint holds) and the same rounded values as F32.
    private static Tensor ToBf16(Tensor f32)
    {
        Tensor bf16 = new(f32.Shape, DType.BF16);
        ReadOnlySpan<float> source = f32.AsReadOnlySpan<float>();
        Span<ushort> bits = bf16.AsSpan<ushort>();
        for (int i = 0; i < source.Length; i++) bits[i] = TensorCasts.F32ToBf16Bits(source[i]);
        return bf16;
    }

    private static Tensor RoundedF32(Tensor bf16)
    {
        Tensor f32 = new(bf16.Shape, DType.F32);
        ReadOnlySpan<ushort> bits = bf16.AsReadOnlySpan<ushort>();
        Span<float> values = f32.AsSpan<float>();
        for (int i = 0; i < bits.Length; i++) values[i] = BFloat16Bits.ToFloat(bits[i]);
        return f32;
    }

    [Theory]
    [InlineData("text", "inputs_text.json")]
    [InlineData("image", "inputs_image.json")]
    public void Residual_MatchesTheHfEngramOutput(string prefix, string inputs)
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        Hy4FlashEngramWeights weights = Weights(config, name => set.GetTensor(name));
        int[] lookup = set.GetTensor("model.layers.1.engram_layer.lookup_table").AsReadOnlySpan<int>().ToArray();
        using Hy4FlashEngramLayer engram = Layer(config, weights, set.GetTensor("model.layers.1.engram_layer.multi_head_embedding.embedding.weight"), lookup);

        int[] ids = Ids(inputs);
        (int[] ngram, bool[] mask) = Hy4FlashEngramLayer.BuildInputs(config, ids, 0, ids.Length);
        (long[] shape, float[] stream) = NpyReader.ReadFloat32(Path.Combine(DataDir, $"{prefix}__layer_0_out.npy"));
        Assert.Equal([1L, ids.Length, config.HcMult, config.HiddenSize], shape);
        (_, float[] expected) = NpyReader.ReadFloat32(Path.Combine(DataDir, $"{prefix}__engram_1_out.npy"));

        float[] actual = engram.Residual(stream, ids.Length, ngram, mask);
        float worst = MaxAbs(expected, actual);
        Assert.True(worst < 1e-4f, $"max difference {worst}");
        Assert.True(expected.Any(static v => v != 0f), "the HF output must not be all zeros");
        if (prefix == "image") Assert.Contains(false, mask);
    }

    [Fact]
    public void Fp8EmbeddingInsideTheLayer_MatchesTheHfRoundTripAndDiffersFromNone()
    {
        // The fp8 reference keeps the same weights (same seed) and turns engram_fp8_embedding on; its layer 0 output is the stream this layer reads.
        string fp8Dir = Path.Combine(DataDir, "engram_fp8");
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(fp8Dir, "config.json"));
        Assert.True(config.EngramFp8Embedding);
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        Hy4FlashEngramWeights weights = Weights(config, name => set.GetTensor(name));
        int[] lookup = set.GetTensor("model.layers.1.engram_layer.lookup_table").AsReadOnlySpan<int>().ToArray();
        Tensor table = set.GetTensor("model.layers.1.engram_layer.multi_head_embedding.embedding.weight");
        int[] ids = Ids("inputs_text.json");
        (int[] ngram, bool[] mask) = Hy4FlashEngramLayer.BuildInputs(config, ids, 0, ids.Length);
        (_, float[] stream) = NpyReader.ReadFloat32(Path.Combine(fp8Dir, "text__layer_0_out.npy"));
        (_, float[] expected) = NpyReader.ReadFloat32(Path.Combine(fp8Dir, "text__engram_1_out.npy"));

        using Hy4FlashEngramLayer withRoundTrip = Layer(config, weights, table, lookup);
        float[] actual = withRoundTrip.Residual(stream, ids.Length, ngram, mask);
        float worst = MaxAbs(expected, actual);
        output.WriteLine($"fp8 layer vs HF fp8 reference: max difference {worst:E2}");
        Assert.True(worst < 1e-4f, $"max difference {worst}");

        // The same layer without the round trip must land measurably away from the reference, so the in-layer placement is what this test exercises.
        Hy4FlashConfig plain = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using Hy4FlashEngramLayer withoutRoundTrip = Layer(plain, weights, table, lookup);
        float[] unrounded = withoutRoundTrip.Residual(stream, ids.Length, ngram, mask);
        float difference = MaxRelative(expected, unrounded);
        output.WriteLine($"no round trip vs HF fp8 reference: max relative difference {difference:E2}");
        Assert.True(difference > 1e-3f, $"the round trip does not change the output enough to be tested: {difference:E2}");
    }

    [Fact]
    public void Bf16SmallTensors_GiveTheSameResidualAsTheirRoundedF32Values()
    {
        // The released checkpoint stores the Engram gains, biases, scales and projections in BF16. The layer must read them as their values, not as reinterpreted floats.
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        int[] lookup = set.GetTensor("model.layers.1.engram_layer.lookup_table").AsReadOnlySpan<int>().ToArray();
        Tensor table = set.GetTensor("model.layers.1.engram_layer.multi_head_embedding.embedding.weight");

        List<Tensor> bf16 = [], f32 = [];
        Hy4FlashEngramWeights weights16 = Weights(config, name => { Tensor t = ToBf16(set.GetTensor(name)); bf16.Add(t); return t; });
        Hy4FlashEngramWeights weights32 = Weights(config, name => { Tensor t = RoundedF32(ToBf16(set.GetTensor(name))); f32.Add(t); return t; });
        try
        {
            Assert.All(new[] { weights16.UnifiedPreGamma, weights16.ValueBias, weights16.PerHcScales[0] }, static t => Assert.Equal(DType.BF16, t.DType));
            int[] ids = Ids("inputs_text.json");
            (int[] ngram, bool[] mask) = Hy4FlashEngramLayer.BuildInputs(config, ids, 0, ids.Length);
            (_, float[] stream) = NpyReader.ReadFloat32(Path.Combine(DataDir, "text__layer_0_out.npy"));
            using Hy4FlashEngramLayer layer16 = Layer(config, weights16, table, lookup);
            using Hy4FlashEngramLayer layer32 = Layer(config, weights32, table, lookup);
            float[] a = layer16.Residual(stream, ids.Length, ngram, mask);
            float[] b = layer32.Residual(stream, ids.Length, ngram, mask);
            float worst = MaxRelative(b, a);
            output.WriteLine($"BF16 storage vs its rounded F32 values: max relative difference {worst:E2}");
            Assert.True(worst < 1e-6f, $"{worst:E2}");
        }
        finally
        {
            foreach (Tensor t in bf16.Concat(f32)) t.Dispose();
        }
    }

    [Fact]
    public void NegativeLookupEntry_IsRefusedAtConstruction()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        int[] lookup = set.GetTensor("model.layers.1.engram_layer.lookup_table").AsReadOnlySpan<int>().ToArray();
        lookup[7] = -1;
        Assert.Throws<ArgumentException>(() => Layer(config, Weights(config, name => set.GetTensor(name)),
            set.GetTensor("model.layers.1.engram_layer.multi_head_embedding.embedding.weight"), lookup));
    }

    [Fact]
    public void NaNInTheStream_StaysNaNInItsResidualRow()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        int[] lookup = set.GetTensor("model.layers.1.engram_layer.lookup_table").AsReadOnlySpan<int>().ToArray();
        using Hy4FlashEngramLayer layer = Layer(config, Weights(config, name => set.GetTensor(name)),
            set.GetTensor("model.layers.1.engram_layer.multi_head_embedding.embedding.weight"), lookup);
        int[] ids = Ids("inputs_text.json");
        (int[] ngram, bool[] mask) = Hy4FlashEngramLayer.BuildInputs(config, ids, 0, ids.Length);
        (_, float[] stream) = NpyReader.ReadFloat32(Path.Combine(DataDir, "text__layer_0_out.npy"));
        int row = config.HcMult * config.HiddenSize;
        stream[0 * row] = float.NaN;
        float[] residual = layer.Residual(stream, ids.Length, ngram, mask);
        Assert.True(residual.Take(row).Any(float.IsNaN), "a NaN in the stream must reach its residual row");
    }

    [Fact]
    public void RoundE4M3_BisectionEqualsTheNearestCodeByBruteForce()
    {
        Random random = new(4);
        for (int i = 0; i < 20000; i++)
        {
            float value = (float)((random.NextDouble() * 2 - 1) * (i % 3 == 0 ? 600 : 40));
            float expected = BruteForceNearest(value);
            Assert.Equal(expected, Hy4FlashFp8Rounding.RoundE4M3(value));
        }
    }

    // e4m3fn value of a non-negative code (0..126): exponent bias 7, subnormal below exponent 1.
    private static float DecodeE4M3(int code)
    {
        int exponent = code >> 3, mantissa = code & 7;
        return exponent == 0 ? mantissa / 8f * MathF.Pow(2f, -6f) : (1f + mantissa / 8f) * MathF.Pow(2f, exponent - 7);
    }

    // Every finite code, the nearest value, ties to the even code, saturation at the largest finite value.
    private static float BruteForceNearest(float value)
    {
        float magnitude = MathF.Min(MathF.Abs(value), 448f);
        float best = 0f;
        float bestDistance = float.PositiveInfinity;
        int bestCode = 0;
        for (int code = 0; code < 0x7F; code++)
        {
            float v = DecodeE4M3(code);
            float distance = MathF.Abs(v - magnitude);
            if (distance < bestDistance || (distance == bestDistance && (code & 1) == 0 && (bestCode & 1) != 0))
            {
                best = v;
                bestDistance = distance;
                bestCode = code;
            }
        }
        return value < 0f ? -best : best;
    }

    [Fact]
    public void Fp8RoundTrip_MatchesTorch()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Hy4Flash", "fp8_roundtrip_cases.json")));
        JsonElement root = document.RootElement;
        JsonElement.ArrayEnumerator inputs = root.GetProperty("inputs").EnumerateArray(), quantized = root.GetProperty("quantized").EnumerateArray();
        int vectors = 0;
        while (inputs.MoveNext() && quantized.MoveNext())
        {
            float[] x = [.. inputs.Current.EnumerateArray().Select(static v => (float)v.GetDouble())];
            float[] expected = [.. quantized.Current.EnumerateArray().Select(static v => (float)v.GetDouble())];
            Hy4FlashFp8Rounding.RoundTripVector(x);
            for (int i = 0; i < x.Length; i++) Assert.Equal(expected[i], x[i], 1e-6f * MathF.Max(1f, MathF.Abs(expected[i])));
            vectors++;
        }
        Assert.Equal(5, vectors);
    }

    [Fact]
    public void BuildInputs_PadsBeforeTheStartAndMasksImageAndPadTokens()
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        int[] ids = [7, config.EngramPadId, config.ImageTokenId, 9];
        (int[] ngram, bool[] mask) = Hy4FlashEngramLayer.BuildInputs(config, ids, 0, 4);
        int pad = config.EngramPadId, image = config.ImageTokenId;
        Assert.Equal([7, pad, pad, pad, 7, pad, image, pad, 7, 9, image, pad], ngram);
        Assert.Equal([true, false, false, true], mask);
        (int[] tail, _) = Hy4FlashEngramLayer.BuildInputs(config, ids, 3, 1);
        Assert.Equal([9, config.ImageTokenId, config.EngramPadId], tail);
    }
}
