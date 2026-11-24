using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The Engram layer and its fp8 round trip against the Hugging Face classes of transformers PR 50903 at eb1d5db, from the committed tiny-config dump (float32, CPU).</summary>
public sealed class Hy4FlashEngramTests
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

    [Theory]
    [InlineData("text", "inputs_text.json")]
    [InlineData("image", "inputs_image.json")]
    public void Residual_MatchesTheHfEngramOutput(string prefix, string inputs)
    {
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(DataDir, "config.json"));
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([Path.Combine(DataDir, "weights.safetensors")]);
        int layer = 1, layerId = 2, hc = config.HcMult;
        string e = $"model.layers.{layer}.engram_layer.";
        Tensor T(string name) => set.GetTensor(e + name);
        Hy4FlashEngramWeights weights = new()
        {
            PerHeadNorms = [.. Enumerable.Range(0, config.EngramHashHeads).Select(i => T($"lookup_per_head_norms.{i}.weight"))],
            ValueProj = T("value_proj.weight"),
            ValueBias = T("value_proj.bias"),
            KeyProjs = [.. Enumerable.Range(0, hc).Select(i => T($"key_projs.{i}.weight"))],
            KeyBiases = [.. Enumerable.Range(0, hc).Select(i => T($"key_projs.{i}.bias"))],
            NormKey = [.. Enumerable.Range(0, hc).Select(i => T($"norm_key.{i}.weight"))],
            NormQuery = [.. Enumerable.Range(0, hc).Select(i => T($"norm_query.{i}.weight"))],
            UnifiedPreGamma = T("unified_pre_gamma"),
            PerHcScales = [.. Enumerable.Range(0, hc).Select(i => T($"unified_per_hc_scales.{i}"))],
        };
        int[] lookup = T("lookup_table").AsReadOnlySpan<int>().ToArray();
        Hy4FlashEngramLayer engram = new(new CpuBackend(), config, layerId, weights, new Hy4FlashRowTable(T("multi_head_embedding.embedding.weight")), lookup, F32Eps);

        int[] ids = Ids(inputs);
        (int[] ngram, bool[] mask) = Hy4FlashEngramLayer.BuildInputs(config, ids, 0, ids.Length);
        (long[] shape, float[] stream) = NpyReader.ReadFloat32(Path.Combine(DataDir, $"{prefix}__layer_0_out.npy"));
        Assert.Equal([1L, ids.Length, hc, config.HiddenSize], shape);
        (_, float[] expected) = NpyReader.ReadFloat32(Path.Combine(DataDir, $"{prefix}__engram_1_out.npy"));

        float[] actual = engram.Residual(stream, ids.Length, ngram, mask);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        Assert.True(worst < 1e-4f, $"max difference {worst}");
        Assert.True(expected.Any(static v => v != 0f), "the HF output must not be all zeros");
        if (prefix == "image") Assert.Contains(false, mask);
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
