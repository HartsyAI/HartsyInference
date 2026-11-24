using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Hy4Flash;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The checkpoint weight source: block-scaled e4m3 decoding (against torch's own float8 decoder), dtype handling of dense tensors, and an FP8 checkpoint against its exact
/// F32 twin, the values it decodes to. The FP8 model is not compared with the unquantized HF dump, which differs by quantization error.</summary>
public sealed class Hy4FlashCheckpointTests(ITestOutputHelper output)
{
    private static string DataDir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data"))) dir = Path.GetDirectoryName(dir);
            return dir is null ? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data not found.") : Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data");
        }
    }

    private static float[] Floats(string base64)
    {
        byte[] bytes = Convert.FromBase64String(base64);
        float[] values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static float MaxRelative(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        float scale = 1e-30f, worst = 0f;
        foreach (float v in expected) scale = MathF.Max(scale, MathF.Abs(v));
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]) / scale);
        return worst;
    }

    private static IEnumerable<string> QuantizedNames(int layers, int moeFrom, int visionLayers)
    {
        for (int v = 0; v < visionLayers; v++) yield return $"vit.layers.{v}.self_attn.o_proj.weight";
        for (int l = 0; l < layers; l++)
        {
            yield return $"model.layers.{l}.self_attn.o_proj.weight";
            yield return $"model.layers.{l}.self_attn.q_b_proj.weight";
            yield return $"model.layers.{l}.self_attn.linear_gate.weight";
            if (l < moeFrom) continue;
            yield return $"model.layers.{l}.mlp.experts.gate_up_proj";
            yield return $"model.layers.{l}.mlp.experts.down_proj";
        }
    }

    [Fact]
    public void BlockDequant_MatchesTorchsFloat8Decoder_IncludingPartialEdgeBlocks()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, "fp8", "block_case.json")));
        JsonElement root = document.RootElement;
        long[] shape = [.. root.GetProperty("shape").EnumerateArray().Select(static e => e.GetInt64())];
        byte[] codes = Convert.FromBase64String(root.GetProperty("raw_b64").GetString()!);
        float[] scale = Floats(root.GetProperty("scale_b64").GetString()!);
        float[] expected = Floats(root.GetProperty("expected_b64").GetString()!);

        float[] actual = Hy4FlashFp8.DequantizeBlocks(codes, shape, scale);
        Assert.Equal(expected.Length, actual.Length);
        float worst = MaxRelative(expected, actual);
        output.WriteLine($"block dequant max relative difference {worst:E2} over {actual.Length} values in 4 blocks (2 partial)");
        Assert.True(worst < 1e-7f, $"{worst:E2}");
    }

    [Fact]
    public void BlockDequant_RefusesAScaleGridOfTheWrongSize()
    {
        byte[] codes = new byte[130 * 150];
        float[] scale = new float[3];
        Assert.Throws<InvalidDataException>(() => Hy4FlashFp8.DequantizeBlocks(codes, [1, 130, 150], scale));
    }

    [Fact]
    public void DenseTensors_KeepTheirValuesAndDtypes()
    {
        using Hy4FlashCheckpoint checkpoint = Hy4FlashCheckpoint.Open(Path.Combine(DataDir));
        using Hy4FlashCheckpoint fp8 = Hy4FlashCheckpoint.Open(Path.Combine(DataDir, "fp8"));
        using Tensor lookup = checkpoint.Read("model.layers.1.engram_layer.lookup_table");
        Assert.Equal(DType.I32, lookup.DType);
        Assert.Equal(250, lookup.ElementCount);

        const string norm = "model.layers.0.input_layernorm.weight";
        using Tensor a = checkpoint.Read(norm), b = fp8.Read(norm);
        Assert.Equal(DType.F32, b.DType);
        Assert.Equal(0f, MaxRelative(a.AsReadOnlySpan<float>(), b.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void Fp8Checkpoint_DequantizesEveryQuantizedTensorToTheTwinExactly()
    {
        using Hy4FlashCheckpoint fp8 = Hy4FlashCheckpoint.Open(Path.Combine(DataDir, "fp8"));
        using Hy4FlashCheckpoint twin = Hy4FlashCheckpoint.Open(Path.Combine(DataDir, "fp8_twin"));
        Assert.True(fp8.Config.IsFp8);
        Assert.False(twin.Config.IsFp8);
        int count = 0;
        foreach (string name in QuantizedNames(fp8.Config.NumHiddenLayers, moeFrom: 1, visionLayers: 3))
        {
            using Tensor decoded = fp8.Read(name);
            using Tensor reference = twin.Read(name);
            Assert.Equal(DType.F32, decoded.DType);
            Assert.Equal(reference.Shape.ToString(), decoded.Shape.ToString());
            float worst = MaxRelative(reference.AsReadOnlySpan<float>(), decoded.AsReadOnlySpan<float>());
            Assert.True(worst < 1e-7f, $"{name}: {worst:E2}");
            count++;
        }
        Assert.Equal(16, count);
    }

    [Fact]
    public void Fp8Model_MatchesItsTwinModel_OverTheWholeSequence()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, "inputs_text.json")));
        int[] ids = [.. document.RootElement.GetProperty("input_ids").EnumerateArray().Select(static e => e.GetInt32())];
        CpuBackend backend = new();
        using Hy4FlashCheckpoint fp8 = Hy4FlashCheckpoint.Open(Path.Combine(DataDir, "fp8"));
        using Hy4FlashCheckpoint twin = Hy4FlashCheckpoint.Open(Path.Combine(DataDir, "fp8_twin"));
        using Hy4FlashModel quantized = new(backend, fp8.Config, fp8, 64);
        using Hy4FlashModel reference = new(backend, twin.Config, twin, 64);
        using Hy4FlashSequenceState a = quantized.CreateState(), b = reference.CreateState();
        float[] got = quantized.Forward(ids, a, allRows: true);
        float[] want = reference.Forward(ids, b, allRows: true);
        float worst = MaxRelative(want, got);
        output.WriteLine($"fp8 model vs its F32 twin, logits: max relative difference {worst:E2}");
        Assert.True(worst < 1e-5f, $"{worst:E2}");

        using JsonDocument hf = JsonDocument.Parse(File.ReadAllText(Path.Combine(DataDir, "inputs_text.json")));
        float[] dump = NpyReader.ReadFloat32(Path.Combine(DataDir, "text__logits.npy")).Data;
        output.WriteLine($"fp8 model vs the unquantized HF dump, logits: max relative difference {MaxRelative(dump, got):E2} (quantization error, not asserted)");
    }
}
