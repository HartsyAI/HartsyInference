using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The load-time pass for ComfyUI <c>asym_w4a8_int8</c> layers: it reads the descriptor's <c>group_size</c> and
/// ConvRot group, decodes the weight to dense BF16, drops every companion, and leaves other layers alone. Layer naming
/// follows Comfy-Org's <c>qwen3.5_9b_w4a8_gptq.safetensors</c> header (<c>.weight</c>, <c>.weight_s_rel</c>,
/// <c>.weight_s_channel</c>, <c>.weight_codebook</c>, <c>.comfy_quant</c>).</summary>
public sealed class W4A8CheckpointPassTests
{
    [Fact]
    public void Descriptor_ReadsTheGroupSize()
    {
        ComfyQuantDescriptor? d = ComfyQuantDescriptor.TryParse(
            Encoding.UTF8.GetBytes("{\"format\": \"asym_w4a8_int8\", \"group_size\": 16, \"convrot\": true, \"convrot_groupsize\": 256}"));

        Assert.NotNull(d);
        Assert.Equal((W4A8ConvRotCodec.Format, 16, 256), (d!.Format, d.GroupSize, d.ConvRotGroupSize));
        Assert.Equal(16, ComfyQuantDescriptor.TryParse(d.Serialize())!.GroupSize);
    }

    [Fact]
    public void Pass_DecodesTheLayerAndDropsItsCompanions()
    {
        Dictionary<string, Tensor> source = Layer("mlp.down_proj", group: 4, codes: [[0, 15, 3, 8, 1, 2, 14, 7]]);
        source["norm.weight"] = F32([1f, 2f], 2);

        Dictionary<string, Tensor> result = CheckpointConvertUtils.DequantW4A8(source);

        Assert.Equal(new[] { "mlp.down_proj.weight", "norm.weight" }, result.Keys.Order(StringComparer.Ordinal));
        Tensor weight = result["mlp.down_proj.weight"];
        Assert.Equal(DType.BF16, weight.DType);
        Assert.Equal(new TensorShape(1, 8), weight.Shape);
        using Tensor f32 = weight.CastTo(DType.F32);
        // codebook[code] = code - 8, s_rel 2 per group, s_channel 0.5: q = (code - 8)·2, w = q·0.5 = code - 8.
        Assert.Equal(new float[] { -8, 7, -5, 0, -7, -6, 6, -1 }, f32.AsReadOnlySpan<float>().ToArray());
        weight.Dispose();
        Dispose(source);
    }

    [Fact]
    public void Pass_LeavesADictionaryWithoutW4A8Unchanged()
    {
        Dictionary<string, Tensor> source = new(StringComparer.Ordinal) { ["a.weight"] = F32([1f], 1) };

        Assert.Same(source, CheckpointConvertUtils.DequantW4A8(source));
        Dispose(source);
    }

    [Fact]
    public void Pass_RefusesALayerMissingAScale()
    {
        Dictionary<string, Tensor> source = Layer("q_proj", group: 4, codes: [[1, 2, 3, 4, 5, 6, 7, 8]]);
        source.Remove("q_proj.weight_s_channel", out Tensor? removed);
        removed!.Dispose();

        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => CheckpointConvertUtils.DequantW4A8(source));
        Assert.Contains("q_proj", ex.Message);
        Dispose(source);
    }

    private static Dictionary<string, Tensor> Layer(string name, int group, int[][] codes)
    {
        int n = codes.Length, k = codes[0].Length;
        Tensor packed = new Tensor(new TensorShape(n, k / 2), DType.I8);
        Span<byte> bytes = packed.AsSpan<byte>();
        for (int r = 0; r < n; r++)
            for (int c = 0; c < k; c += 2)
                bytes[r * (k / 2) + c / 2] = (byte)(codes[r][c] | (codes[r][c + 1] << 4));
        byte[] json = Encoding.UTF8.GetBytes($"{{\"format\": \"asym_w4a8_int8\", \"group_size\": {group}, \"convrot\": false}}");
        Tensor descriptor = new Tensor(new TensorShape(json.Length), DType.U8);
        json.CopyTo(descriptor.AsSpan<byte>());
        return new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            [$"{name}.weight"] = packed,
            [$"{name}.weight_s_rel"] = F32(Enumerable.Repeat(2f, n * k / group).ToArray(), n, k / group),
            [$"{name}.weight_s_channel"] = F32(Enumerable.Repeat(0.5f, n).ToArray(), n),
            [$"{name}.weight_codebook"] = F32(Enumerable.Range(0, 16).Select(i => i - 8f).ToArray(), 16),
            [$"{name}.comfy_quant"] = descriptor,
        };
    }

    private static Tensor F32(float[] values, params long[] dims)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static void Dispose(Dictionary<string, Tensor> tensors)
    {
        foreach (Tensor t in tensors.Values)
            t.Dispose();
    }
}
