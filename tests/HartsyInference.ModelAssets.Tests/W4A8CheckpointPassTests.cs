using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The load-time pass for ComfyUI <c>asym_w4a8_int8</c> layers: it reads the descriptor's <c>group_size</c> and
/// ConvRot group, decodes the weight to its INT8 grid (or to dense BF16 when a correction term rules that out), drops every
/// companion, and leaves other layers alone. Layer naming
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
        // No correction term: the layer stays on its INT8 grid, an int8_tensorwise weight scaled by s_channel.
        Assert.Equal((DType.I8, new TensorShape(1, 8)), (weight.DType, weight.Shape));
        Assert.Equal(("int8_tensorwise", 0), (weight.QuantInfo!.Format, weight.QuantInfo.ConvRotGroupSize));
        Assert.Same(source["mlp.down_proj.weight_s_channel"], weight.QuantInfo.RowScale);
        // codebook[code] = code - 8, s_rel 2 per group: q = (code - 8)·2; s_channel 0.5 makes w = code - 8.
        Assert.Equal(new sbyte[] { -16, 14, -10, 0, -14, -12, 12, -2 }, weight.AsReadOnlySpan<sbyte>().ToArray());
        using Tensor bf16 = Int8ConvRotCodec.DequantToBf16(weight, weight.QuantInfo.RowScale!, 0);
        using Tensor f32 = bf16.CastTo(DType.F32);
        Assert.Equal(new float[] { -8, 7, -5, 0, -7, -6, 6, -1 }, f32.AsReadOnlySpan<float>().ToArray());
        weight.Dispose();
        Dispose(source);
    }

    [Fact]
    public void Pass_DecodesALayerWithACorrectionToBf16()
    {
        Dictionary<string, Tensor> source = Layer("mlp.down_proj", group: 4, codes: [[0, 15, 3, 8, 1, 2, 14, 7]]);
        source["mlp.down_proj.weight_correction"] = F32([0.5f, -1f], 2, 1); // [groups, N]

        Dictionary<string, Tensor> result = CheckpointConvertUtils.DequantW4A8(source);

        Tensor weight = result["mlp.down_proj.weight"];
        Assert.False(result.ContainsKey("mlp.down_proj.weight_correction"));
        Assert.Equal((DType.BF16, (QuantWeightInfo?)null), (weight.DType, weight.QuantInfo));
        using Tensor f32 = weight.CastTo(DType.F32);
        Assert.Equal(new float[] { -7.5f, 7.5f, -4.5f, 0.5f, -8, -7, 5, -2 }, f32.AsReadOnlySpan<float>().ToArray());
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

    [Fact]
    public void Pass_ReportsEarlierDecodesWhenALaterLayerRefuses()
    {
        Dictionary<string, Tensor> source = Layer("a_proj", group: 4, codes: [[1, 2, 3, 4, 5, 6, 7, 8]]);
        foreach ((string key, Tensor tensor) in Layer("b_proj", group: 4, codes: [[1, 2, 3, 4, 5, 6, 7, 8]]))
            source[key] = tensor;
        source.Remove("b_proj.weight_s_channel", out Tensor? removed);
        removed!.Dispose();
        List<Tensor> allocated = [];

        Assert.Throws<InvalidDataException>(() => CheckpointConvertUtils.DequantW4A8(source, allocated));

        // The caller owns what was decoded before the refusal, so it can free it.
        Tensor decoded = Assert.Single(allocated);
        Assert.Equal((DType.I8, new TensorShape(1, 8)), (decoded.DType, decoded.Shape));
        decoded.Dispose();
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
