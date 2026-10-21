using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>T5 and UMT5 never scale the QK product (HF `scaling = 1.0`); the relative position bias carries the scale.</summary>
public sealed class T5AttentionScaleTests
{
    private const int DModel = 8, Heads = 2, HeadDim = 4, Seq = 5;

    public static TheoryData<string> Presets()
    {
        TheoryData<string> names = new();
        foreach (PropertyInfo p in typeof(T5TextEncoderConfig).GetProperties(BindingFlags.Public | BindingFlags.Static))
            if (p.PropertyType == typeof(T5TextEncoderConfig)) names.Add(p.Name);
        return names;
    }

    [Theory]
    [MemberData(nameof(Presets))]
    public void Preset_IsUnscaled(string preset) =>
        Assert.Equal(1f, ((T5TextEncoderConfig)typeof(T5TextEncoderConfig).GetProperty(preset)!.GetValue(null)!).AttentionScale);

    [Fact]
    public void DefaultBlock_MatchesAnUnscaledBlock()
    {
        Random rng = new(3);
        Tensor T(int n, params long[] dims)
        {
            Tensor t = new(new TensorShape(dims), DType.F32);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < n; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
            return t;
        }
        int inner = Heads * HeadDim;
        Dictionary<string, Tensor> w = new()
        {
            ["b.layer.0.layer_norm.weight"] = T(DModel, DModel),
            ["b.layer.0.SelfAttention.q.weight"] = T(inner * DModel, inner, DModel),
            ["b.layer.0.SelfAttention.k.weight"] = T(inner * DModel, inner, DModel),
            ["b.layer.0.SelfAttention.v.weight"] = T(inner * DModel, inner, DModel),
            ["b.layer.0.SelfAttention.o.weight"] = T(DModel * inner, DModel, inner),
            ["b.layer.1.layer_norm.weight"] = T(DModel, DModel),
            ["b.layer.1.DenseReluDense.wi_0.weight"] = T(16 * DModel, 16, DModel),
            ["b.layer.1.DenseReluDense.wi_1.weight"] = T(16 * DModel, 16, DModel),
            ["b.layer.1.DenseReluDense.wo.weight"] = T(DModel * 16, DModel, 16),
        };
        T5Block byDefault = new(DModel, Heads, HeadDim, 1e-6f), unscaled = new(DModel, Heads, HeadDim, 1e-6f, attentionScale: 1f),
            scaled = new(DModel, Heads, HeadDim, 1e-6f, attentionScale: 0.5f);
        byDefault.LoadWeights(w, "b");
        unscaled.LoadWeights(w, "b");
        scaled.LoadWeights(w, "b");
        using CpuBackend backend = new();
        using Tensor input = T(Seq * DModel, 1, Seq, DModel);
        using Tensor bias = T(Heads * Seq * Seq, 1, Heads, Seq, Seq);
        using Tensor a = byDefault.Forward(backend, input, bias, null), b = unscaled.Forward(backend, input, bias, null);
        using Tensor c = scaled.Forward(backend, input, bias, null);
        Assert.Equal(b.AsReadOnlySpan<float>().ToArray(), a.AsReadOnlySpan<float>().ToArray());
        Assert.NotEqual(c.AsReadOnlySpan<float>().ToArray(), a.AsReadOnlySpan<float>().ToArray());
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
