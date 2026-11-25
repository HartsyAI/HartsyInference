using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A runtime LoRA term on any LongCat linear must act exactly as the same delta merged into the weight, on every path the block takes:
/// the fused QKV, the windowed fused KV, the adaLN projections and the SwiGLU matrices.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAdjunctTests : IDisposable
{
    private const int Frames = 9, Height = 4, Width = 4, TextRows = 6, Rank = 3;
    private const float Scale = 0.6f;
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _base;

    public LongCatAdjunctTests() => _base = LongCatTestData.Weights(_config, 5);

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_base);
        _backend.Dispose();
    }

    public static TheoryData<string> Targets =>
    [
        "blocks.1.adaLN_modulation.1.weight", "blocks.0.attn.qkv.weight", "blocks.1.attn.proj.weight", "blocks.0.cross_attn.q_linear.weight",
        "blocks.1.cross_attn.kv_linear.weight", "blocks.0.cross_attn.proj.weight", "blocks.1.ffn.w1.weight", "blocks.0.ffn.w2.weight",
        "blocks.1.ffn.w3.weight", "final_layer.adaLN_modulation.1.weight", "final_layer.linear.weight", "t_embedder.mlp.0.weight",
        "y_embedder.y_proj.2.weight",
    ];

    [Theory]
    [MemberData(nameof(Targets))]
    public void AnActiveAdjunctEqualsTheMergedWeight(string key)
    {
        LowRankAdjunctTerm term = Term(key, 11, null);
        Dictionary<string, Tensor> withAdjunct = new(_base) { [key] = _base[key].WithLowRankAdjunct(new LowRankAdjunct { Terms = [term] }) };
        using Tensor merged = Merged(_base[key], term);
        Dictionary<string, Tensor> mergedWeights = new(_base) { [key] = merged };

        float[] expected = Run(mergedWeights);
        float[] actual = Run(withAdjunct);
        float[] untouched = Run(_base);

        Assert.True(LongCatTestData.RelRms(actual, expected) < 2e-5, $"{key}: adjunct differs from the merged weight");
        Assert.True(LongCatTestData.RelRms(actual, untouched) > 1e-4, $"{key}: the adjunct had no effect, so this path ignores it");
        term.Down.Dispose();
        term.Up!.Dispose();
        withAdjunct[key].Dispose();
    }

    [Fact]
    public void ADisabledGateIsTheBaseForwardBitForBit()
    {
        const string key = "blocks.0.attn.qkv.weight";
        AdjunctGate gate = new();
        LowRankAdjunctTerm term = Term(key, 11, gate);
        Dictionary<string, Tensor> gated = new(_base) { [key] = _base[key].WithLowRankAdjunct(new LowRankAdjunct { Terms = [term] }) };

        float[] off = Run(gated);
        float[] on;
        using (gate.Enable(1f))
            on = Run(gated);
        float[] baseline = Run(_base);

        Assert.Equal(baseline, off);
        Assert.True(LongCatTestData.RelRms(on, baseline) > 1e-4);
        term.Down.Dispose();
        term.Up!.Dispose();
        gated[key].Dispose();
    }

    private float[] Run(IReadOnlyDictionary<string, Tensor> weights)
    {
        using Tensor latent = LongCatTestData.Latent(_config, Frames, Height, Width, 2);
        using Tensor text = LongCatTestData.Encoder(_config, TextRows, 3);
        using LongCatTransformer transformer = new(_config);
        transformer.LoadWeights(weights);
        LongCatConditioning conditioning = new(NumCondLatents: 2);
        using Tensor velocity = transformer.Forward(_backend, latent, text, conditioning.FrameTimesteps(700f, Frames), in conditioning);
        return velocity.AsReadOnlySpan<float>().ToArray();
    }

    private LowRankAdjunctTerm Term(string key, int seed, AdjunctGate? gate)
    {
        TensorShape shape = _base[key].Shape;
        return new LowRankAdjunctTerm
        {
            Down = LongCatTestData.Random(new TensorShape(Rank, shape[1]), seed, 0.5f),
            Up = LongCatTestData.Random(new TensorShape(shape[0], Rank), seed + 1, 0.5f),
            Scale = Scale,
            Gate = gate,
        };
    }

    private static Tensor Merged(Tensor weight, LowRankAdjunctTerm term)
    {
        long rows = weight.Shape[0], cols = weight.Shape[1];
        Tensor merged = new(weight.Shape, DType.F32);
        ReadOnlySpan<float> w = weight.AsReadOnlySpan<float>(), down = term.Down.AsReadOnlySpan<float>(), up = term.Up!.AsReadOnlySpan<float>();
        Span<float> m = merged.AsSpan<float>();
        for (long r = 0; r < rows; r++)
            for (long c = 0; c < cols; c++)
            {
                double delta = 0;
                for (int k = 0; k < Rank; k++)
                    delta += up[(int)(r * Rank + k)] * (double)down[(int)(k * cols + c)];
                m[(int)(r * cols + c)] = (float)(w[(int)(r * cols + c)] + Scale * delta);
            }
        return merged;
    }
}
