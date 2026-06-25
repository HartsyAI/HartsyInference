using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>CLIP-skip ("stop layer") behaviour for <see cref="ClipTextEncoder"/>. Uses a tiny
/// synthetic-weight encoder on the CPU backend so it runs without any downloaded checkpoint.
/// Verifies (a) different stop layers yield different hidden states, and (b) the default behaviour
/// is unchanged: <c>EncodePenultimate</c> with no argument equals an explicit <c>layersFromEnd: 2</c>.</summary>
public sealed class ClipSkipTests
{
    // Small config: enough layers that layer 1 vs 2 vs 3 of skip is meaningful.
    private static ClipTextEncoderConfig TinyConfig => new()
    {
        HiddenSize = 32,
        IntermediateSize = 64,
        NumLayers = 4,
        NumHeads = 4,
        MaxPositionEmbeddings = 16,
        VocabSize = 64,
        UseQuickGelu = true,
        ProjectionDim = 0,
    };

    [Fact]
    public void Encode_DifferentClipSkip_ProducesDifferentHiddenStates()
    {
        using CpuBackend backend = new CpuBackend();
        ClipTextEncoder encoder = BuildSyntheticEncoder(TinyConfig, seed: 1234);

        int[][] tokens = [[1, 5, 9, 13, 2]];

        Tensor last = encoder.Encode(backend, tokens, layersFromEnd: 1);
        Tensor penultimate = encoder.Encode(backend, tokens, layersFromEnd: 2);

        Assert.Equal(last.Shape.ToString(), penultimate.Shape.ToString());
        Assert.False(AllClose(last, penultimate, 1e-4f),
            "layersFromEnd=1 and layersFromEnd=2 should produce different hidden states.");

        last.Dispose();
        penultimate.Dispose();
    }

    [Fact]
    public void EncodePenultimate_DefaultEqualsExplicitTwo_RegressionGuard()
    {
        using CpuBackend backend = new CpuBackend();
        // text_projection present so pooled output path runs too.
        ClipTextEncoder encoder = BuildSyntheticEncoder(TinyConfig with { ProjectionDim = 32 }, seed: 99);

        int[][] tokens = [[1, 7, 3, 11, 2]];
        int[] eos = [4];

        (Tensor hiddenDefault, Tensor? pooledDefault) = encoder.EncodePenultimate(backend, tokens, eos);
        (Tensor hiddenTwo, Tensor? pooledTwo) = encoder.EncodePenultimate(backend, tokens, eos, layersFromEnd: 2);

        Assert.True(AllClose(hiddenDefault, hiddenTwo, 0f),
            "EncodePenultimate default must be byte-identical to explicit layersFromEnd: 2.");
        Assert.NotNull(pooledDefault);
        Assert.NotNull(pooledTwo);
        Assert.True(AllClose(pooledDefault!, pooledTwo!, 0f),
            "Pooled output must be unchanged by the default vs explicit penultimate call.");

        // And clip-skip 1 (final layer) must differ from the penultimate default.
        (Tensor hiddenOne, _) = encoder.EncodePenultimate(backend, tokens, eos, layersFromEnd: 1);
        Assert.False(AllClose(hiddenDefault, hiddenOne, 1e-4f),
            "Final-layer (layersFromEnd=1) hidden state should differ from penultimate.");

        hiddenDefault.Dispose();
        hiddenTwo.Dispose();
        hiddenOne.Dispose();
        pooledDefault!.Dispose();
        pooledTwo!.Dispose();
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    private static bool AllClose(Tensor a, Tensor b, float tol)
    {
        ReadOnlySpan<float> sa = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> sb = b.AsReadOnlySpan<float>();
        if (sa.Length != sb.Length) return false;
        for (int i = 0; i < sa.Length; i++)
        {
            if (MathF.Abs(sa[i] - sb[i]) > tol) return false;
        }
        return true;
    }

    private static ClipTextEncoder BuildSyntheticEncoder(ClipTextEncoderConfig config, int seed)
    {
        Dictionary<string, Tensor> w = new();
        int h = config.HiddenSize;
        int inter = config.IntermediateSize;

        uint state = (uint)seed;
        float Next() // small deterministic pseudo-random in [-0.1, 0.1]
        {
            state = state * 1664525u + 1013904223u;
            return ((state >> 8) / (float)(1 << 24) - 0.5f) * 0.2f;
        }

        Tensor Make(string name, params int[] dims)
        {
            Tensor t = new Tensor(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] = Next();
            w[name] = t;
            return t;
        }

        // LayerNorm weights default to ~1 so the norm is well-conditioned.
        Tensor MakeNormWeight(string name, int dim)
        {
            Tensor t = new Tensor(new TensorShape(dim), DType.F32);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] = 1.0f + Next();
            w[name] = t;
            return t;
        }

        Make("text_model.embeddings.token_embedding.weight", config.VocabSize, h);
        Make("text_model.embeddings.position_embedding.weight", config.MaxPositionEmbeddings, h);

        for (int i = 0; i < config.NumLayers; i++)
        {
            string p = $"text_model.encoder.layers.{i}";
            MakeNormWeight($"{p}.layer_norm1.weight", h);
            Make($"{p}.layer_norm1.bias", h);
            Make($"{p}.self_attn.q_proj.weight", h, h);
            Make($"{p}.self_attn.q_proj.bias", h);
            Make($"{p}.self_attn.k_proj.weight", h, h);
            Make($"{p}.self_attn.k_proj.bias", h);
            Make($"{p}.self_attn.v_proj.weight", h, h);
            Make($"{p}.self_attn.v_proj.bias", h);
            Make($"{p}.self_attn.out_proj.weight", h, h);
            Make($"{p}.self_attn.out_proj.bias", h);
            MakeNormWeight($"{p}.layer_norm2.weight", h);
            Make($"{p}.layer_norm2.bias", h);
            Make($"{p}.mlp.fc1.weight", inter, h);
            Make($"{p}.mlp.fc1.bias", inter);
            Make($"{p}.mlp.fc2.weight", h, inter);
            Make($"{p}.mlp.fc2.bias", h);
        }

        MakeNormWeight("text_model.final_layer_norm.weight", h);
        Make("text_model.final_layer_norm.bias", h);

        if (config.ProjectionDim > 0)
        {
            Make("text_projection.weight", config.ProjectionDim, h);
        }

        ClipTextEncoder encoder = new ClipTextEncoder(config);
        encoder.LoadWeights(w);
        return encoder;
    }
}
