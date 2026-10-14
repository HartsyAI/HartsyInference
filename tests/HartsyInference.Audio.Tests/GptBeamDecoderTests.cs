using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary><see cref="GptBeamDecoder"/>'s per-beam KV caches (moved or copied on every reorder) must give the same
/// beam search as recomputing each beam's whole sequence without a cache, on a tiny random GPT.</summary>
public sealed unsafe class GptBeamDecoderTests : IDisposable
{
    private const int Hidden = 16;
    private const int Vocab = 24;
    private const int Eos = Vocab - 1;
    private const int PrefixLength = 5;

    private static readonly GptConfig Cfg = new() { Hidden = Hidden, NumLayers = 2, NumHeads = 2, BlockSize = 64, Bias = true };

    private readonly List<Tensor> _owned = [];
    private readonly CpuBackend _backend = new();

    private Tensor Rand(Random rng, double amp, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * amp);
        _owned.Add(t);
        return t;
    }

    private GptBackbone BuildGpt(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        for (int i = 0; i < Cfg.NumLayers; i++)
        {
            string p = $"h.{i}";
            w[$"{p}.layernorm_1.weight"] = Rand(rng, 0.1, Hidden);
            foreach (ref float v in w[$"{p}.layernorm_1.weight"].AsSpan<float>()) v += 1f;
            w[$"{p}.layernorm_1.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.attn.att_proj.weight"] = Rand(rng, 0.3, 3 * Hidden, Hidden);
            w[$"{p}.attn.att_proj.bias"] = Rand(rng, 0.02, 3 * Hidden);
            w[$"{p}.attn.out_proj.weight"] = Rand(rng, 0.3, Hidden, Hidden);
            w[$"{p}.attn.out_proj.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.layernorm_2.weight"] = Rand(rng, 0.1, Hidden);
            foreach (ref float v in w[$"{p}.layernorm_2.weight"].AsSpan<float>()) v += 1f;
            w[$"{p}.layernorm_2.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.mlp.in_proj.weight"] = Rand(rng, 0.3, Cfg.MlpDim, Hidden);
            w[$"{p}.mlp.in_proj.bias"] = Rand(rng, 0.02, Cfg.MlpDim);
            w[$"{p}.mlp.out_proj.weight"] = Rand(rng, 0.3, Hidden, Cfg.MlpDim);
            w[$"{p}.mlp.out_proj.bias"] = Rand(rng, 0.02, Hidden);
        }
        w["ln_f.weight"] = Rand(rng, 0.1, Hidden);
        foreach (ref float v in w["ln_f.weight"].AsSpan<float>()) v += 1f;
        w["ln_f.bias"] = Rand(rng, 0.02, Hidden);
        GptBackbone gpt = new(Cfg);
        gpt.LoadWeights(w, posKey: null, blockPrefix: "h", lnFGammaKey: "ln_f.weight", lnFBetaKey: "ln_f.bias");
        return gpt;
    }

    private sealed class Tables(Tensor embed, Tensor pos, Tensor head)
    {
        public Tensor EmbedStep(int token, int step)
        {
            Tensor t = new(new TensorShape(1, 1, Hidden), DType.F32);
            Span<float> dst = t.AsSpan<float>();
            ReadOnlySpan<float> e = embed.AsSpan<float>().Slice(token * Hidden, Hidden);
            ReadOnlySpan<float> p = pos.AsSpan<float>().Slice((PrefixLength + step) * Hidden, Hidden);
            for (int c = 0; c < Hidden; c++) dst[c] = e[c] + p[c];
            return t;
        }

        public Tensor Logits(Tensor hidden)
        {
            Tensor t = new(new TensorShape(1, 1, Vocab), DType.F32);
            Span<float> dst = t.AsSpan<float>();
            ReadOnlySpan<float> h = hidden.AsSpan<float>();
            ReadOnlySpan<float> w = head.AsSpan<float>();
            for (int v = 0; v < Vocab; v++)
            {
                float sum = 0f;
                for (int c = 0; c < Hidden; c++) sum += w[v * Hidden + c] * h[c];
                dst[v] = sum;
            }
            return t;
        }
    }

    // Recomputes every beam's full sequence each step: no cache to reorder.
    private sealed class Recompute(CpuBackend backend, GptBackbone gpt, Tensor prefix, Tables tables, int beams) : IBeamSearchModel
    {
        private List<int>[] _fed = [.. Enumerable.Range(0, beams).Select(_ => new List<int>())];

        public void Step(int beam, int token, int step, Span<float> logits)
        {
            _fed[beam].Add(token);
            int t = PrefixLength + _fed[beam].Count;
            using Tensor seq = new(new TensorShape(1, t, Hidden), DType.F32);
            prefix.AsSpan<float>().CopyTo(seq.AsSpan<float>());
            for (int i = 0; i < _fed[beam].Count; i++)
            {
                using Tensor row = tables.EmbedStep(_fed[beam][i], i);
                row.AsSpan<float>().CopyTo(seq.AsSpan<float>().Slice((PrefixLength + i) * Hidden, Hidden));
            }
            using Tensor output = gpt.Forward(backend, seq, nonCausal: false, cache: null, positionsApplied: true);
            using Tensor last = new(new TensorShape(1, 1, Hidden), DType.F32);
            output.AsSpan<float>().Slice((t - 1) * Hidden, Hidden).CopyTo(last.AsSpan<float>());
            using Tensor result = tables.Logits(last);
            result.AsSpan<float>().CopyTo(logits);
        }

        public void Reorder(ReadOnlySpan<int> parents)
        {
            List<int>[] next = new List<int>[parents.Length];
            for (int b = 0; b < parents.Length; b++) next[b] = [.. _fed[parents[b]]];
            _fed = next;
        }
    }

    [Theory]
    [InlineData(3, 1.0f)]
    [InlineData(4, 0.0f)]
    public void CachedBeams_MatchRecomputedBeams(int beams, float lengthPenalty)
    {
        Random rng = new(5);
        using GptBackbone gpt = BuildGpt(rng);
        Tables tables = new(Rand(rng, 1.0, Vocab, Hidden), Rand(rng, 0.3, 64, Hidden), Rand(rng, 1.0, Vocab, Hidden));
        Tensor prefix = Rand(rng, 1.0, 1, PrefixLength, Hidden);
        BeamSearchSettings settings = new()
        {
            NumBeams = beams, MaxNewTokens = 14, EosToken = Eos, RepetitionPenalty = 1.5f, LengthPenalty = lengthPenalty,
        };

        uint a = 1, b = 1;
        int[] expected = BeamSearch.Generate(new Recompute(_backend, gpt, prefix, tables, beams), settings, [Eos - 1], Eos - 1, Vocab, ref a);
        using GptBeamDecoder cached = new(_backend, gpt, beams, prefix, tables.EmbedStep, tables.Logits);
        int[] actual = BeamSearch.Generate(cached, settings, [Eos - 1], Eos - 1, Vocab, ref b);

        Assert.NotEmpty(actual);
        Assert.Equal(expected, actual);
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}
