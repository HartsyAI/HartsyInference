using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Regression test for the Granite/MiniCPM graph-decode perf gap found auditing the LLM decode path:
/// <see cref="GenericTransformer.SupportsGraphDecode"/> used to exclude any model with
/// <see cref="TransformerConfig.EmbeddingScale"/> != 1.0 outright (Granite-3, MiniCPM), even though nothing else
/// about those architectures is graph-decode-incompatible — the ONLY blocker was
/// <see cref="HartsyInference.Core.Backends.IBackend.EmbedGatherDecodeStep"/> having no scale parameter.
/// <see cref="GenericTransformer.EnsureEmbedResidentForGraphDecode"/> now pre-applies the scale once to a
/// dedicated GPU copy instead, so this verifies (a) the model is now graph-decode-eligible and (b) the returned
/// table's actual GPU-resident values are exactly <c>embed * EmbeddingScale</c>, while the ordinary <c>_embed</c>
/// table (and <see cref="GenericTransformer.EmbedLookup"/>'s own host path) stays unscaled.</summary>
[Collection("CudaSerial")]
public sealed unsafe class GraphDecodeEmbeddingScaleTests
{
    private static uint _rng = 0xE4BED5u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    [Fact]
    public void GraniteLikeEmbeddingScale_IsGraphDecodeEligible_AndScalesTableCorrectly()
    {
        if (!CudaContext.IsAvailable()) { Console.Error.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int vocab = 12, hidden = 8;
        const float embeddingScale = 2.5f;
        TransformerConfig cfg = new()
        {
            HiddenSize = hidden, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, HeadDim = 4,
            IntermediateSize = 16, VocabSize = vocab, MaxPositionEmbeddings = 32, AttentionBias = false, QkNorm = false,
            TieWordEmbeddings = true, EmbeddingScale = embeddingScale,
        };
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(vocab, hidden),
            ["model.norm.weight"] = Ones(hidden),
        };
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(hidden);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(hidden);
            w[$"{p}.self_attn.q_proj.weight"] = F2(cfg.QDim, hidden);
            w[$"{p}.self_attn.k_proj.weight"] = F2(cfg.KvDim, hidden);
            w[$"{p}.self_attn.v_proj.weight"] = F2(cfg.KvDim, hidden);
            w[$"{p}.self_attn.o_proj.weight"] = F2(hidden, cfg.QDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(cfg.IntermediateSize, hidden);
            w[$"{p}.mlp.up_proj.weight"] = F2(cfg.IntermediateSize, hidden);
            w[$"{p}.mlp.down_proj.weight"] = F2(hidden, cfg.IntermediateSize);
        }
        float[] unscaledEmbed = new float[vocab * hidden];
        new Span<float>((float*)w["model.embed_tokens.weight"].DataPointer, unscaledEmbed.Length).CopyTo(unscaledEmbed);

        using CudaBackend backend = new(0, ptxDir);
        try
        {
            using GenericTransformer model = new(cfg);
            model.LoadWeights(w, "model");
            Assert.True(model.SupportsGraphDecode(backend), "EmbeddingScale != 1.0 must no longer disqualify graph decode");

            Tensor scaled = model.EnsureEmbedResidentForGraphDecode(backend);
            backend.Sync();
            float* sp = (float*)scaled.DataPointer;
            for (int i = 0; i < unscaledEmbed.Length; i++)
                Assert.True(MathF.Abs(sp[i] - unscaledEmbed[i] * embeddingScale) < 1e-4f,
                    $"i={i}: expected {unscaledEmbed[i] * embeddingScale}, got {sp[i]}");

            // EmbedLookup's own host path must stay unaffected (unscaled table, per-call host scale) — this is
            // the normal prefill path and must not double-apply the scale now baked into the graph-decode copy.
            using Tensor lookupOut = new(new TensorShape(1, 2, hidden), DType.F32);
            model.EmbedLookup(lookupOut, [0, 1]);
            float* lp = (float*)lookupOut.DataPointer;
            for (int t = 0; t < 2; t++)
                for (int d = 0; d < hidden; d++)
                    Assert.True(MathF.Abs(lp[t * hidden + d] - unscaledEmbed[t * hidden + d] * embeddingScale) < 1e-4f,
                        $"EmbedLookup t={t} d={d}: scale mismatch");
        }
        finally
        {
            foreach (Tensor t in w.Values) t.Dispose();
            try { backend.Dispose(); } catch (Exception ex) { Console.Error.WriteLine($"[teardown-ignored] {ex.GetType().Name}: {ex.Message}"); }
        }
    }
}
