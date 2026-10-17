using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runs FLUX.3 end to end on a one-head synthetic config: per-mode stacks, span-modulated joint stack, final
/// layer. Gates wiring, shapes and the precompute/flow-map contracts, not numerics; parity lives in
/// <c>Parity/Flux3ParityTests</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed unsafe class Flux3TransformerSmokeTests
{
    private const int Hidden = 128;
    private const int Mlp = 64;
    private const int Context = 32;
    private int _seed = 7;

    [Fact]
    public void Forward_PrecomputedStatesMatchInlineStates()
    {
        IBackend backend = new CpuBackend();
        (Flux3Transformer transformer, Dictionary<string, Tensor> weights) = Build(flowMap: false);
        using Tensor text = Rand(1, 5, Context);
        using Tensor latent = Rand(1, 6, 128);
        Flux3ConditionInput input = Flux3ConditionInput.TextOnly(text);
        try
        {
            using Flux3ConditionStates states = transformer.PrecomputeConditioning(backend, input);
            using Tensor cached = transformer.Forward(backend, latent, 2, 3, 0.7f, input, states);
            using Tensor inline = transformer.Forward(backend, latent, 2, 3, 0.7f, input, null);
            Assert.Equal(new TensorShape(1, 6, 128), cached.Shape);
            AssertFinite(cached);
            Assert.Equal(cached.AsReadOnlySpan<float>().ToArray(), inline.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            Dispose(transformer, weights);
        }
    }

    [Fact]
    public void Forward_ReferencesConditionTheImageRows()
    {
        IBackend backend = new CpuBackend();
        (Flux3Transformer transformer, Dictionary<string, Tensor> weights) = Build(flowMap: false);
        using Tensor text = Rand(1, 4, Context);
        using Tensor latent = Rand(1, 4, 128);
        using Tensor references = Rand(1, 4 + 2, 128);
        try
        {
            using Tensor plain = transformer.Forward(backend, latent, 2, 2, 0.5f, Flux3ConditionInput.TextOnly(text), null);
            Flux3ConditionInput edit = new Flux3ConditionInput(text, references, [(2, 2), (1, 2)]);
            using Tensor edited = transformer.Forward(backend, latent, 2, 2, 0.5f, edit, null);
            AssertFinite(edited);
            Assert.NotEqual(plain.AsReadOnlySpan<float>().ToArray(), edited.AsReadOnlySpan<float>().ToArray());
            Flux3ConditionInput mismatched = new Flux3ConditionInput(text, references, [(2, 2)]);
            Assert.Throws<ArgumentException>(() => transformer.Forward(backend, latent, 2, 2, 0.5f, mismatched, null));
        }
        finally
        {
            Dispose(transformer, weights);
        }
    }

    [Fact]
    public void FlowMap_NeedsTheTargetSigma_AndDependsOnIt()
    {
        IBackend backend = new CpuBackend();
        (Flux3Transformer transformer, Dictionary<string, Tensor> weights) = Build(flowMap: true);
        using Tensor text = Rand(1, 3, Context);
        using Tensor latent = Rand(1, 4, 128);
        Flux3ConditionInput input = Flux3ConditionInput.TextOnly(text);
        try
        {
            Assert.True(transformer.Config.FlowMap);
            Assert.Throws<ArgumentException>(() => transformer.Forward(backend, latent, 2, 2, 0.8f, input, null));
            using Tensor near = transformer.Forward(backend, latent, 2, 2, 0.8f, input, null, targetSigma: 0.6f);
            using Tensor far = transformer.Forward(backend, latent, 2, 2, 0.8f, input, null, targetSigma: 0f);
            AssertFinite(near);
            Assert.NotEqual(near.AsReadOnlySpan<float>().ToArray(), far.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            Dispose(transformer, weights);
        }
    }

    private (Flux3Transformer Transformer, Dictionary<string, Tensor> Weights) Build(bool flowMap)
    {
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["emb_in.image.weight"] = Rand(Hidden, 128),
            ["emb_in.image_cond.weight"] = Rand(Hidden, 128),
            ["txt_in.weight"] = Rand(Hidden, Context),
            ["time_in.in_layer.weight"] = Rand(Hidden, 256),
            ["time_in.out_layer.weight"] = Rand(Hidden, Hidden),
            ["final_layer.image.adaLN_modulation.1.weight"] = Rand(2 * Hidden, Hidden),
            ["final_layer.image.linear.weight"] = Rand(128, Hidden),
        };
        if (flowMap)
        {
            w["ref_time_in.in_layer.weight"] = Rand(Hidden, 256);
            w["ref_time_in.out_layer.weight"] = Rand(Hidden, Hidden);
        }
        foreach (string mode in new[] { "image", "image_cond", "txt" })
        {
            w[$"early_stream_modulations.{mode}.lin.weight"] = Rand(3 * Hidden, Hidden);
            w[$"single_stream_modulations.{mode}.lin.weight"] = Rand(3 * Hidden, Hidden);
        }
        foreach (string prefix in new[] { "content_mode_blocks.image.0", "content_mode_blocks.image_cond.0", "txt_mode_blocks.0", "single_blocks.0", "single_blocks.1" })
        {
            w[$"{prefix}.q_proj.weight"] = Rand(Hidden, Hidden);
            w[$"{prefix}.k_proj.weight"] = Rand(Hidden, Hidden);
            w[$"{prefix}.v_proj.weight"] = Rand(Hidden, Hidden);
            w[$"{prefix}.mlp_in.weight"] = Rand(2 * Mlp, Hidden);
            w[$"{prefix}.attn_out.weight"] = Rand(Hidden, Hidden);
            w[$"{prefix}.mlp_out.weight"] = Rand(Hidden, Mlp);
            w[$"{prefix}.norm.query_norm.weight"] = Ones(Hidden);
            w[$"{prefix}.norm.key_norm.weight"] = Ones(Hidden);
        }
        Flux3Config config = Flux3Config.FromWeights(w) with { ContextInDim = Context };
        Flux3Transformer transformer = new Flux3Transformer(config);
        transformer.LoadWeights(w);
        return (transformer, w);
    }

    [Fact]
    public void DefaultF16Knob_OnCpu_RunsTheF32Stream()
    {
        // numerics.ditF16 defaults on, but the CPU backend has no F16 activation kernels, so the stream stays F32 there.
        Assert.Equal(DType.F16, DitDtype.Act);
        IBackend backend = new CpuBackend();
        (Flux3Transformer transformer, Dictionary<string, Tensor> weights) = Build(flowMap: false);
        using Tensor text = Rand(1, 5, Context);
        using Tensor latent = Rand(1, 6, 128);
        Flux3ConditionInput input = Flux3ConditionInput.TextOnly(text);
        try
        {
            using Tensor defaults = transformer.Forward(backend, latent, 2, 3, 0.7f, input, null);
            Tensor pinned;
            using (KnobProfile.Create("flux3-f32-check").With(EngineKnobs.DitF16, false).Push())
                pinned = transformer.Forward(backend, latent, 2, 3, 0.7f, input, null);
            using (pinned)
            {
                Assert.Equal(pinned.AsReadOnlySpan<float>().ToArray(), defaults.AsReadOnlySpan<float>().ToArray());
            }
        }
        finally
        {
            Dispose(transformer, weights);
        }
    }

    private Tensor Rand(params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            _seed = _seed * 1103515245 + 12345;
            p[i] = ((_seed >> 16) & 0x7fff) / 32768f * 0.2f - 0.1f;
        }
        return t;
    }

    private static Tensor Ones(int length)
    {
        Tensor t = new Tensor(new TensorShape(length), DType.F32);
        t.AsSpan<float>().Fill(1f);
        return t;
    }

    private static void AssertFinite(Tensor t)
    {
        foreach (float v in t.AsReadOnlySpan<float>()) Assert.True(float.IsFinite(v));
    }

    private static void Dispose(Flux3Transformer transformer, Dictionary<string, Tensor> weights)
    {
        transformer.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
    }
}
