using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Proves <c>MiniMaxMusic3Dit.ForwardCfg</c>'s batch-2 guidance pass computes what the two separate
/// forwards computed. Every failure mode of a batched rewrite is silent at the API boundary: a rope table that is
/// not repeated per batch element, a row offset that reads the wrong branch back out, or a permute that mixes the
/// two branches all return correctly-shaped finite velocities and only show up as subtly wrong music. The CUDA arm
/// exists because this model has twice shipped a device-residency bug that passed on <c>CpuBackend</c>.</summary>
public sealed unsafe class MiniMaxMusic3DitCfgBatchTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxMusic3DitCfgBatchTests(ITestOutputHelper output) => _output = output;

    /// <summary>Small enough to run in the Unit lane, but keeps every geometric feature that could break: more than
    /// one block, partial rotary, a gated feed-forward, and a condition width unequal to the latent width.</summary>
    private static MiniMaxMusic3DitConfig TinyConfig => new()
    {
        InChannels = 4,
        ConditionDim = 8,
        NumLayers = 2,
        NumAttentionHeads = 2,
        AttentionHeadDim = 8,
        FfInnerDim = 32,
        RotaryDim = 4,
        FourierEmbeddingDim = 8,
    };

    [Fact]
    public void ForwardCfg_MatchesTwoSeparateForwards_OnCpu()
    {
        using CpuBackend backend = new CpuBackend();
        AssertMatchesSeparateForwards(backend, TinyConfig, length: 11, tolerance: 1e-5f);
    }

    /// <summary>The 3060 is the working card here, so the ordinal is configurable — the parity gates that hardcode
    /// ordinal 0 need the 4090's VRAM, this does not.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ForwardCfg_MatchesTwoSeparateForwards_OnCuda()
    {
        if (!TryCuda(out CudaBackend? backend))
        {
            return;
        }
        using (backend)
        {
            AssertMatchesSeparateForwards(backend!, TinyConfig, length: 11, tolerance: 2e-3f);
        }
    }

    /// <summary>The checkpoint's real width, head count and window length over one random-weight block, anchored on
    /// <see cref="CpuBackend"/>. The tiny config above cannot answer whether the batched pass is sound at the shapes
    /// that ship: at <c>m = 11</c> the CUDA Linear never reaches its int8 path and a two-head 8-wide attention takes
    /// a different SDPA branch than 32 heads at 690 tokens. Batch 2 doubles <c>m</c>, which is exactly the condition
    /// under which cuBLAS may pick a different algorithm — so a raw batched-vs-single delta cannot distinguish a
    /// layout bug from re-association. The CPU anchor can: if batching moved the answer LESS than running on CUDA at
    /// all does, the difference is rounding, not wrong math.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ForwardCfg_DriftIsSmallerThanTheBackendsOwn_AtCheckpointGeometry()
    {
        if (!TryCuda(out CudaBackend? cuda))
        {
            return;
        }
        const int length = 689;
        const float timestep = 0.37f;
        MiniMaxMusic3DitConfig config = MiniMaxMusic3DitConfig.Default with { NumLayers = 1 };
        Dictionary<string, Tensor> weights = BuildWeights(config, seed: 20260814);
        using CpuBackend cpu = new CpuBackend();
        using (cuda)
        using (MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit(config))
        {
            try
            {
                dit.LoadWeights(weights);
                using Tensor latents = Random(new TensorShape(1, config.InChannels, length), 7, 0.8f);
                using Tensor condition = Random(new TensorShape(1, length, config.ConditionDim), 11, 0.5f);
                using Tensor reference = dit.Forward(cpu, latents, timestep, condition);
                using Tensor single = dit.Forward(cuda!, latents, timestep, condition);
                (Tensor batched, Tensor unconditional) = dit.ForwardCfg(cuda!, latents, timestep, condition);
                using (batched)
                using (unconditional)
                {
                    double backendDrift = RelativeDelta(reference, single);
                    double batchDrift = RelativeDelta(single, batched);
                    double batchVsReference = RelativeDelta(reference, batched);
                    _output.WriteLine($"relative maxAbs — cuda-vs-cpu {backendDrift:0.000e+0}, "
                        + $"batched-vs-single {batchDrift:0.000e+0}, batched-vs-cpu {batchVsReference:0.000e+0}");
                    Assert.True(batchDrift < backendDrift,
                        $"batching moved the result more ({batchDrift}) than running on CUDA at all does ({backendDrift}) — "
                        + "that is a wrong-math signature, not re-association.");
                    Assert.True(batchDrift < 5e-3, $"batched branch drifted by {batchDrift} relative.");
                }
            }
            finally
            {
                foreach (Tensor weight in weights.Values)
                {
                    weight.Dispose();
                }
            }
        }
    }

    private bool TryCuda(out CudaBackend? backend)
    {
        backend = null;
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return false;
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return false;
        }
        int ordinal = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_GPU"), out int parsed) ? parsed : 0;
        backend = new CudaBackend(deviceOrdinal: ordinal, ptxDir: ptxDir);
        return true;
    }

    private void AssertMatchesSeparateForwards(IBackend backend, MiniMaxMusic3DitConfig config, int length, float tolerance)
    {
        const float timestep = 0.37f;
        Dictionary<string, Tensor> weights = BuildWeights(config, seed: 20260814);
        using MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit(config);
        try
        {
            dit.LoadWeights(weights);
            using Tensor latents = Random(new TensorShape(1, config.InChannels, length), 7, 0.8f);
            using Tensor condition = Random(new TensorShape(1, length, config.ConditionDim), 11, 0.5f);
            using Tensor zeros = new Tensor(condition.Shape, DType.F32);

            using Tensor conditionalReference = dit.Forward(backend, latents, timestep, condition);
            using Tensor unconditionalReference = dit.Forward(backend, latents, timestep, zeros);
            (Tensor conditional, Tensor unconditional) = dit.ForwardCfg(backend, latents, timestep, condition);
            using (conditional)
            using (unconditional)
            {
                Assert.Equal(conditionalReference.Shape, conditional.Shape);
                Assert.Equal(unconditionalReference.Shape, unconditional.Shape);
                double condError = MaxAbsDelta(conditionalReference, conditional, out double condScale);
                double uncondError = MaxAbsDelta(unconditionalReference, unconditional, out double uncondScale);
                _output.WriteLine($"cond maxAbs {condError:0.000e+0} (peak {condScale:0.000}), "
                    + $"uncond maxAbs {uncondError:0.000e+0} (peak {uncondScale:0.000})");

                // Both branches must be checked: a batching bug that fed BOTH rows the conditional tokens leaves the
                // conditional branch exact and only shows up here.
                Assert.True(condError < tolerance, $"conditional branch diverged by {condError}.");
                Assert.True(uncondError < tolerance, $"unconditional branch diverged by {uncondError}.");

                // The two branches must not be the same tensor's contents — zero conditioning has to matter, or the
                // test would pass on an implementation that ignored the condition entirely.
                Assert.True(MaxAbsDelta(conditional, unconditional, out _) > 1e-3, "guidance branches are identical.");
            }
        }
        finally
        {
            foreach (Tensor weight in weights.Values)
            {
                weight.Dispose();
            }
        }
    }

    /// <summary>Max abs delta as a fraction of the first tensor's peak magnitude — random weights through an
    /// unnormalized stack leave activations far from O(1), so an absolute threshold means nothing here.</summary>
    private static double RelativeDelta(Tensor a, Tensor b)
    {
        double worst = MaxAbsDelta(a, b, out double peak);
        return peak > 0 ? worst / peak : worst;
    }

    private static double MaxAbsDelta(Tensor a, Tensor b, out double peak)
    {
        ReadOnlySpan<float> left = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> right = b.AsReadOnlySpan<float>();
        double worst = 0;
        double magnitude = 0;
        for (int i = 0; i < left.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(left[i] - right[i]));
            magnitude = Math.Max(magnitude, Math.Abs(left[i]));
        }
        peak = magnitude;
        return worst;
    }

    private static Dictionary<string, Tensor> BuildWeights(MiniMaxMusic3DitConfig config, int seed)
    {
        int inner = config.InnerDim;
        int concat = config.ConcatChannels;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["time_proj.weight"] = Random(new TensorShape(1, config.FourierEmbeddingDim / 2), seed + 1, 1f),
            ["time_embed.linear_1.weight"] = Random(new TensorShape(inner, config.FourierEmbeddingDim), seed + 2, 0.2f),
            ["time_embed.linear_1.bias"] = Random(new TensorShape(inner), seed + 3, 0.1f),
            ["time_embed.linear_2.weight"] = Random(new TensorShape(inner, inner), seed + 4, 0.2f),
            ["time_embed.linear_2.bias"] = Random(new TensorShape(inner), seed + 5, 0.1f),
            ["preprocess_conv.weight"] = Random(new TensorShape(concat, concat, 1), seed + 6, 0.15f),
            ["proj_in.weight"] = Random(new TensorShape(inner, concat), seed + 7, 0.2f),
            ["proj_out.weight"] = Random(new TensorShape(config.InChannels, inner), seed + 8, 0.2f),
            ["postprocess_conv.weight"] = Random(new TensorShape(config.InChannels, config.InChannels, 1), seed + 9, 0.15f),
        };
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            string prefix = $"transformer_blocks.{layer}";
            int offset = seed + 100 + (layer * 20);
            weights[$"{prefix}.norm1.weight"] = Constant(new TensorShape(inner), 1f);
            weights[$"{prefix}.norm1.bias"] = Random(new TensorShape(inner), offset + 1, 0.05f);
            weights[$"{prefix}.norm2.weight"] = Constant(new TensorShape(inner), 1f);
            weights[$"{prefix}.norm2.bias"] = Random(new TensorShape(inner), offset + 2, 0.05f);
            weights[$"{prefix}.attn.to_q.weight"] = Random(new TensorShape(inner, inner), offset + 3, 0.2f);
            weights[$"{prefix}.attn.to_k.weight"] = Random(new TensorShape(inner, inner), offset + 4, 0.2f);
            weights[$"{prefix}.attn.to_v.weight"] = Random(new TensorShape(inner, inner), offset + 5, 0.2f);
            weights[$"{prefix}.attn.to_out.0.weight"] = Random(new TensorShape(inner, inner), offset + 6, 0.2f);
            weights[$"{prefix}.ff_in.weight"] = Random(new TensorShape(2 * config.FfInnerDim, inner), offset + 7, 0.2f);
            weights[$"{prefix}.ff_in.bias"] = Random(new TensorShape(2 * config.FfInnerDim), offset + 8, 0.05f);
            weights[$"{prefix}.ff_out.weight"] = Random(new TensorShape(inner, config.FfInnerDim), offset + 9, 0.2f);
            weights[$"{prefix}.ff_out.bias"] = Random(new TensorShape(inner), offset + 10, 0.05f);
        }
        return weights;
    }

    private static Tensor Random(TensorShape shape, int seed, float scale)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        Span<float> values = new Span<float>((float*)tensor.DataPointer, (int)shape.ElementCount);
        Random random = new Random(seed);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)((random.NextDouble() * 2.0) - 1.0) * scale;
        }
        return tensor;
    }

    private static Tensor Constant(TensorShape shape, float value)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        new Span<float>((float*)tensor.DataPointer, (int)shape.ElementCount).Fill(value);
        return tensor;
    }
}
