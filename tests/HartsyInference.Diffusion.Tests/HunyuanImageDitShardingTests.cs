using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT-sharding parity for HunyuanImage 2.1: <see cref="HunyuanImageTransformer.ForwardSharded"/> — a
/// flat block-range split of the 20-double + 40-single loop across two <see cref="CudaBackend"/>s — vs
/// <see cref="HunyuanImageTransformer.Forward"/> on a tiny synthetic config. Two bars, both load-bearing:
/// same-ordinal splits are BIT-EXACT (identical kernels, so any drift is a split-logic bug), while the
/// cross-device split gets a tight tolerance — F32 cuBLAS GEMM reduction order legitimately differs between GPU
/// architectures, a real cross-hardware numeric difference, not a wiring bug (the QwenImage precedent). The two
/// same-device cases pin the split inside EACH heterogeneous region (doubles, then singles) and the cross-device
/// case lands it exactly on the double→single boundary, so all three flat-index regimes are covered.</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class HunyuanImageDitShardingTests
{
    private readonly ITestOutputHelper _output;
    public HunyuanImageDitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // HeadDim stays 128: the config's default RopeAxesDim [64, 64] must sum to it. 2 double + 2 single blocks
    // so every split point (1, 2, 3) has real work on both sides; TextEmbedDim2 null skips the ByT5 branch.
    private static HunyuanImageConfig TinyConfig => new()
    {
        HiddenSize = 128,
        NumHeads = 1,
        HeadDim = 128,
        NumDoubleBlocks = 2,
        NumSingleBlocks = 2,
        NumRefinerLayers = 1,
        PatchSize = 1,
        InChannels = 16,
        TextEmbedDim = 32,
        TextEmbedDim2 = null,
        GuidanceEmbed = false,
        MlpRatio = 2.0f,
    };

    [Fact]
    public void ForwardSharded_SameDevice_SplitInsideDoubles_MatchesUnsharded_BitParity()
    {
        RunParityCase(splitBlock: 1, secondOrdinal: 0, exact: true);
    }

    [Fact]
    public void ForwardSharded_SameDevice_SplitInsideSingles_MatchesUnsharded_BitParity()
    {
        RunParityCase(splitBlock: 3, secondOrdinal: 0, exact: true);
    }

    [Fact]
    public void ForwardSharded_CrossDevice_MatchesUnsharded_WithinTolerance()
    {
        if (CudaContext.IsAvailable() && CudaContext.GetDeviceCount() < 2)
        {
            _output.WriteLine("SKIPPED: needs 2 physical GPUs for the cross-device case.");
            return;
        }
        RunParityCase(splitBlock: 2, secondOrdinal: 1, exact: false);
    }

    private void RunParityCase(int splitBlock, int secondOrdinal, bool exact)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        _output.WriteLine($"Sharded backend B on ordinal {secondOrdinal}; splitBlock={splitBlock}; exact={exact}.");

        HunyuanImageConfig cfg = TinyConfig;
        int blockCount = cfg.NumDoubleBlocks + cfg.NumSingleBlocks;
        const int hPacked = 4, wPacked = 4;
        int patchDim = cfg.PatchSize * cfg.PatchSize * cfg.InChannels;
        int imgSeq = hPacked * wPacked;
        const int txtSeq = 5;
        const float timestep = 700.0f;

        using Tensor patchedLatent = HunyuanImageWeightBuilder.Rand(new TensorShape(1, imgSeq, patchDim), 100, 0.5f);
        using Tensor encoderHidden = HunyuanImageWeightBuilder.Rand(new TensorShape(1, txtSeq, cfg.TextEmbedDim), 200, 0.2f);

        // ── Reference: one backend, whole DiT resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = HunyuanImageWeightBuilder.Build(cfg);
        float[] refValues;
        using (CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir()))
        using (HunyuanImageTransformer refTransformer = new(cfg))
        {
            refTransformer.LoadWeights(wRef);
            refBackend.PreloadWeights(refTransformer.EnumerateWeights());
            using Tensor velocityRef = refTransformer.Forward(
                refBackend, patchedLatent, encoderHidden, encoderHidden2: null, timestep, 1.0f, hPacked, wPacked);
            refValues = ToArray(velocityRef);
            refBackend.FreeWeights(refTransformer.EnumerateWeights());
        }
        HunyuanImageWeightBuilder.DisposeAll(wRef);

        // ── Sharded: SAME weight VALUES (independent build, same seeds), asymmetric preload — backend A gets
        // shared + blocks[0,split), backend B ONLY blocks[split,BlockCount). ──
        Dictionary<string, Tensor> wSharded = HunyuanImageWeightBuilder.Build(cfg);
        float[] shardedValues;
        long peerCopies;
        using (CudaBackend backendA = new(deviceOrdinal: 0, PtxDir()))
        using (CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir()))
        using (HunyuanImageTransformer shardedTransformer = new(cfg))
        {
            shardedTransformer.LoadWeights(wSharded);
            List<Tensor> aWeights = new(shardedTransformer.EnumerateSharedWeights());
            aWeights.AddRange(shardedTransformer.EnumerateBlockRangeWeights(0, splitBlock));
            backendA.PreloadWeights(aWeights);
            List<Tensor> bWeights = new(shardedTransformer.EnumerateBlockRangeWeights(splitBlock, blockCount));
            backendB.PreloadWeights(bWeights);

            long peerBefore = backendB.GetPeerCopyCount() + backendA.GetPeerCopyCount();
            using Tensor velocitySharded = shardedTransformer.ForwardSharded(
                backendA, backendB, patchedLatent, encoderHidden, encoderHidden2: null, timestep, 1.0f,
                hPacked, wPacked, splitBlock);
            peerCopies = backendB.GetPeerCopyCount() + backendA.GetPeerCopyCount() - peerBefore;
            shardedValues = ToArray(velocitySharded);

            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
        }
        HunyuanImageWeightBuilder.DisposeAll(wSharded);
        _output.WriteLine($"P2P peer copies observed: {peerCopies} (0 = host-staged fallback path, expected on non-P2P boxes).");

        Assert.Equal(refValues.Length, shardedValues.Length);
        if (exact)
        {
            int mismatches = 0;
            for (int i = 0; i < refValues.Length; i++)
            {
                if (refValues[i] != shardedValues[i])
                {
                    if (mismatches < 5)
                        _output.WriteLine($"mismatch at {i}: ref={refValues[i]} sharded={shardedValues[i]}");
                    mismatches++;
                }
            }
            Assert.True(mismatches == 0, $"{mismatches}/{refValues.Length} values differ between unsharded and same-device sharded forward.");
        }
        else
        {
            // Cross-architecture GEMM order drift is a few ULP per op; normalize against the output SCALE, not
            // per-element magnitude — near-zero elements would otherwise inflate legitimate drift into huge
            // per-element ratios. 1e-3 of scale still catches any real handoff corruption (wrong tensor, stale
            // copy, shape mixup all produce O(1)-of-scale errors).
            double scale = 0;
            foreach (float v in refValues) scale = Math.Max(scale, Math.Abs(v));
            double maxErr = 0;
            for (int i = 0; i < refValues.Length; i++)
                maxErr = Math.Max(maxErr, Math.Abs(refValues[i] - shardedValues[i]));
            double relToScale = maxErr / Math.Max(scale, 1e-6);
            _output.WriteLine($"cross-device: max abs err {maxErr:E3}, output scale {scale:E3}, err/scale {relToScale:E3}");
            Assert.True(relToScale < 1e-3, $"cross-device sharded forward diverged (err/scale {relToScale:E3}).");
        }
    }

    private static unsafe float[] ToArray(Tensor t)
    {
        float[] arr = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) arr[i] = p[i];
        return arr;
    }
}

/// <summary>Deterministic synthetic weights for a tiny <see cref="HunyuanImageConfig"/> in the diffusers naming
/// <see cref="HunyuanImageTransformer.LoadWeights"/> expects (same builder pattern as <c>QwenImageWeightBuilder</c>).
/// Skips the guidance embedder (GuidanceEmbed=false), the ByT5 projection (TextEmbedDim2=null), and the refiner
/// FFN's optional gated <c>ff.net.0.linear</c> branch.</summary>
internal static class HunyuanImageWeightBuilder
{
    public static Dictionary<string, Tensor> Build(HunyuanImageConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int H = c.HiddenSize;
        int mlp = (int)(c.HiddenSize * c.MlpRatio);
        int patchDim = c.PatchSize * c.PatchSize * c.InChannels;
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d, float center) => w[k] = Const(new TensorShape(d), center, seed++);

        Lin("x_embedder.proj.weight", H, patchDim); Vec("x_embedder.proj.bias", H, 0f);
        Lin("time_guidance_embed.timestep_embedder.linear_1.weight", H, 256); Vec("time_guidance_embed.timestep_embedder.linear_1.bias", H, 0f);
        Lin("time_guidance_embed.timestep_embedder.linear_2.weight", H, H); Vec("time_guidance_embed.timestep_embedder.linear_2.bias", H, 0f);
        Lin("norm_out.linear.weight", 2 * H, H); Vec("norm_out.linear.bias", 2 * H, 0f);
        Lin("proj_out.weight", patchDim, H); Vec("proj_out.bias", patchDim, 0f);

        Lin("context_embedder.time_text_embed.timestep_embedder.linear_1.weight", H, 256); Vec("context_embedder.time_text_embed.timestep_embedder.linear_1.bias", H, 0f);
        Lin("context_embedder.time_text_embed.timestep_embedder.linear_2.weight", H, H); Vec("context_embedder.time_text_embed.timestep_embedder.linear_2.bias", H, 0f);
        Lin("context_embedder.time_text_embed.text_embedder.linear_1.weight", H, c.TextEmbedDim); Vec("context_embedder.time_text_embed.text_embedder.linear_1.bias", H, 0f);
        Lin("context_embedder.time_text_embed.text_embedder.linear_2.weight", H, H); Vec("context_embedder.time_text_embed.text_embedder.linear_2.bias", H, 0f);
        Lin("context_embedder.proj_in.weight", H, c.TextEmbedDim); Vec("context_embedder.proj_in.bias", H, 0f);
        for (int i = 0; i < c.NumRefinerLayers; i++)
        {
            string p = $"context_embedder.token_refiner.refiner_blocks.{i}";
            Vec($"{p}.norm1.weight", H, 1f); Vec($"{p}.norm1.bias", H, 0f);
            Vec($"{p}.norm2.weight", H, 1f); Vec($"{p}.norm2.bias", H, 0f);
            Lin($"{p}.attn.to_q.weight", H, H); Vec($"{p}.attn.to_q.bias", H, 0f);
            Lin($"{p}.attn.to_k.weight", H, H); Vec($"{p}.attn.to_k.bias", H, 0f);
            Lin($"{p}.attn.to_v.weight", H, H); Vec($"{p}.attn.to_v.bias", H, 0f);
            Lin($"{p}.attn.to_out.0.weight", H, H); Vec($"{p}.attn.to_out.0.bias", H, 0f);
            Lin($"{p}.ff.net.0.proj.weight", mlp, H); Vec($"{p}.ff.net.0.proj.bias", mlp, 0f);
            Lin($"{p}.ff.net.2.weight", H, mlp); Vec($"{p}.ff.net.2.bias", H, 0f);
            Lin($"{p}.norm_out.linear.weight", 2 * H, H); Vec($"{p}.norm_out.linear.bias", 2 * H, 0f);
        }

        for (int i = 0; i < c.NumDoubleBlocks; i++)
        {
            string p = $"transformer_blocks.{i}";
            Lin($"{p}.norm1.linear.weight", 6 * H, H); Vec($"{p}.norm1.linear.bias", 6 * H, 0f);
            Lin($"{p}.norm1_context.linear.weight", 6 * H, H); Vec($"{p}.norm1_context.linear.bias", 6 * H, 0f);
            Lin($"{p}.attn.to_q.weight", H, H); Vec($"{p}.attn.to_q.bias", H, 0f);
            Lin($"{p}.attn.to_k.weight", H, H); Vec($"{p}.attn.to_k.bias", H, 0f);
            Lin($"{p}.attn.to_v.weight", H, H); Vec($"{p}.attn.to_v.bias", H, 0f);
            Lin($"{p}.attn.to_out.0.weight", H, H); Vec($"{p}.attn.to_out.0.bias", H, 0f);
            Lin($"{p}.attn.add_q_proj.weight", H, H); Vec($"{p}.attn.add_q_proj.bias", H, 0f);
            Lin($"{p}.attn.add_k_proj.weight", H, H); Vec($"{p}.attn.add_k_proj.bias", H, 0f);
            Lin($"{p}.attn.add_v_proj.weight", H, H); Vec($"{p}.attn.add_v_proj.bias", H, 0f);
            Lin($"{p}.attn.to_add_out.weight", H, H); Vec($"{p}.attn.to_add_out.bias", H, 0f);
            Vec($"{p}.attn.norm_q.weight", c.HeadDim, 1f);
            Vec($"{p}.attn.norm_k.weight", c.HeadDim, 1f);
            Vec($"{p}.attn.norm_added_q.weight", c.HeadDim, 1f);
            Vec($"{p}.attn.norm_added_k.weight", c.HeadDim, 1f);
            Lin($"{p}.ff.net.0.proj.weight", mlp, H); Vec($"{p}.ff.net.0.proj.bias", mlp, 0f);
            Lin($"{p}.ff.net.2.weight", H, mlp); Vec($"{p}.ff.net.2.bias", H, 0f);
            Lin($"{p}.ff_context.net.0.proj.weight", mlp, H); Vec($"{p}.ff_context.net.0.proj.bias", mlp, 0f);
            Lin($"{p}.ff_context.net.2.weight", H, mlp); Vec($"{p}.ff_context.net.2.bias", H, 0f);
        }

        for (int i = 0; i < c.NumSingleBlocks; i++)
        {
            string p = $"single_transformer_blocks.{i}";
            Lin($"{p}.norm.linear.weight", 3 * H, H); Vec($"{p}.norm.linear.bias", 3 * H, 0f);
            Lin($"{p}.attn.to_q.weight", H, H); Vec($"{p}.attn.to_q.bias", H, 0f);
            Lin($"{p}.attn.to_k.weight", H, H); Vec($"{p}.attn.to_k.bias", H, 0f);
            Lin($"{p}.attn.to_v.weight", H, H); Vec($"{p}.attn.to_v.bias", H, 0f);
            Vec($"{p}.attn.norm_q.weight", c.HeadDim, 1f);
            Vec($"{p}.attn.norm_k.weight", c.HeadDim, 1f);
            Lin($"{p}.proj_mlp.weight", mlp, H); Vec($"{p}.proj_mlp.bias", mlp, 0f);
            Lin($"{p}.proj_out.weight", H, H + mlp); Vec($"{p}.proj_out.bias", H, 0f);
        }
        return w;
    }

    public static unsafe Tensor Rand(TensorShape s, int seed, float scale)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    public static unsafe Tensor Const(TensorShape s, float center, int seed)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = center + (float)((rng.NextDouble() * 2 - 1) * 0.02);
        return t;
    }

    public static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
