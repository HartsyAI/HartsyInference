using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT-sharding parity for Chroma: <see cref="ChromaTransformer.ForwardSharded"/> — a flat block-range
/// split of the 19-double + 38-single loop across two <see cref="CudaBackend"/>s — vs
/// <see cref="ChromaTransformer.Forward"/> on a tiny-dim synthetic config that keeps the REAL block counts (the
/// flat space [0,19) doubles / [19,57) singles is what the split points exercise). Two bars, both load-bearing:
/// same-ordinal split is BIT-EXACT (identical kernels, so any drift is a split-logic bug — including the F16
/// activation mirror, since HARTSY_DIT_F16 is default-ON for Chroma), while the cross-device split gets a tight
/// tolerance — GEMM reduction order legitimately differs between GPU architectures (the Qwen-Image precedent).
/// A split inside the doubles crosses TWO streams (img+txt) plus the copied modTable; a split inside the singles
/// crosses ONE concatenated stream. The per-token attention mask is passed so the per-backend SDPA-mask cache
/// (each backend builds/uploads its own host tensor — never peer-copied, never staged twice from one host
/// tensor) is exercised on both sides of the boundary.</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class ChromaDitShardingTests
{
    private readonly ITestOutputHelper _output;
    public ChromaDitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // HeadDim must stay 128: ChromaTransformer.GetRope hardcodes FluxRope's [16, 56, 56] axis split (sums to
    // 128). Real block counts (19 + 38) so the flat split points land where they do on the real model; every
    // other dim is tiny. ApproximatorNumChannels stays 64 (the [t, t0, mod_proj] concat layout is fixed).
    private static ChromaConfig TinyConfig => new()
    {
        Depth = 19,
        DepthSingleBlocks = 38,
        HiddenSize = 128,
        NumHeads = 1,
        HeadDim = 128,
        ApproximatorHiddenDim = 64,
        ApproximatorLayers = 2,
    };

    [Fact]
    public void ForwardSharded_SameDevice_SplitInsideDoubles_MatchesUnsharded_BitParity()
    {
        RunParityCase(splitBlock: 2, secondOrdinal: 0, exact: true);
    }

    [Fact]
    public void ForwardSharded_SameDevice_SplitInsideSingles_MatchesUnsharded_BitParity()
    {
        RunParityCase(splitBlock: 30, secondOrdinal: 0, exact: true);
    }

    [Fact]
    public void ForwardSharded_CrossDevice_MatchesUnsharded_WithinTolerance()
    {
        if (CudaContext.IsAvailable() && CudaContext.GetDeviceCount() < 2)
        {
            _output.WriteLine("SKIPPED: needs 2 physical GPUs for the cross-device case.");
            return;
        }
        RunParityCase(splitBlock: 30, secondOrdinal: 1, exact: false);
    }

    private void RunParityCase(int splitBlock, int secondOrdinal, bool exact)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        _output.WriteLine($"Sharded backend B on ordinal {secondOrdinal}; splitBlock={splitBlock}; exact={exact}.");

        ChromaConfig cfg = TinyConfig;
        int blockCount = cfg.Depth + cfg.DepthSingleBlocks;
        const int hPacked = 4, wPacked = 4;
        int imgSeq = hPacked * wPacked;
        const int txtSeq = 5;
        const int ctxDim = 32;

        using Tensor packedLatent = ChromaShardingWeightBuilder.Rand(new TensorShape(1, imgSeq, 64), 100, 0.5f);
        using Tensor encoderHidden = ChromaShardingWeightBuilder.Rand(new TensorShape(1, txtSeq, ctxDim), 200, 0.2f);
        // Chroma-style keep-mask with a masked padding tail so the per-backend SDPA-mask path is exercised.
        using Tensor attentionMask = ChromaShardingWeightBuilder.KeepMask(txtSeq, keep: 4);

        // ── Reference: one backend, whole DiT resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = ChromaShardingWeightBuilder.Build(cfg, ctxDim);
        float[] refValues;
        using (CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir()))
        using (ChromaTransformer refTransformer = new(cfg))
        {
            refTransformer.LoadWeights(wRef);
            refBackend.PreloadWeights(refTransformer.EnumerateWeights());
            using Tensor velocityRef = refTransformer.Forward(
                refBackend, packedLatent, encoderHidden, 0.7f, txtSeq, hPacked, wPacked, attentionMask);
            refValues = ToArray(velocityRef);
            refBackend.FreeWeights(refTransformer.EnumerateWeights());
        }
        ChromaShardingWeightBuilder.DisposeAll(wRef);

        // ── Sharded: SAME weight VALUES (independent build, same seeds), asymmetric preload — backend A gets
        // shared + blocks[0,split), backend B ONLY blocks[split,BlockCount). ──
        Dictionary<string, Tensor> wSharded = ChromaShardingWeightBuilder.Build(cfg, ctxDim);
        float[] shardedValues;
        long peerCopies;
        using (CudaBackend backendA = new(deviceOrdinal: 0, PtxDir()))
        using (CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir()))
        using (ChromaTransformer shardedTransformer = new(cfg))
        {
            shardedTransformer.LoadWeights(wSharded);
            List<Tensor> aWeights = new(shardedTransformer.EnumerateSharedWeights());
            aWeights.AddRange(shardedTransformer.EnumerateBlockRangeWeights(0, splitBlock));
            backendA.PreloadWeights(aWeights);
            List<Tensor> bWeights = new(shardedTransformer.EnumerateBlockRangeWeights(splitBlock, blockCount));
            backendB.PreloadWeights(bWeights);

            long peerBefore = backendB.GetPeerCopyCount() + backendA.GetPeerCopyCount();
            using Tensor velocitySharded = shardedTransformer.ForwardSharded(
                backendA, backendB, packedLatent, encoderHidden, 0.7f, txtSeq, hPacked, wPacked,
                attentionMask, splitBlock);
            peerCopies = backendB.GetPeerCopyCount() + backendA.GetPeerCopyCount() - peerBefore;
            shardedValues = ToArray(velocitySharded);

            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
        }
        ChromaShardingWeightBuilder.DisposeAll(wSharded);
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

/// <summary>Deterministic synthetic weights for a tiny <see cref="ChromaConfig"/> in the post-conversion diffusers
/// naming <see cref="ChromaTransformer.LoadWeights"/> expects (unfused q/k/v layout — the same builder pattern as
/// <c>QwenImageWeightBuilder</c>).</summary>
internal static class ChromaShardingWeightBuilder
{
    public static Dictionary<string, Tensor> Build(ChromaConfig c, int ctxDim)
    {
        Dictionary<string, Tensor> w = new();
        int H = c.HiddenSize;
        int mlp = H * 4;
        int approxHidden = c.ApproximatorHiddenDim;
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d, float center) => w[k] = Const(new TensorShape(d), center, seed++);

        Lin("x_embedder.weight", H, 64); Vec("x_embedder.bias", H, 0f);
        Lin("context_embedder.weight", H, ctxDim); Vec("context_embedder.bias", H, 0f);
        Lin("proj_out.weight", 64, H); Vec("proj_out.bias", 64, 0f);

        Lin("distilled_guidance_layer.in_proj.weight", approxHidden, c.ApproximatorNumChannels);
        Vec("distilled_guidance_layer.in_proj.bias", approxHidden, 0f);
        for (int i = 0; i < c.ApproximatorLayers; i++)
        {
            Lin($"distilled_guidance_layer.layers.{i}.linear_1.weight", approxHidden, approxHidden);
            Vec($"distilled_guidance_layer.layers.{i}.linear_1.bias", approxHidden, 0f);
            Lin($"distilled_guidance_layer.layers.{i}.linear_2.weight", approxHidden, approxHidden);
            Vec($"distilled_guidance_layer.layers.{i}.linear_2.bias", approxHidden, 0f);
            Vec($"distilled_guidance_layer.norms.{i}.weight", approxHidden, 1f);
        }
        Lin("distilled_guidance_layer.out_proj.weight", H, approxHidden);
        Vec("distilled_guidance_layer.out_proj.bias", H, 0f);

        for (int i = 0; i < c.Depth; i++)
        {
            string p = $"transformer_blocks.{i}";
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

        for (int i = 0; i < c.DepthSingleBlocks; i++)
        {
            string p = $"single_transformer_blocks.{i}";
            Vec($"{p}.attn.norm_q.weight", c.HeadDim, 1f);
            Vec($"{p}.attn.norm_k.weight", c.HeadDim, 1f);
            Lin($"{p}.attn.to_q.weight", H, H); Vec($"{p}.attn.to_q.bias", H, 0f);
            Lin($"{p}.attn.to_k.weight", H, H); Vec($"{p}.attn.to_k.bias", H, 0f);
            Lin($"{p}.attn.to_v.weight", H, H); Vec($"{p}.attn.to_v.bias", H, 0f);
            Lin($"{p}.proj_mlp.weight", mlp, H); Vec($"{p}.proj_mlp.bias", mlp, 0f);
            Lin($"{p}.proj_out.weight", H, H + mlp); Vec($"{p}.proj_out.bias", H, 0f);
        }
        return w;
    }

    /// <summary>Transformer-side [1, seqLen] keep-mask: positions <c>[0, keep)</c> = 1, the padding tail = 0.</summary>
    public static unsafe Tensor KeepMask(int seqLen, int keep)
    {
        Tensor t = new(new TensorShape(1, seqLen), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < seqLen; i++) p[i] = i < keep ? 1.0f : 0.0f;
        return t;
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
