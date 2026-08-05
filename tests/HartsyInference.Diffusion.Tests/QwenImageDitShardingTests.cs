using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT-sharding parity for Qwen-Image: <see cref="QwenImageTransformer.ForwardSharded"/> — a block-range
/// split of the 60-block dual-stream loop across two <see cref="CudaBackend"/>s — vs
/// <see cref="QwenImageTransformer.Forward"/> on a tiny synthetic config. Two bars, both load-bearing:
/// same-ordinal split is BIT-EXACT (identical kernels, so any drift is a split-logic bug), while the cross-device
/// split gets a tight tolerance — F32 cuBLAS GEMM reduction order legitimately differs between GPU
/// architectures (SM 8.9 vs 8.6 here), a real cross-hardware numeric difference, not a wiring bug (the Krea2
/// bit-exact cross-device precedent only held because its tiny GEMMs picked the same kernel on both cards).
/// Covers both the plain t2i shape and the Edit shape (trailing reference tokens + timestep-zero modulation) —
/// the edit path adds tembZero as a fourth peer-copied tensor at the split boundary.</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class QwenImageDitShardingTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageDitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // HeadDim must stay 128: QwenImageTransformer hardcodes the rope's [16, 56, 56] axis split (sums to 128).
    // 4 blocks so the split point (2) has real work on both sides.
    private static QwenImageConfig TinyConfig => new()
    {
        HiddenSize = 128,
        NumHeads = 1,
        HeadDim = 128,
        Depth = 4,
        MlpRatio = 2.0f,
        ContextDim = 32,
        InChannels = 16,
        PatchSize = 2,
    };

    [Fact]
    public void ForwardSharded_SameDevice_MatchesUnsharded_BitParity()
    {
        RunParityCase(editRefs: false, secondOrdinal: 0, exact: true);
    }

    [Fact]
    public void ForwardSharded_SameDevice_EditReferenceTokens_MatchesUnsharded_BitParity()
    {
        RunParityCase(editRefs: true, secondOrdinal: 0, exact: true);
    }

    [Fact]
    public void ForwardSharded_CrossDevice_MatchesUnsharded_WithinTolerance()
    {
        if (CudaContext.IsAvailable() && CudaContext.GetDeviceCount() < 2)
        {
            _output.WriteLine("SKIPPED: needs 2 physical GPUs for the cross-device case.");
            return;
        }
        RunParityCase(editRefs: true, secondOrdinal: 1, exact: false);
    }

    private void RunParityCase(bool editRefs, int secondOrdinal, bool exact)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        _output.WriteLine($"Sharded backend B on ordinal {secondOrdinal}; editRefs={editRefs}; exact={exact}.");

        QwenImageConfig cfg = TinyConfig;
        const int splitBlock = 2;
        const int hPacked = 4, wPacked = 4;
        int patchDim = cfg.PatchSize * cfg.PatchSize * cfg.InChannels;
        int mainSeq = hPacked * wPacked;
        (int H, int W)[] refGrids = editRefs ? [(2, 2)] : [];
        int refSeq = editRefs ? 4 : 0;
        int imgSeq = mainSeq + refSeq;
        const int txtSeq = 5;

        using Tensor packedLatent = QwenImageWeightBuilder.Rand(new TensorShape(1, imgSeq, patchDim), 100, 0.5f);
        using Tensor encoderHidden = QwenImageWeightBuilder.Rand(new TensorShape(1, txtSeq, cfg.ContextDim), 200, 0.2f);

        // ── Reference: one backend, whole DiT resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = QwenImageWeightBuilder.Build(cfg);
        float[] refValues;
        using (CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir()))
        using (QwenImageTransformer refTransformer = new(cfg))
        {
            refTransformer.LoadWeights(wRef);
            refBackend.PreloadWeights(refTransformer.EnumerateWeights());
            using Tensor velocityRef = refTransformer.Forward(
                refBackend, packedLatent, encoderHidden, 0.7f, hPacked, wPacked, refGrids, refTimestepZero: editRefs);
            refValues = ToArray(velocityRef);
            refBackend.FreeWeights(refTransformer.EnumerateWeights());
        }
        QwenImageWeightBuilder.DisposeAll(wRef);

        // ── Sharded: SAME weight VALUES (independent build, same seeds), asymmetric preload — backend A gets
        // shared + blocks[0,split), backend B ONLY blocks[split,Depth). ──
        Dictionary<string, Tensor> wSharded = QwenImageWeightBuilder.Build(cfg);
        float[] shardedValues;
        long peerCopies;
        using (CudaBackend backendA = new(deviceOrdinal: 0, PtxDir()))
        using (CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir()))
        using (QwenImageTransformer shardedTransformer = new(cfg))
        {
            shardedTransformer.LoadWeights(wSharded);
            List<Tensor> aWeights = new(shardedTransformer.EnumerateSharedWeights());
            aWeights.AddRange(shardedTransformer.EnumerateBlockRangeWeights(0, splitBlock));
            backendA.PreloadWeights(aWeights);
            List<Tensor> bWeights = new(shardedTransformer.EnumerateBlockRangeWeights(splitBlock, cfg.Depth));
            backendB.PreloadWeights(bWeights);

            long peerBefore = backendB.GetPeerCopyCount() + backendA.GetPeerCopyCount();
            using Tensor velocitySharded = shardedTransformer.ForwardSharded(
                backendA, backendB, packedLatent, encoderHidden, 0.7f, hPacked, wPacked, splitBlock,
                refGrids, refTimestepZero: editRefs);
            peerCopies = backendB.GetPeerCopyCount() + backendA.GetPeerCopyCount() - peerBefore;
            shardedValues = ToArray(velocitySharded);

            backendA.FreeWeights(aWeights);
            backendB.FreeWeights(bWeights);
        }
        QwenImageWeightBuilder.DisposeAll(wSharded);
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
            // Cross-architecture GEMM order drift is a few ULP per op (measured ~1e-5 absolute here); normalize
            // against the output SCALE, not per-element magnitude — near-zero elements would otherwise inflate
            // legitimate drift into huge per-element ratios. 1e-3 of scale still catches any real handoff
            // corruption (wrong tensor, stale copy, shape mixup all produce O(1)-of-scale errors).
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

/// <summary>Deterministic synthetic weights for a tiny <see cref="QwenImageConfig"/> in the diffusers naming
/// <see cref="QwenImageTransformer.LoadWeights"/> expects (same builder pattern as <c>Krea2WeightBuilder</c>).</summary>
internal static class QwenImageWeightBuilder
{
    public static Dictionary<string, Tensor> Build(QwenImageConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int H = c.HiddenSize;
        int mlp = (int)(c.HiddenSize * c.MlpRatio);
        int patchDim = c.PatchSize * c.PatchSize * c.InChannels;
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d, float center) => w[k] = Const(new TensorShape(d), center, seed++);

        Lin("img_in.weight", H, patchDim); Vec("img_in.bias", H, 0f);
        Vec("txt_norm.weight", c.ContextDim, 1f);
        Lin("txt_in.weight", H, c.ContextDim); Vec("txt_in.bias", H, 0f);
        Lin("time_text_embed.timestep_embedder.linear_1.weight", H, 256); Vec("time_text_embed.timestep_embedder.linear_1.bias", H, 0f);
        Lin("time_text_embed.timestep_embedder.linear_2.weight", H, H); Vec("time_text_embed.timestep_embedder.linear_2.bias", H, 0f);
        Lin("norm_out.linear.weight", 2 * H, H); Vec("norm_out.linear.bias", 2 * H, 0f);
        Lin("proj_out.weight", patchDim, H); Vec("proj_out.bias", patchDim, 0f);

        for (int i = 0; i < c.Depth; i++)
        {
            string p = $"transformer_blocks.{i}";
            Lin($"{p}.img_mod.1.weight", 6 * H, H); Vec($"{p}.img_mod.1.bias", 6 * H, 0f);
            Lin($"{p}.txt_mod.1.weight", 6 * H, H); Vec($"{p}.txt_mod.1.bias", 6 * H, 0f);
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
            Lin($"{p}.img_mlp.net.0.proj.weight", mlp, H); Vec($"{p}.img_mlp.net.0.proj.bias", mlp, 0f);
            Lin($"{p}.img_mlp.net.2.weight", H, mlp); Vec($"{p}.img_mlp.net.2.bias", H, 0f);
            Lin($"{p}.txt_mlp.net.0.proj.weight", mlp, H); Vec($"{p}.txt_mlp.net.0.proj.bias", mlp, 0f);
            Lin($"{p}.txt_mlp.net.2.weight", H, mlp); Vec($"{p}.txt_mlp.net.2.bias", H, 0f);
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
