using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT sharding (ROADMAP.md §1): verifies <see cref="Sd3Transformer.ForwardSharded"/> — a block-range
/// split of the <c>JointBlock</c> loop across two <see cref="CudaBackend"/>s — matches
/// <see cref="Sd3Transformer.Forward"/> (unsharded, one backend, whole MMDiT resident) bit-for-bit on a synthetic
/// tiny config. Same-GPU split (two distinct <see cref="CudaBackend"/> instances on the same ordinal — still
/// exercises per-backend weight/activation caching, just not a second physical card): with every op deterministic
/// F32, the two paths produce IDENTICAL floats. Cross-architecture pairs (measured: 4090 SM 8.9 + 3060 SM 8.6)
/// do NOT reproduce bit-exact here — the JointBlock's SDPA/GEMM sequence picks a different cuDNN/cuBLAS kernel
/// per SM generation, diverging in the last 1-2 ULPs and compounding across 4 blocks; that is a hardware/library
/// rounding artifact, not a sharding defect (same-GPU passing bit-exact proves the split logic itself is
/// correct). Real cross-device verification uses SSIM, not bit-parity — see <c>Sd3DitShardingEngineTests</c>.
/// The tiny config's last block (index 3) is <c>context_pre_only</c> — exactly like the real 24/38-block
/// configs — so this also exercises the context-stream identity-passthrough hand-off (see
/// <see cref="Sd3Transformer"/>'s <c>ForwardBlocksRange</c> doc).</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class Sd3DitShardingTests
{
    private readonly ITestOutputHelper _output;
    public Sd3DitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // 4 blocks (vs. a 1-2 block smoke test) so the split point (2) has real work on both sides, and the last
    // block (index 3) exercises context_pre_only inside backend B's range.
    private static Sd3Config TinyConfig => new()
    {
        Depth = 4,
        HiddenSize = 32,
        NumHeads = 4,
        PatchSize = 2,
        InChannels = 4,
        JointAttentionDim = 16,
        PooledProjectionDim = 12,
        UseQkNorm = true,
        DualAttentionLayers = null,
    };

    [Fact]
    public void ForwardSharded_MatchesUnsharded_BitParity()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // Same-GPU by design (see class doc): a second CudaBackend instance on the SAME ordinal, not a second
        // physical card — cross-architecture pairs introduce real last-ULP rounding differences unrelated to
        // the split logic (verified separately; see Sd3DitShardingEngineTests for the real cross-device path).
        const int secondOrdinal = 0;
        _output.WriteLine($"Devices: {CudaContext.GetDeviceCount()}; sharded backend B on ordinal {secondOrdinal} (same-GPU split).");

        Sd3Config cfg = TinyConfig;
        const int splitBlock = 2;

        // ── Reference: one backend, whole MMDiT resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = Sd3WeightBuilder.Build(cfg);
        using CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir());
        using Sd3Transformer refTransformer = new(cfg);
        refTransformer.LoadWeights(wRef);
        refBackend.PreloadWeights(refTransformer.EnumerateWeights());

        int h = 8, wd = 8;
        using Tensor latent = Sd3WeightBuilder.Rand(new TensorShape(1, cfg.InChannels, h, wd), 100, 0.5f);
        const int ctxSeq = 6;
        using Tensor context = Sd3WeightBuilder.Rand(new TensorShape(1, ctxSeq, cfg.HiddenSize), 200, 0.2f);
        using Tensor pooled = Sd3WeightBuilder.Rand(new TensorShape(1, cfg.HiddenSize), 300, 0.2f);

        using Tensor velocityRef = refTransformer.Forward(refBackend, latent, 500.0f, context, pooled);
        float[] refValues = ToArray(velocityRef);
        refBackend.FreeWeights(refTransformer.EnumerateWeights());
        Sd3WeightBuilder.DisposeAll(wRef);

        // ── Sharded: SAME weight VALUES (independent load, same seeds), split across two backends. Backend A
        // gets shared weights + blocks[0,split); backend B gets ONLY blocks[split,Depth) — the asymmetric
        // preload that makes this VRAM-pooling rather than replication. ──
        Dictionary<string, Tensor> wSharded = Sd3WeightBuilder.Build(cfg);
        using CudaBackend backendA = new(deviceOrdinal: 0, PtxDir());
        using CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir());
        using Sd3Transformer shardedTransformer = new(cfg);
        shardedTransformer.LoadWeights(wSharded);

        List<Tensor> aWeights = new(shardedTransformer.EnumerateSharedWeights());
        aWeights.AddRange(shardedTransformer.EnumerateBlockRangeWeights(0, splitBlock));
        backendA.PreloadWeights(aWeights);
        List<Tensor> bWeights = new(shardedTransformer.EnumerateBlockRangeWeights(splitBlock, cfg.Depth));
        backendB.PreloadWeights(bWeights);

        using Tensor velocitySharded = shardedTransformer.ForwardSharded(
            backendA, backendB, latent, 500.0f, context, pooled, splitBlock);
        float[] shardedValues = ToArray(velocitySharded);

        backendA.FreeWeights(aWeights);
        backendB.FreeWeights(bWeights);
        Sd3WeightBuilder.DisposeAll(wSharded);

        Assert.Equal(refValues.Length, shardedValues.Length);
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
        Assert.True(mismatches == 0, $"{mismatches}/{refValues.Length} values differ between unsharded and sharded forward.");
    }

    private static unsafe float[] ToArray(Tensor t)
    {
        float[] arr = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) arr[i] = p[i];
        return arr;
    }
}

/// <summary>Tiny synthetic SD3 weight builder shared by the sharding tests — mirrors <c>Krea2WeightBuilder</c>'s shape.</summary>
internal static class Sd3WeightBuilder
{
    public static Dictionary<string, Tensor> Build(Sd3Config c)
    {
        Dictionary<string, Tensor> w = new();
        int hidden = c.HiddenSize;
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d) => w[k] = Rand(new TensorShape(d), seed++, 0.02f);

        w["pos_embed.proj.weight"] = Rand(new TensorShape(hidden, c.InChannels, c.PatchSize, c.PatchSize), seed++, 0.06f);
        Vec("pos_embed.proj.bias", hidden);

        Lin("time_text_embed.timestep_embedder.linear_1.weight", hidden, 256);
        Vec("time_text_embed.timestep_embedder.linear_1.bias", hidden);
        Lin("time_text_embed.timestep_embedder.linear_2.weight", hidden, hidden);
        Vec("time_text_embed.timestep_embedder.linear_2.bias", hidden);

        Lin("time_text_embed.text_embedder.linear_1.weight", hidden, c.PooledProjectionDim);
        Vec("time_text_embed.text_embedder.linear_1.bias", hidden);
        Lin("time_text_embed.text_embedder.linear_2.weight", hidden, hidden);
        Vec("time_text_embed.text_embedder.linear_2.bias", hidden);

        Lin("context_embedder.weight", hidden, c.JointAttentionDim);
        Vec("context_embedder.bias", hidden);

        Lin("norm_out.linear.weight", 2 * hidden, hidden);
        Vec("norm_out.linear.bias", 2 * hidden);
        Lin("proj_out.weight", c.PatchSize * c.PatchSize * c.InChannels, hidden);
        Vec("proj_out.bias", c.PatchSize * c.PatchSize * c.InChannels);

        int headDim = hidden / c.NumHeads;
        int ffDim = hidden * 4;
        HashSet<int> dualLayers = c.DualAttentionLayers is null ? new HashSet<int>() : new HashSet<int>(c.DualAttentionLayers);
        for (int i = 0; i < c.Depth; i++)
        {
            bool isPreOnly = i == c.Depth - 1;
            bool useDual = dualLayers.Contains(i);
            string p = $"transformer_blocks.{i}";

            int imgModParams = useDual ? 9 : 6;
            Lin($"{p}.norm1.linear.weight", imgModParams * hidden, hidden);
            Vec($"{p}.norm1.linear.bias", imgModParams * hidden);
            int ctxModParams = isPreOnly ? 2 : 6;
            Lin($"{p}.norm1_context.linear.weight", ctxModParams * hidden, hidden);
            Vec($"{p}.norm1_context.linear.bias", ctxModParams * hidden);

            Lin($"{p}.attn.to_q.weight", hidden, hidden); Vec($"{p}.attn.to_q.bias", hidden);
            Lin($"{p}.attn.to_k.weight", hidden, hidden); Vec($"{p}.attn.to_k.bias", hidden);
            Lin($"{p}.attn.to_v.weight", hidden, hidden); Vec($"{p}.attn.to_v.bias", hidden);
            Lin($"{p}.attn.to_out.0.weight", hidden, hidden); Vec($"{p}.attn.to_out.0.bias", hidden);

            Lin($"{p}.attn.add_q_proj.weight", hidden, hidden); Vec($"{p}.attn.add_q_proj.bias", hidden);
            Lin($"{p}.attn.add_k_proj.weight", hidden, hidden); Vec($"{p}.attn.add_k_proj.bias", hidden);
            Lin($"{p}.attn.add_v_proj.weight", hidden, hidden); Vec($"{p}.attn.add_v_proj.bias", hidden);
            if (!isPreOnly)
            {
                Lin($"{p}.attn.to_add_out.weight", hidden, hidden); Vec($"{p}.attn.to_add_out.bias", hidden);
            }

            if (c.UseQkNorm)
            {
                Vec($"{p}.attn.norm_q.weight", headDim);
                Vec($"{p}.attn.norm_k.weight", headDim);
                Vec($"{p}.attn.norm_added_q.weight", headDim);
                Vec($"{p}.attn.norm_added_k.weight", headDim);
            }

            if (useDual)
            {
                Lin($"{p}.attn2.to_q.weight", hidden, hidden); Vec($"{p}.attn2.to_q.bias", hidden);
                Lin($"{p}.attn2.to_k.weight", hidden, hidden); Vec($"{p}.attn2.to_k.bias", hidden);
                Lin($"{p}.attn2.to_v.weight", hidden, hidden); Vec($"{p}.attn2.to_v.bias", hidden);
                Lin($"{p}.attn2.to_out.0.weight", hidden, hidden); Vec($"{p}.attn2.to_out.0.bias", hidden);
                if (c.UseQkNorm)
                {
                    Vec($"{p}.attn2.norm_q.weight", headDim);
                    Vec($"{p}.attn2.norm_k.weight", headDim);
                }
            }

            Lin($"{p}.ff.net.0.proj.weight", ffDim, hidden); Vec($"{p}.ff.net.0.proj.bias", ffDim);
            Lin($"{p}.ff.net.2.weight", hidden, ffDim); Vec($"{p}.ff.net.2.bias", hidden);

            if (!isPreOnly)
            {
                Lin($"{p}.ff_context.net.0.proj.weight", ffDim, hidden); Vec($"{p}.ff_context.net.0.proj.bias", ffDim);
                Lin($"{p}.ff_context.net.2.weight", hidden, ffDim); Vec($"{p}.ff_context.net.2.bias", hidden);
            }
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

    public static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
