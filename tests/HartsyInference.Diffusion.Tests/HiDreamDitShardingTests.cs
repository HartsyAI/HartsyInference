using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT sharding (ROADMAP.md §1): verifies <see cref="HiDreamTransformer.ForwardSharded"/> — a
/// block-range split of the flat double-then-single block loop across two <see cref="CudaBackend"/>s — matches
/// <see cref="HiDreamTransformer.Forward"/> (unsharded, one backend, whole transformer resident) bit-for-bit on
/// a synthetic tiny config (3 double + 3 single blocks). Same-GPU split by design (see
/// <c>Sd3DitShardingTests</c>' class doc). Parameterized over every distinct split-point SHAPE
/// <see cref="HiDreamTransformer.ForwardBlocksRange"/> can hit: split entirely inside the double range (1, 2),
/// split exactly ON the double→single boundary (3 — backend A never transitions, backend B transitions
/// immediately), split MID-RANGE crossing the boundary inside backend A's own range (4 — the highest-risk
/// path, since the transition + per-block Llama conditioning hand-off + MoE routing all have to agree on
/// which "state shape", an (img, encoder) pair or a joint tensor, crosses to backend B), and split entirely
/// inside the single range (5).</summary>
[Trait("Category", "Integration")]
[Collection("CudaSerial")]
public sealed class HiDreamDitShardingTests
{
    private readonly ITestOutputHelper _output;
    public HiDreamDitShardingTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    // 3 double + 3 single blocks (vs. the real 16 + 32) — small enough for a fast synthetic test while still
    // giving the split point (4) real work crossing the double→single boundary mid-range (see class doc).
    // 2 routed experts (both always activated: top-2-of-2) still exercises the MoE gate/renorm/RowGatedAccumulate
    // path, just without any token ever seeing a truly "unselected" expert.
    private static HiDreamConfig TinyConfig => new()
    {
        PatchSize = 2,
        InChannels = 4,
        OutChannels = 4,
        NumLayers = 3,
        NumSingleLayers = 3,
        NumAttentionHeads = 2,
        AttentionHeadDim = 8,
        TextEmbDim = 12,
        AxesDimsRope = [4, 2, 2],
        NumRoutedExperts = 2,
        NumActivatedExperts = 2,
        RopeTheta = 10000f,
    };

    [Theory]
    [InlineData(1)] // A: 1 double block. B: 2 double + 3 single (transitions itself).
    [InlineData(2)] // A: 2 double blocks. B: 1 double + 3 single (transitions itself).
    [InlineData(3)] // A: all 3 double (ends exactly at the boundary, does NOT transition). B: all 3 single (transitions immediately).
    [InlineData(4)] // A: all 3 double + 1 single (crosses the boundary mid-range, transitions itself). B: 2 single.
    [InlineData(5)] // A: all 3 double + 2 single. B: 1 single block only.
    public void ForwardSharded_MatchesUnsharded_BitParity(int splitBlock)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // Same-GPU by design — see class doc.
        const int secondOrdinal = 0;
        _output.WriteLine($"Devices: {CudaContext.GetDeviceCount()}; sharded backend B on ordinal {secondOrdinal} (same-GPU split). splitBlock={splitBlock}");

        HiDreamConfig cfg = TinyConfig;
        int totalBlocks = cfg.NumLayers + cfg.NumSingleLayers;

        // ── Reference: one backend, whole transformer resident, unsharded Forward. ──
        Dictionary<string, Tensor> wRef = HiDreamWeightBuilder.Build(cfg);
        using CudaBackend refBackend = new(deviceOrdinal: 0, PtxDir());
        using HiDreamTransformer refTransformer = new(cfg);
        refTransformer.LoadWeights(wRef);
        refBackend.PreloadWeights(refTransformer.EnumerateWeights());

        int h = 8, wd = 8;
        using Tensor latent = HiDreamWeightBuilder.Rand(new TensorShape(1, cfg.InChannels, h, wd), 100, 0.5f);
        const int t5Seq = 4, llamaSeq = 3;
        using Tensor t5Hidden = HiDreamWeightBuilder.Rand(new TensorShape(1, t5Seq, 16), 200, 0.2f);
        using Tensor pooled = HiDreamWeightBuilder.Rand(new TensorShape(1, cfg.TextEmbDim), 300, 0.2f);
        List<Tensor> llamaLayers = new(totalBlocks);
        for (int i = 0; i < totalBlocks; i++)
            llamaLayers.Add(HiDreamWeightBuilder.Rand(new TensorShape(1, llamaSeq, 16), 400 + i, 0.2f));

        using Tensor velocityRef = refTransformer.Forward(refBackend, latent, 500.0f, t5Hidden, llamaLayers, pooled);
        float[] refValues = ToArray(velocityRef);
        refBackend.FreeWeights(refTransformer.EnumerateWeights());
        HiDreamWeightBuilder.DisposeAll(wRef);

        // ── Sharded: SAME weight VALUES (independent load, same seeds), split across two backends. Backend A
        // gets shared weights + blocks[0,split); backend B gets ONLY blocks[split,totalBlocks) — the asymmetric
        // preload that makes this VRAM-pooling rather than replication. ──
        Dictionary<string, Tensor> wSharded = HiDreamWeightBuilder.Build(cfg);
        using CudaBackend backendA = new(deviceOrdinal: 0, PtxDir());
        using CudaBackend backendB = new(deviceOrdinal: secondOrdinal, PtxDir());
        using HiDreamTransformer shardedTransformer = new(cfg);
        shardedTransformer.LoadWeights(wSharded);

        List<Tensor> aWeights = new(shardedTransformer.EnumerateSharedWeights());
        aWeights.AddRange(shardedTransformer.EnumerateBlockRangeWeights(0, splitBlock));
        backendA.PreloadWeights(aWeights);
        List<Tensor> bWeights = new(shardedTransformer.EnumerateBlockRangeWeights(splitBlock, totalBlocks));
        backendB.PreloadWeights(bWeights);

        using Tensor velocitySharded = shardedTransformer.ForwardSharded(
            backendA, backendB, latent, 500.0f, t5Hidden, llamaLayers, pooled, splitBlock);
        float[] shardedValues = ToArray(velocitySharded);

        backendA.FreeWeights(aWeights);
        backendB.FreeWeights(bWeights);
        HiDreamWeightBuilder.DisposeAll(wSharded);
        foreach (Tensor t in llamaLayers) t.Dispose();

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

/// <summary>Tiny synthetic HiDream weight builder shared by the sharding tests — mirrors <c>Krea2WeightBuilder</c>'s shape.</summary>
internal static class HiDreamWeightBuilder
{
    public static Dictionary<string, Tensor> Build(HiDreamConfig c)
    {
        Dictionary<string, Tensor> w = new();
        int hidden = c.InnerDim;
        int seed = 1;
        void Lin(string k, int o, int i) => w[k] = Rand(new TensorShape(o, i), seed++, 0.06f);
        void Vec(string k, int d) => w[k] = Rand(new TensorShape(d), seed++, 0.02f);

        Lin("x_embedder.proj.weight", hidden, c.InChannels * c.PatchSize * c.PatchSize);
        Vec("x_embedder.proj.bias", hidden);

        Lin("t_embedder.timestep_embedder.linear_1.weight", hidden, 256);
        Vec("t_embedder.timestep_embedder.linear_1.bias", hidden);
        Lin("t_embedder.timestep_embedder.linear_2.weight", hidden, hidden);
        Vec("t_embedder.timestep_embedder.linear_2.bias", hidden);

        Lin("p_embedder.pooled_embedder.linear_1.weight", hidden, c.TextEmbDim);
        Vec("p_embedder.pooled_embedder.linear_1.bias", hidden);
        Lin("p_embedder.pooled_embedder.linear_2.weight", hidden, hidden);
        Vec("p_embedder.pooled_embedder.linear_2.bias", hidden);

        int numCaptionProjections = c.NumLayers + c.NumSingleLayers + 1;
        for (int i = 0; i < numCaptionProjections; i++)
            Lin($"caption_projection.{i}.linear.weight", hidden, 16);

        Lin("final_layer.adaLN_modulation.1.weight", 2 * hidden, hidden);
        Vec("final_layer.adaLN_modulation.1.bias", 2 * hidden);
        Lin("final_layer.linear.weight", c.PatchSize * c.PatchSize * c.OutChannels, hidden);
        Vec("final_layer.linear.bias", c.PatchSize * c.PatchSize * c.OutChannels);

        int headDim = c.AttentionHeadDim;
        int sharedInner = 6, routedInner = 8;
        void MoeFfn(string p)
        {
            Lin($"{p}.shared_experts.w1.weight", sharedInner, hidden);
            Lin($"{p}.shared_experts.w2.weight", hidden, sharedInner);
            Lin($"{p}.shared_experts.w3.weight", sharedInner, hidden);
            for (int e = 0; e < c.NumRoutedExperts; e++)
            {
                Lin($"{p}.experts.{e}.w1.weight", routedInner, hidden);
                Lin($"{p}.experts.{e}.w2.weight", hidden, routedInner);
                Lin($"{p}.experts.{e}.w3.weight", routedInner, hidden);
            }
            Lin($"{p}.gate.weight", c.NumRoutedExperts, hidden);
        }

        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"double_stream_blocks.{i}.block";
            Lin($"{p}.adaLN_modulation.1.weight", 12 * hidden, hidden);
            Vec($"{p}.adaLN_modulation.1.bias", 12 * hidden);

            Lin($"{p}.attn1.to_q.weight", hidden, hidden);
            Lin($"{p}.attn1.to_k.weight", hidden, hidden);
            Lin($"{p}.attn1.to_v.weight", hidden, hidden);
            Lin($"{p}.attn1.to_out.weight", hidden, hidden);
            Vec($"{p}.attn1.q_rms_norm.weight", headDim);
            Vec($"{p}.attn1.k_rms_norm.weight", headDim);

            Lin($"{p}.attn1.to_q_t.weight", hidden, hidden);
            Lin($"{p}.attn1.to_k_t.weight", hidden, hidden);
            Lin($"{p}.attn1.to_v_t.weight", hidden, hidden);
            Lin($"{p}.attn1.to_out_t.weight", hidden, hidden);
            Vec($"{p}.attn1.q_rms_norm_t.weight", headDim);
            Vec($"{p}.attn1.k_rms_norm_t.weight", headDim);

            MoeFfn($"{p}.ff_i");

            Lin($"{p}.ff_t.w1.weight", routedInner, hidden);
            Lin($"{p}.ff_t.w2.weight", hidden, routedInner);
            Lin($"{p}.ff_t.w3.weight", routedInner, hidden);
        }

        for (int i = 0; i < c.NumSingleLayers; i++)
        {
            string p = $"single_stream_blocks.{i}.block";
            Lin($"{p}.adaLN_modulation.1.weight", 6 * hidden, hidden);
            Vec($"{p}.adaLN_modulation.1.bias", 6 * hidden);

            Lin($"{p}.attn1.to_q.weight", hidden, hidden);
            Lin($"{p}.attn1.to_k.weight", hidden, hidden);
            Lin($"{p}.attn1.to_v.weight", hidden, hidden);
            Lin($"{p}.attn1.to_out.weight", hidden, hidden);
            Vec($"{p}.attn1.q_rms_norm.weight", headDim);
            Vec($"{p}.attn1.k_rms_norm.weight", headDim);

            MoeFfn($"{p}.ff_i");
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
