using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Tensor-parallel (v1) contracts, mirroring <see cref="LlmPlacementTests"/>' synthetic harness: a
/// degree-2 TP forward over two CPU backends + <see cref="HostStagedComm"/> must reproduce the unstaged
/// <c>GenericTransformer.ForwardEmbeds</c> bit-close, the weight partitioner's row/column slices must
/// reassemble to the full tensors exactly (a dropped or shifted block silently corrupts a projection), and
/// bad degrees / indivisible head counts / quant-misaligned column splits must be refused loudly at
/// construction or load — never discovered as wrong output. CPU backend, synthetic weights: Unit tier.</summary>
public sealed unsafe class TensorParallelTests
{
    private static uint _rng = 0x9E377u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int n) => Fill(new Tensor(new TensorShape(n), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Embeds(int t, int h) => Fill(new Tensor(new TensorShape(1, t, h), DType.F32));

    private static TransformerConfig Config() => new()
    {
        HiddenSize = 16, NumLayers = 4, NumHeads = 4, NumKvHeads = 2, HeadDim = 8,
        IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = false, QkNorm = true,
    };

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim, hd = c.HeadDim;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(c.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
            if (c.QkNorm)
            {
                w[$"{p}.self_attn.q_norm.weight"] = Ones(hd);
                w[$"{p}.self_attn.k_norm.weight"] = Ones(hd);
            }
            if (c.AttentionBias)
            {
                w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
                w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
                w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
                w[$"{p}.self_attn.o_proj.bias"] = F1(h);
            }
        }
        return w;
    }

    /// <summary>Degree-2 TP over two distinct CPU backends + host-staged AllReduce vs the unstaged forward:
    /// prefill 4 tokens then two 1-token decode steps (the shapes the real pipeline drives), 5-decimal parity
    /// on hidden state AND logits, per-rank KV bookkeeping advancing in lockstep with the unstaged cache.</summary>
    [Fact]
    public void ForwardEmbedsTp_DegreeTwo_MatchesUnstaged_AcrossPrefillAndDecode()
    {
        TransformerConfig cfg = Config();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend rank0 = new();
        using CpuBackend rank1 = new();
        using HostStagedComm comm = new(2);
        TpPlacement placement = new([rank0, rank1], comm);

        using GenericTransformer oracle = new(cfg);
        oracle.LoadWeights(w, "model");
        using TensorParallelTransformer tp = new(cfg, placement);
        tp.LoadWeights(w, "model");

        using KvCache unstagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        KvCache[] tpCaches = tp.CreateKvCaches();
        try
        {
            int[] steps = [4, 1, 1];
            int pos = 0;
            foreach (int t in steps)
            {
                using Tensor embeds = Embeds(t, cfg.HiddenSize);
                using Tensor embedsCopy = new(embeds.Shape, DType.F32);
                rank0.CopyTo(embedsCopy, embeds);

                using Tensor expected = oracle.ForwardEmbeds(rank0, embeds, t, pos, unstagedCache);
                using Tensor actual = tp.ForwardEmbedsTp(embedsCopy, t, pos, tpCaches);

                Assert.Equal(expected.Shape, actual.Shape);
                float* ep = (float*)expected.DataPointer;
                float* ap = (float*)actual.DataPointer;
                for (long i = 0; i < expected.ElementCount; i++)
                {
                    // Tolerance (not decimal-rounding) equality: the row-parallel partial sums genuinely
                    // reassociate the o_proj/down_proj dot products, so ~1e-7 diffs are expected and a value
                    // sitting on a rounding boundary would fail the 5-decimal form spuriously.
                    Assert.Equal(ep[i], ap[i], 1e-5);
                }

                using Tensor expectedLogits = oracle.ProjectLogits(rank0, expected, t);
                using Tensor actualLogits = tp.ProjectLogits(actual, t);
                float* elp = (float*)expectedLogits.DataPointer;
                float* alp = (float*)actualLogits.DataPointer;
                for (long i = 0; i < expectedLogits.ElementCount; i++)
                {
                    Assert.Equal(elp[i], alp[i], 1e-5);
                }

                pos += t;
                // Advance contract: each rank's cache advances exactly once per TP call, matching unstaged.
                Assert.Equal(pos, unstagedCache.CurrentLength);
                foreach (KvCache cache in tpCaches)
                {
                    Assert.Equal(pos, cache.CurrentLength);
                }
            }
        }
        finally
        {
            foreach (KvCache cache in tpCaches) cache.Dispose();
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>Qwen2-shaped variant (QKV bias, no QK-norm) including an o_proj bias: Q/K/V biases must split
    /// with their head rows, and the row-parallel o bias must enter the AllReduce sum exactly ONCE (rank 0
    /// only) — a replicated o bias would be silently added degree× with no crash.</summary>
    [Fact]
    public void ForwardEmbedsTp_AttentionBias_MatchesUnstaged_ObiasCountedOnce()
    {
        TransformerConfig cfg = Config() with { AttentionBias = true, QkNorm = false };
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend rank0 = new();
        using CpuBackend rank1 = new();
        using HostStagedComm comm = new(2);
        TpPlacement placement = new([rank0, rank1], comm);

        using GenericTransformer oracle = new(cfg);
        oracle.LoadWeights(w, "model");
        using TensorParallelTransformer tp = new(cfg, placement);
        tp.LoadWeights(w, "model");

        using KvCache unstagedCache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        KvCache[] tpCaches = tp.CreateKvCaches();
        try
        {
            int[] steps = [4, 1, 1];
            int pos = 0;
            foreach (int t in steps)
            {
                using Tensor embeds = Embeds(t, cfg.HiddenSize);
                using Tensor embedsCopy = new(embeds.Shape, DType.F32);
                rank0.CopyTo(embedsCopy, embeds);

                using Tensor expected = oracle.ForwardEmbeds(rank0, embeds, t, pos, unstagedCache);
                using Tensor actual = tp.ForwardEmbedsTp(embedsCopy, t, pos, tpCaches);

                Assert.Equal(expected.Shape, actual.Shape);
                float* ep = (float*)expected.DataPointer;
                float* ap = (float*)actual.DataPointer;
                for (long i = 0; i < expected.ElementCount; i++)
                {
                    Assert.Equal(ep[i], ap[i], 1e-5);
                }
                pos += t;
            }
        }
        finally
        {
            foreach (KvCache cache in tpCaches) cache.Dispose();
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>Fills a tensor's raw byte storage with a position-dependent pattern so slice comparisons
    /// detect any offset/stride error, for any dtype including block-quantized ones.</summary>
    private static Tensor PatternBytes(TensorShape shape, DType dtype)
    {
        Tensor t = new(shape, dtype);
        long bytes = dtype.ComputeByteCount(t.ElementCount);
        byte* p = (byte*)t.DataPointer;
        for (long i = 0; i < bytes; i++) p[i] = (byte)(i * 31 + 7);
        return t;
    }

    /// <summary>Column-parallel (OUT-row) slices concatenate back to the source bytes exactly — F32 and Q8_0
    /// (rows are whole quant blocks, so a row range never splits a block).</summary>
    [Theory]
    [InlineData("F32")]
    [InlineData("Q8_0")]
    public void WeightPartition_OutRowSlices_ReassembleExactly(string dtypeName)
    {
        DType dtype = dtypeName == "F32" ? DType.F32 : DType.Q8_0;
        const int rows = 4, cols = 64, degree = 2;
        using Tensor src = PatternBytes(new TensorShape(rows, cols), dtype);
        long srcBytes = dtype.ComputeByteCount((long)rows * cols);
        long sliceBytes = 0;
        byte* sp = (byte*)src.DataPointer;
        long offset = 0;
        for (int r = 0; r < degree; r++)
        {
            using Tensor slice = TensorParallelTransformer.SliceOutRows(src, r, degree, "test");
            Assert.Equal(rows / degree, (int)slice.Shape[0]);
            Assert.Equal(cols, (int)slice.Shape[1]);
            long bytes = dtype.ComputeByteCount(slice.ElementCount);
            byte* dp = (byte*)slice.DataPointer;
            for (long i = 0; i < bytes; i++)
            {
                Assert.Equal(sp[offset + i], dp[i]);
            }
            offset += bytes;
            sliceBytes += bytes;
        }
        Assert.Equal(srcBytes, sliceBytes);
    }

    /// <summary>Row-parallel (IN-column) slices tile every source row exactly — each rank's per-row byte range
    /// lands at rank·(row bytes / degree), verified byte-for-byte for F32 and a block-aligned Q8_0 split.</summary>
    [Theory]
    [InlineData("F32")]
    [InlineData("Q8_0")]
    public void WeightPartition_InColSlices_ReassembleExactly(string dtypeName)
    {
        DType dtype = dtypeName == "F32" ? DType.F32 : DType.Q8_0;
        const int rows = 4, cols = 64, degree = 2;
        using Tensor src = PatternBytes(new TensorShape(rows, cols), dtype);
        long srcRowBytes = dtype.ComputeByteCount(cols);
        long dstRowBytes = dtype.ComputeByteCount(cols / degree);
        byte* sp = (byte*)src.DataPointer;
        long coveredBytes = 0;
        for (int r = 0; r < degree; r++)
        {
            using Tensor slice = TensorParallelTransformer.SliceInCols(src, r, degree, "test");
            Assert.Equal(rows, (int)slice.Shape[0]);
            Assert.Equal(cols / degree, (int)slice.Shape[1]);
            byte* dp = (byte*)slice.DataPointer;
            for (long row = 0; row < rows; row++)
            {
                for (long i = 0; i < dstRowBytes; i++)
                {
                    Assert.Equal(sp[row * srcRowBytes + r * dstRowBytes + i], dp[row * dstRowBytes + i]);
                }
            }
            coveredBytes += dtype.ComputeByteCount(slice.ElementCount);
        }
        Assert.Equal(dtype.ComputeByteCount((long)rows * cols), coveredBytes);
    }

    /// <summary>A quantized down_proj whose per-rank IN-column count is not a whole number of blocks must be
    /// refused at LOAD (Q8_0 blocks are 32 elements; 32 columns over degree 2 gives 16/rank) — slicing mid-block
    /// would silently scramble the weight, so this failure must be loud and early.</summary>
    [Fact]
    public void LoadWeights_RejectsMisalignedQuantColumns()
    {
        TransformerConfig cfg = Config() with { NumLayers = 1 };
        Dictionary<string, Tensor> w = Weights(cfg);
        Tensor f32Down = w["model.layers.0.mlp.down_proj.weight"];
        f32Down.Dispose();
        w["model.layers.0.mlp.down_proj.weight"] = PatternBytes(new TensorShape(cfg.HiddenSize, cfg.IntermediateSize), DType.Q8_0);

        using CpuBackend rank0 = new();
        using CpuBackend rank1 = new();
        using HostStagedComm comm = new(2);
        using TensorParallelTransformer tp = new(cfg, new TpPlacement([rank0, rank1], comm));
        Assert.Throws<NotSupportedException>(() => tp.LoadWeights(w, "model"));

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void Construction_RejectsIndivisibleGeometryAndUnsupportedConfig()
    {
        using CpuBackend rank0 = new();
        using CpuBackend rank1 = new();
        using HostStagedComm comm = new(2);
        TpPlacement placement = new([rank0, rank1], comm);

        // 3 KV heads cannot tile across 2 ranks (whole-KV-head slices per rank).
        TransformerConfig oddKv = Config() with { NumHeads = 6, NumKvHeads = 3 };
        Assert.Throws<ArgumentException>(() => new TensorParallelTransformer(oddKv, placement));

        // 5 Q heads cannot tile across 2 ranks.
        TransformerConfig oddQ = Config() with { NumHeads = 5, NumKvHeads = 1 };
        Assert.Throws<ArgumentException>(() => new TensorParallelTransformer(oddQ, placement));

        // FFN columns must tile too (33 is not divisible by 2).
        TransformerConfig oddFfn = Config() with { IntermediateSize = 33 };
        Assert.Throws<ArgumentException>(() => new TensorParallelTransformer(oddFfn, placement));

        // Outside the v1 dense shape: refused as NotSupported, never a silent wrong forward.
        Assert.Throws<NotSupportedException>(() => new TensorParallelTransformer(Config() with { ParallelResidual = true }, placement));
        Assert.Throws<NotSupportedException>(() => new TensorParallelTransformer(Config() with { UseLayerNorm = true }, placement));
        Assert.Throws<NotSupportedException>(() => new TensorParallelTransformer(
            Config() with { Moe = new MoeConfig { NumExperts = 4, NumExpertsPerTok = 2, MoeIntermediateSize = 16 } }, placement));
    }

    [Fact]
    public void TpPlacement_RejectsBadDegreeAndCommMismatch()
    {
        using CpuBackend a = new();
        using CpuBackend b = new();
        using CpuBackend c = new();
        using HostStagedComm comm2 = new(2);

        // Degree 1 is just the single-device path — TP must not pretend to run it.
        Assert.Throws<ArgumentException>(() => new TpPlacement([a], comm2));
        // Communicator rank count must match the backend list.
        Assert.Throws<ArgumentException>(() => new TpPlacement([a, b, c], comm2));
        // Caches-per-rank contract on the forward entry.
        TransformerConfig cfg = Config();
        Dictionary<string, Tensor> w = Weights(cfg);
        using TensorParallelTransformer tp = new(cfg, new TpPlacement([a, b], comm2));
        tp.LoadWeights(w, "model");
        using Tensor embeds = Embeds(1, cfg.HiddenSize);
        using KvCache lone = new(cfg.NumLayers, 1, cfg.NumKvHeads / 2, cfg.HeadDim);
        Assert.Throws<ArgumentException>(() => tp.ForwardEmbedsTp(embeds, 1, 0, [lone]));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>The per-rank weight enumeration must tile with no tensor shared between ranks (each slice has
    /// exactly one owning backend — sharing one Tensor across device binders is the documented hazard) and
    /// rank 0 must carry the head-side extras (final norm + lm_head).</summary>
    [Fact]
    public void EnumerateRankWeights_DisjointAcrossRanks_HeadOnRankZero()
    {
        TransformerConfig cfg = Config();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend rank0 = new();
        using CpuBackend rank1 = new();
        using HostStagedComm comm = new(2);
        using TensorParallelTransformer tp = new(cfg, new TpPlacement([rank0, rank1], comm));
        tp.LoadWeights(w, "model");

        HashSet<Tensor> set0 = new(tp.EnumerateRankWeights(0), ReferenceEqualityComparer.Instance);
        HashSet<Tensor> set1 = new(tp.EnumerateRankWeights(1), ReferenceEqualityComparer.Instance);
        Assert.True(set0.Count > 0 && set1.Count > 0);
        Assert.Empty(set0.Intersect(set1, ReferenceEqualityComparer.Instance));
        // Rank 0 carries final norm + (tied F32) head: 2 extras beyond the per-layer set.
        Assert.Equal(set1.Count + 2, set0.Count);

        foreach (Tensor t in w.Values) t.Dispose();
    }
}
