using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The speculative-decode merge on CUDA and on Vulkan against <see cref="DecodeMergeReference"/>, and on each
/// chained after the decode that leaves the prefix output and its LSE on the device. Tolerances were fixed before either
/// kernel first ran: 2e-5 against the reference for one merge, 5e-5 for a decode and a merge chained against one causal
/// decode.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class DecodeMergeGpuTests
{
    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    /// <summary>One verify step: queries, a cache of committed rows, the step's own keys and values, and the prefix-only
    /// decode of the step over the cache, computed by the host reference.</summary>
    private sealed class Step : IDisposable
    {
        public readonly int Batch, Heads, KvHeads, Dim, Rows, Capacity;
        public readonly Tensor Query, CacheKey, CacheValue, Key, Value, Prefix, Lse, Merged, Expected;

        public Step(int batch, int heads, int kvHeads, int dim, int rows, int capacity, int[] lengths)
        {
            (Batch, Heads, KvHeads, Dim, Rows, Capacity) = (batch, heads, kvHeads, dim, rows, capacity);
            Query = Random(new TensorShape(batch, heads, rows, dim), 61);
            CacheKey = Random(new TensorShape(batch, kvHeads, capacity, dim), 62);
            CacheValue = Random(CacheKey.Shape, 63);
            Key = Random(new TensorShape(batch, kvHeads, rows, dim), 64);
            Value = Random(Key.Shape, 65);
            Prefix = new Tensor(new TensorShape(batch, rows, heads * dim), DType.F32);
            Lse = new Tensor(new TensorShape(batch, heads, rows), DType.F32);
            Merged = new Tensor(Prefix.Shape, DType.F32);
            Expected = new Tensor(Prefix.Shape, DType.F32);
            DecodeLengthsReference.Run(Prefix, Query, CacheKey, CacheValue, lengths, causal: false, Lse.AsSpan<float>());
        }

        /// <summary>The reference's merge of the current prefix, into <see cref="Expected"/>.</summary>
        public void Reference(ReadOnlySpan<int> mask, float? scale = null) =>
            DecodeMergeReference.TreeMerge(Expected, Prefix, Lse.AsReadOnlySpan<float>(), Query, Key, Value, mask, scale);

        public void AssertMerged(Tensor actual, float tolerance, string what)
        {
            ReadOnlySpan<float> expected = Expected.AsReadOnlySpan<float>(), values = actual.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - values[i]) <= tolerance, $"{what} {i}: {values[i]}, expected {expected[i]}");
        }

        public void Dispose()
        {
            foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[Query, CacheKey, CacheValue, Key, Value, Prefix, Lse, Merged, Expected]) tensor.Dispose();
        }
    }

    /// <summary>A tree mask: each row names itself's predecessors by a fixed pattern, and every third row names nothing.</summary>
    private static int[] TreeMask(int rows)
    {
        int[] mask = new int[rows];
        Random random = new(7);
        for (int r = 0; r < rows; r++)
        {
            int bits = rows == 32 ? random.Next() ^ (random.Next() << 1) : random.Next(1 << rows);
            mask[r] = r % 3 == 2 ? 0 : bits | (1 << r);
        }
        return mask;
    }

    public static TheoryData<int, int, int, int, int, int, int[]> Shapes => new()
    {
        // batch, heads, kv heads, head dim, step rows, cache capacity, prefix lengths
        { 1, 4, 2, 64, 5, 48, new[] { 40 } },
        { 2, 6, 2, 64, 4, 40, new[] { 33, 17 } },
        { 1, 4, 4, 128, 1, 32, new[] { 32 } },
        { 1, 8, 1, 256, 6, 48, new[] { 29 } },     // a head dim the decode kernels do not take; the merge takes any
        { 2, 4, 2, 64, 3, 16, new[] { 0, 9 } },    // an empty prefix in one sequence
        { 1, 2, 2, 16, 32, 9, new[] { 9 } },       // the most rows a step holds, at a head dim below the warp width
    };

    [CudaTheory]
    [MemberData(nameof(Shapes))]
    public void TheCudaKernelMatchesTheReferenceForAStepAndForATree(int batch, int heads, int kvHeads, int dim, int rows, int capacity, int[] lengths)
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        MatchesTheReference(backend, batch, heads, kvHeads, dim, rows, capacity, lengths);
    }

    [VulkanFact]
    public void TheVulkanKernelMatchesTheReferenceForAStepAndForATree()
    {
        using VulkanBackend backend = new();
        foreach (object[] shape in Shapes)
            MatchesTheReference(backend, (int)shape[0], (int)shape[1], (int)shape[2], (int)shape[3], (int)shape[4], (int)shape[5], (int[])shape[6]);
    }

    private static void MatchesTheReference(IBackend backend, int batch, int heads, int kvHeads, int dim, int rows, int capacity, int[] lengths)
    {
        Assert.True(backend.HasDecodeMerge);
        using Step step = new(batch, heads, kvHeads, dim, rows, capacity, lengths);
        foreach (int[] mask in (int[][])[DecodeMergeReference.CausalStepMask(rows), TreeMask(rows)])
        {
            step.Reference(mask);
            step.Merged.AsSpan<float>().Fill(-7f);
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(16);
            backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Lse, step.Query, step.Key, step.Value, mask);
            backend.Sync();
            step.AssertMerged(step.Merged, 2e-5f, "merged");
            Assert.Equal(rows * sizeof(int), Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.StagingCopy).Bytes);
        }
        // A custom scale.
        int[] causal = DecodeMergeReference.CausalStepMask(rows);
        step.Reference(causal, 0.21f);
        backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Lse, step.Query, step.Key, step.Value, causal, 0.21f);
        backend.Sync();
        step.AssertMerged(step.Merged, 2e-5f, "scaled");
    }

    /// <summary>The device pipeline the merge exists for: a prefix-only decode that leaves its output and LSE on the device,
    /// then the merge, with nothing read back in between. The result is the causal decode over a cache that already holds
    /// the step's rows.</summary>
    [CudaFact]
    public void OnCudaADecodeThenAMergeOnTheDeviceIsTheCausalDecodeOverTheJoinedCache()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        ADecodeThenAMergeOnTheDeviceIsTheCausalDecodeOverTheJoinedCache(backend);
    }

    [VulkanFact]
    public void OnVulkanADecodeThenAMergeOnTheDeviceIsTheCausalDecodeOverTheJoinedCache()
    {
        using VulkanBackend backend = new();
        ADecodeThenAMergeOnTheDeviceIsTheCausalDecodeOverTheJoinedCache(backend);
    }

    private static void ADecodeThenAMergeOnTheDeviceIsTheCausalDecodeOverTheJoinedCache(IBackend backend)
    {
        const int batch = 2, heads = 8, kvHeads = 2, dim = 64, rows = 5, capacity = 120;
        int[] prefix = [100, 37];
        using Step step = new(batch, heads, kvHeads, dim, rows, capacity, prefix);
        using Tensor devicePrefix = new(step.Prefix.Shape, DType.F32);
        using Tensor deviceLse = new(step.Lse.Shape, DType.F32);
        AttentionCall decode = AttentionCall.FromDecodeWithLengths(devicePrefix, step.Query, step.CacheKey, step.CacheValue, prefix, causal: false,
            1f / MathF.Sqrt(dim), deviceLse);
        backend.Attention(in decode);
        backend.DecodeTreeMerge(step.Merged, devicePrefix, deviceLse, step.Query, step.Key, step.Value, DecodeMergeReference.CausalStepMask(rows));
        backend.Sync();

        using Tensor joinedKey = new(new TensorShape(batch, kvHeads, capacity + rows, dim), DType.F32);
        using Tensor joinedValue = new(joinedKey.Shape, DType.F32);
        int[] lengths = new int[batch];
        for (int b = 0; b < batch; b++)
        {
            lengths[b] = prefix[b] + rows;
            for (int kh = 0; kh < kvHeads; kh++)
            {
                int from = (b * kvHeads + kh) * capacity * dim, to = (b * kvHeads + kh) * (capacity + rows) * dim, stepAt = (b * kvHeads + kh) * rows * dim;
                step.CacheKey.AsReadOnlySpan<float>().Slice(from, prefix[b] * dim).CopyTo(joinedKey.AsSpan<float>().Slice(to));
                step.CacheValue.AsReadOnlySpan<float>().Slice(from, prefix[b] * dim).CopyTo(joinedValue.AsSpan<float>().Slice(to));
                step.Key.AsReadOnlySpan<float>().Slice(stepAt, rows * dim).CopyTo(joinedKey.AsSpan<float>().Slice(to + prefix[b] * dim));
                step.Value.AsReadOnlySpan<float>().Slice(stepAt, rows * dim).CopyTo(joinedValue.AsSpan<float>().Slice(to + prefix[b] * dim));
            }
        }
        DecodeLengthsReference.Run(step.Expected, step.Query, joinedKey, joinedValue, lengths, causal: true);
        step.AssertMerged(step.Merged, 5e-5f, "chained");
    }

    [CudaFact]
    public void OnCudaMergingInPlaceGivesTheSameResult()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        MergingInPlaceGivesTheSameResult(backend);
    }

    [VulkanFact]
    public void OnVulkanMergingInPlaceGivesTheSameResult()
    {
        using VulkanBackend backend = new();
        MergingInPlaceGivesTheSameResult(backend);
    }

    private static void MergingInPlaceGivesTheSameResult(IBackend backend)
    {
        using Step step = new(2, 4, 2, 64, 5, 25, [25, 11]);
        int[] mask = DecodeMergeReference.CausalStepMask(5);
        step.Reference(mask);
        backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Lse, step.Query, step.Key, step.Value, mask);
        backend.Sync();
        float[] separate = step.Merged.AsReadOnlySpan<float>().ToArray();
        backend.DecodeTreeMerge(step.Prefix, step.Prefix, step.Lse, step.Query, step.Key, step.Value, mask);
        backend.Sync();
        Assert.True(separate.AsSpan().SequenceEqual(step.Prefix.AsReadOnlySpan<float>()));
        step.AssertMerged(step.Prefix, 2e-5f, "in place");
    }

    [CudaFact]
    public void OnCudaAnEmptyPrefixAndAStepRowNoBitNamesAreNotRead()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        AnEmptyPrefixAndAStepRowNoBitNamesAreNotRead(backend);
    }

    [VulkanFact]
    public void OnVulkanAnEmptyPrefixAndAStepRowNoBitNamesAreNotRead()
    {
        using VulkanBackend backend = new();
        AnEmptyPrefixAndAStepRowNoBitNamesAreNotRead(backend);
    }

    private static void AnEmptyPrefixAndAStepRowNoBitNamesAreNotRead(IBackend backend)
    {
        const int heads = 4, rows = 4, dim = 64;
        using Step step = new(2, heads, 2, dim, rows, 12, [0, 12]);
        // Step row 3 is named by no row: its keys and values may hold anything. Sequence 0 has no prefix: so may its prefix output.
        int[] mask = [0b0001, 0b0011, 0b0101, 0b0111];
        step.Reference(mask);
        for (int bh = 0; bh < 2 * 2; bh++)
        {
            step.Key.AsSpan<float>().Slice((bh * rows + 3) * dim, dim).Fill(float.NaN);
            step.Value.AsSpan<float>().Slice((bh * rows + 3) * dim, dim).Fill(float.NaN);
        }
        step.Prefix.AsSpan<float>().Slice(0, rows * heads * dim).Fill(float.NaN);
        backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Lse, step.Query, step.Key, step.Value, mask);
        backend.Sync();
        Assert.DoesNotContain(step.Merged.AsReadOnlySpan<float>().ToArray(), float.IsNaN);
        step.AssertMerged(step.Merged, 2e-5f, "merged");
    }

    [CudaFact]
    public void OnCudaARowWithNoKeyAtAllIsZero()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        ARowWithNoKeyAtAllIsZero(backend);
    }

    [VulkanFact]
    public void OnVulkanARowWithNoKeyAtAllIsZero()
    {
        using VulkanBackend backend = new();
        ARowWithNoKeyAtAllIsZero(backend);
    }

    private static void ARowWithNoKeyAtAllIsZero(IBackend backend)
    {
        const int heads = 4, rows = 3, dim = 64, width = heads * dim;
        using Step step = new(1, heads, 2, dim, rows, 8, [0]);
        step.Merged.AsSpan<float>().Fill(7f);
        backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Lse, step.Query, step.Key, step.Value, [0b001, 0, 0b101]);
        backend.Sync();
        ReadOnlySpan<float> merged = step.Merged.AsReadOnlySpan<float>();
        Assert.Equal(-1, merged.Slice(width, width).IndexOfAnyExcept(0f));
        Assert.NotEqual(-1, merged.Slice(0, width).IndexOfAnyExcept(0f));
        Assert.Equal(-1, merged.IndexOf(7f));
    }

    [CudaFact]
    public void OnCudaAMalformedMergeIsRefusedBeforeAnythingIsWritten()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        AMalformedMergeIsRefusedBeforeAnythingIsWritten(backend);
    }

    [VulkanFact]
    public void OnVulkanAMalformedMergeIsRefusedBeforeAnythingIsWritten()
    {
        using VulkanBackend backend = new();
        AMalformedMergeIsRefusedBeforeAnythingIsWritten(backend);
    }

    private static void AMalformedMergeIsRefusedBeforeAnythingIsWritten(IBackend backend)
    {
        using Step step = new(1, 4, 2, 64, 4, 12, [12]);
        int[] mask = DecodeMergeReference.CausalStepMask(4);
        step.Merged.AsSpan<float>().Fill(7f);
        using Tensor wrongLse = new(new TensorShape(1, 4, 5), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.DecodeTreeMerge(step.Merged, step.Prefix, wrongLse, step.Query, step.Key, step.Value, mask));
        Assert.Throws<ArgumentException>(() => backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Lse, step.Query, step.Key, step.Value, [1, 3, 7, 0b10000]));
        Assert.Throws<ArgumentException>(() => backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Lse, step.Query, step.Key, step.CacheValue, mask));
        Assert.Throws<ArgumentException>(() => backend.DecodeTreeMerge(step.Merged, step.Prefix, step.Prefix, step.Query, step.Key, step.Value, mask));
        backend.Sync();
        Assert.Equal(-1, step.Merged.AsReadOnlySpan<float>().IndexOfAnyExcept(7f));
    }
}
