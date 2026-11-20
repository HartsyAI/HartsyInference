using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Transformer;

/// <summary>The paged cache presents the same keys and values as a contiguous <see cref="FixedKvCache"/> fed the same
/// appends, through prefill, decode, rollback, reset and a rewrite of the same position, and it copies each row out of its
/// pages once instead of on every call.</summary>
[Collection("AttentionSemantics")]
public sealed class PagedKvCacheTests
{
    private const int KvHeads = 2, PageSize = 4, MaxPages = 64, MaxTokens = PageSize * MaxPages;
    private static readonly int[] HeadDims = [4, 6, 4];

    /// <summary>A paged cache and a contiguous reference driven with identical data.</summary>
    private sealed class Pair : IDisposable
    {
        public readonly CpuBackend Backend = new();
        public readonly PagedKvPool Pool;
        public readonly PagedKvCache Paged;
        public readonly FixedKvCache Fixed;
        private readonly Random _random = new(1234);

        public Pair(int expectedTokens = 0, int maxPages = MaxPages)
        {
            Pool = new PagedKvPool(HeadDims.Length, KvHeads, HeadDims, PageSize, maxPages);
            Paged = new PagedKvCache(Pool, expectedTokens);
            Fixed = new FixedKvCache(HeadDims.Length, 1, KvHeads, HeadDims, PageSize * maxPages, DType.F32);
        }

        /// <summary>Appends <paramref name="tokens"/> new rows to one layer of both caches, without advancing.</summary>
        public void Append(int layer, int tokens)
        {
            using Tensor k = Rows(HeadDims[layer], tokens), v = Rows(HeadDims[layer], tokens);
            Paged.AppendStep(Backend, layer, k, v);
            Fixed.AppendStep(Backend, layer, k, v);
        }

        /// <summary>One model step: every layer appends, attends over its prefix (checked here), then the length advances.</summary>
        public void Step(int tokens)
        {
            for (int layer = 0; layer < HeadDims.Length; layer++)
            {
                Append(layer, tokens);
                AssertLayerMatches(layer, Paged.CurrentLength + tokens);
            }
            Paged.AdvanceLength(tokens);
            Fixed.AdvanceLength(tokens);
        }

        public void Truncate(int length)
        {
            Paged.Truncate(length);
            Fixed.Truncate(length);
        }

        public void Reset()
        {
            Paged.Reset();
            Fixed.Reset();
        }

        public void AssertLayerMatches(int layer, int validRows)
        {
            AssertSameRows(Paged.KeyPrefix(layer), Fixed.KeyPrefix(layer), validRows, HeadDims[layer], $"layer {layer} keys");
            AssertSameRows(Paged.ValuePrefix(layer), Fixed.ValuePrefix(layer), validRows, HeadDims[layer], $"layer {layer} values");
        }

        private Tensor Rows(int headDim, int tokens)
        {
            Tensor tensor = new(new TensorShape(1, KvHeads, tokens, headDim), DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)_random.NextDouble();
            return tensor;
        }

        public void Dispose()
        {
            Paged.Dispose();
            Fixed.Dispose();
            Pool.Dispose();
            Backend.Dispose();
        }
    }

    private static void AssertSameRows(Tensor paged, Tensor contiguous, int validRows, int headDim, string what)
    {
        Assert.True(paged.Shape[2] >= validRows, $"{what}: the paged prefix holds {paged.Shape[2]} rows, fewer than the {validRows} valid ones.");
        ReadOnlySpan<float> actual = paged.AsReadOnlySpan<float>(), expected = contiguous.AsReadOnlySpan<float>();
        int pagedRows = (int)paged.Shape[2], contiguousRows = (int)contiguous.Shape[2];
        for (int head = 0; head < KvHeads; head++)
            for (int row = 0; row < validRows; row++)
                Assert.True(
                    actual.Slice((head * pagedRows + row) * headDim, headDim).SequenceEqual(expected.Slice((head * contiguousRows + row) * headDim, headDim)),
                    $"{what}: head {head}, row {row} of {validRows} differs.");
    }

    [Fact]
    public void PrefillThenDecodeMatchesAContiguousCache()
    {
        using Pair pair = new();
        pair.Step(10);
        for (int step = 0; step < 30; step++) pair.Step(1);
        Assert.Equal(40, pair.Paged.CurrentLength);
    }

    [Fact]
    public void RollbackAndResetMatchAContiguousCache()
    {
        using Pair pair = new();
        pair.Step(9);
        for (int step = 0; step < 12; step++) pair.Step(1);
        pair.Truncate(13);
        for (int step = 0; step < 5; step++) pair.Step(1);
        pair.Truncate(3);
        pair.Step(6);
        pair.Reset();
        pair.Step(7);
        pair.Step(1);
        Assert.Equal(8, pair.Paged.CurrentLength);
    }

    [Fact]
    public void RewritingTheSamePositionReplacesWhatWasThere()
    {
        using Pair pair = new();
        pair.Step(6);
        // A step that runs twice before the length advances (a rejected draft, a retried round) writes the same rows again.
        for (int layer = 0; layer < HeadDims.Length; layer++)
        {
            pair.Append(layer, 2);
            pair.AssertLayerMatches(layer, 8);
        }
        for (int layer = 0; layer < HeadDims.Length; layer++)
        {
            pair.Append(layer, 2);
            pair.AssertLayerMatches(layer, 8);
        }
        for (int layer = 0; layer < HeadDims.Length; layer++)
        {
            pair.Append(layer, 1);
            pair.AssertLayerMatches(layer, 7);
        }
        pair.Paged.AdvanceLength(1);
        pair.Fixed.AdvanceLength(1);
        pair.Step(1);
    }

    [Fact]
    public void ALayerReadBeforeLaterLayersAppendSeesItsOwnRows()
    {
        using Pair pair = new();
        pair.Step(5);
        // The layer loop interleaves append and attend: layer 0 reads its prefix while layers 1 and 2 still hold last step's rows.
        pair.Append(0, 1);
        pair.AssertLayerMatches(0, 6);
        pair.AssertLayerMatches(1, 5);
        pair.Append(1, 1);
        pair.AssertLayerMatches(1, 6);
        pair.Append(2, 1);
        pair.AssertLayerMatches(2, 6);
        pair.Paged.AdvanceLength(1);
        pair.Fixed.AdvanceLength(1);
        for (int layer = 0; layer < HeadDims.Length; layer++) pair.AssertLayerMatches(layer, 6);
    }

    /// <summary>Bytes one row of every layer's keys and values takes.</summary>
    private static long RowBytes => HeadDims.Sum(headDim => 2L * KvHeads * headDim * sizeof(float));

    private static long GatheredBytes(AttentionEvidenceSession evidence, int from) =>
        evidence.Snapshot().Skip(from).Where(e => e.Kind == AttentionEvidenceKind.PrefixGather).Sum(e => e.Bytes);

    [Fact]
    public void ADecodeStepCopiesOnlyItsNewRow()
    {
        using Pair pair = new();
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(4096);
        pair.Step(5);
        Assert.Equal(5 * RowBytes, GatheredBytes(evidence, 0));

        int before = evidence.Snapshot().Length;
        pair.Step(1);

        // Six rows are valid and they fit the eight-row buffer: only the new row leaves the pages.
        Assert.Equal(RowBytes, GatheredBytes(evidence, before));
    }

    [Fact]
    public void GrowingTheBufferCopiesTheOldRowsOnceAndNotFromThePages()
    {
        using Pair pair = new();
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(4096);
        pair.Step(8);
        int before = evidence.Snapshot().Length;

        // The ninth row needs a larger buffer: the eight held rows move across in one copy, then the new row is added.
        pair.Step(1);

        Assert.Equal(9 * RowBytes, GatheredBytes(evidence, before));
        Assert.Equal(2 * HeadDims.Length, evidence.Snapshot().Skip(before).Count(e => e.Kind == AttentionEvidenceKind.PrefixGather));
    }

    [Fact]
    public void ALongDecodeCopiesFarLessThanRebuildingEveryStep()
    {
        const int prompt = 16, steps = 200;
        using Pair pair = new();
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(1 << 16);
        pair.Step(prompt);
        for (int step = 0; step < steps; step++) pair.Step(1);

        // Rebuilding the prefix on every step copied every valid row each time.
        long rebuilt = RowBytes * (prompt + Enumerable.Range(1, steps).Sum(step => (long)(prompt + step)));
        // Now: each row once, plus the held rows whenever the buffer grows by a page.
        long growthCopies = Enumerable.Range(1, steps).Where(step => (prompt + step - 1) % PageSize == 0).Sum(step => (long)(prompt + step - 1));
        Assert.Equal(RowBytes * (prompt + steps + growthCopies), GatheredBytes(evidence, 0));
        Assert.True(GatheredBytes(evidence, 0) * 3 < rebuilt);
    }

    [Fact]
    public void ARollbackCopiesOnlyTheRowsWrittenAfterIt()
    {
        using Pair pair = new();
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(4096);
        pair.Step(6);
        pair.Step(1);
        pair.Truncate(5);
        int before = evidence.Snapshot().Length;

        pair.Step(2);

        Assert.Equal(2 * RowBytes, GatheredBytes(evidence, before));
    }

    [Fact]
    public void AKnownTokenBoundSizesThePrefixOnceAndNeverRegrowsIt()
    {
        const int prompt = 10, steps = 30;
        using Pair pair = new(expectedTokens: prompt + steps + 1);
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(4096);
        pair.Step(prompt);
        long rows = pair.Paged.KeyPrefix(0).Shape[2];
        for (int step = 0; step < steps; step++) pair.Step(1);

        // 41 tokens round up to 44 rows, inside the 256-row look-ahead, so the first buffer already holds the whole sequence.
        Assert.Equal(44, rows);
        for (int layer = 0; layer < HeadDims.Length; layer++)
        {
            Assert.Equal(rows, pair.Paged.KeyPrefix(layer).Shape[2]);
            Assert.Equal(rows, pair.Paged.ValuePrefix(layer).Shape[2]);
        }
        // Every row is copied once and nothing moves again.
        Assert.Equal(RowBytes * (prompt + steps), GatheredBytes(evidence, 0));
    }

    [Fact]
    public void ALongSequenceRegrowsItsPrefixAFewTimesWithinItsBound()
    {
        const int prompt = 4, steps = 700, bound = prompt + steps + 1;
        using Pair pair = new(expectedTokens: bound, maxPages: 256);
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(1 << 16);
        pair.Step(prompt);
        HashSet<long> capacities = [pair.Paged.KeyPrefix(0).Shape[2]];
        for (int step = 0; step < steps; step++)
        {
            pair.Step(1);
            capacities.Add(pair.Paged.KeyPrefix(0).Shape[2]);
        }

        // 256 rows ahead of the prompt, then 256 ahead of row 261, then the bound: three buffers for 704 rows.
        Assert.Equal([260L, 520L, 708L], capacities.Order());
        // Each row is written once, and the held rows move when the buffer grows: 260 of them, then 520.
        Assert.Equal(RowBytes * (prompt + steps + 260 + 520), GatheredBytes(evidence, 0));
    }

    [Fact]
    public void ASequenceThatOutgrowsItsBoundStillWorks()
    {
        using Pair pair = new(expectedTokens: 8);
        pair.Step(6);
        for (int step = 0; step < 20; step++) pair.Step(1);
        Assert.Equal(26, pair.Paged.CurrentLength);
    }
}
