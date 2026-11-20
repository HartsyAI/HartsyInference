using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Transformer;

/// <summary>The paged cache presents the same keys and values as a contiguous <see cref="FixedKvCache"/> fed the same
/// appends, through prefill, decode, rollback, reset and a rewrite of the same position.</summary>
public sealed class PagedKvCacheTests
{
    private const int KvHeads = 2, PageSize = 4, MaxPages = 64, MaxTokens = PageSize * MaxPages;
    private static readonly int[] HeadDims = [4, 6, 4];

    /// <summary>A paged cache and a contiguous reference driven with identical data.</summary>
    private sealed class Pair : IDisposable
    {
        public readonly CpuBackend Backend = new();
        public readonly PagedKvPool Pool = new(HeadDims.Length, KvHeads, HeadDims, PageSize, MaxPages);
        public readonly PagedKvCache Paged;
        public readonly FixedKvCache Fixed = new(HeadDims.Length, 1, KvHeads, HeadDims, MaxTokens, DType.F32);
        private readonly Random _random = new(1234);

        public Pair() => Paged = new PagedKvCache(Pool);

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
}
