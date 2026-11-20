using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A decode step over a paged cache reads the pages through the page table on CUDA. Every call here runs twice, over
/// a paged cache and over a contiguous <see cref="FixedKvCache"/> fed the same rows, and the two outputs are compared bit for
/// bit. Two sequences share the pool so neither page table is the pages in order, which is the one table a kernel that
/// ignored it would still read correctly.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class PagedAttentionDecodeCudaTests
{
    private const int Layers = 2, PageSize = 16, MaxPages = 48, Capacity = PageSize * MaxPages;

    /// <summary>One sequence held twice: in the shared page pool and in its own contiguous cache.</summary>
    private sealed class Sequence(PagedKvPool pool, int kvHeads, int headDim, int expectedTokens) : IDisposable
    {
        public readonly PagedKvCache Paged = new(pool, expectedTokens);
        public readonly FixedKvCache Fixed = new(Layers, 1, kvHeads, headDim, Capacity, DType.F32);

        public void Dispose()
        {
            Paged.Dispose();
            Fixed.Dispose();
        }
    }

    private sealed class Rig : IDisposable
    {
        public readonly CudaBackend Backend = new(0, CudaPtxDirectory.Resolve());
        public readonly PagedKvPool Pool;
        public readonly AttentionLayer Layer;
        public readonly Sequence A, B;
        private readonly int _heads, _kvHeads, _headDim;
        private readonly Random _random = new(4242);
        private readonly string _release;

        public Rig(int heads, int kvHeads, int headDim, string release = "none")
        {
            _heads = heads;
            _kvHeads = kvHeads;
            _headDim = headDim;
            _release = release;
            Pool = new PagedKvPool(Layers, kvHeads, headDim, PageSize, MaxPages);
            Layer = new AttentionLayer(new AttentionLayerConfig { Heads = heads, KvHeads = kvHeads, HeadDim = headDim });
            A = new Sequence(Pool, kvHeads, headDim, expectedTokens: 0);
            B = new Sequence(Pool, kvHeads, headDim, expectedTokens: 200);
        }

        /// <summary>One model step of <paramref name="tokens"/> new rows: each layer appends and attends, over both caches,
        /// and the two outputs must be the same bits.</summary>
        public void Step(Sequence sequence, int tokens, float softcap = 0f, int window = 0, Tensor? sinks = null)
        {
            int length = sequence.Paged.CurrentLength;
            for (int layer = 0; layer < Layers; layer++)
            {
                using Tensor q = Rows(_heads, tokens), k = Rows(_kvHeads, tokens), v = Rows(_kvHeads, tokens);
                KvAttentionCall call = new()
                {
                    KvLength = length + tokens, QueryOffset = length, Softcap = softcap, SlidingWindow = window, Sinks = sinks,
                };
                using Tensor paged = Layer.AttendOverKvHeadMajor(Backend, q, k, v, sequence.Paged, layer, layer, in call, OperandOwnership.None);
                using Tensor contiguous = Layer.AttendOverKvHeadMajor(Backend, q, k, v, sequence.Fixed, layer, layer, in call, OperandOwnership.None);
                Assert.True(paged.AsReadOnlySpan<float>().SequenceEqual(contiguous.AsReadOnlySpan<float>()),
                    $"layer {layer}, {tokens} row(s) at {length}: the paged output differs from the contiguous one.");
            }
            sequence.Paged.AdvanceLength(tokens);
            sequence.Fixed.AdvanceLength(tokens);
            if (_release == "offload") Backend.OffloadActivations(long.MaxValue);
        }

        public void Truncate(Sequence sequence, int length)
        {
            sequence.Paged.Truncate(length);
            sequence.Fixed.Truncate(length);
        }

        /// <summary>The contiguous copy the paged cache rebuilds from its pages holds the rows the contiguous cache holds.</summary>
        public void AssertGatheredRowsMatch(Sequence sequence)
        {
            for (int layer = 0; layer < Layers; layer++)
            {
                AssertRows(sequence.Paged.KeyPrefix(layer), sequence.Fixed.KeyPrefix(layer), sequence.Paged.CurrentLength, $"layer {layer} keys");
                AssertRows(sequence.Paged.ValuePrefix(layer), sequence.Fixed.ValuePrefix(layer), sequence.Paged.CurrentLength, $"layer {layer} values");
            }
        }

        private void AssertRows(Tensor gathered, Tensor contiguous, int rows, string what)
        {
            ReadOnlySpan<float> actual = gathered.AsReadOnlySpan<float>(), expected = contiguous.AsReadOnlySpan<float>();
            int gatheredRows = (int)gathered.Shape[2], contiguousRows = (int)contiguous.Shape[2];
            for (int head = 0; head < _kvHeads; head++)
                Assert.True(actual.Slice(head * gatheredRows * _headDim, rows * _headDim).SequenceEqual(expected.Slice(head * contiguousRows * _headDim, rows * _headDim)),
                    $"{what}: head {head} differs within its {rows} rows.");
        }

        private Tensor Rows(int heads, int tokens)
        {
            Tensor tensor = new(new TensorShape(1, heads, tokens, _headDim), DType.F32);
            Span<float> data = tensor.AsSpan<float>();
            for (int i = 0; i < data.Length; i++) data[i] = (float)(_random.NextDouble() * 2 - 1);
            return tensor;
        }

        public void Dispose()
        {
            A.Dispose();
            B.Dispose();
            Pool.Dispose();
            Backend.Dispose();
        }
    }

    private static bool Scattered(IReadOnlyList<int> table)
    {
        for (int i = 0; i < table.Count; i++)
            if (table[i] != i) return true;
        return false;
    }

    private static int Count(AttentionEvidenceEvent[] events, string implementation) =>
        events.Count(e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == implementation);

    [CudaTheory]
    [InlineData(8, 2, 64, "none", 0f, 0)]
    [InlineData(8, 2, 64, "offload", 0f, 0)]
    [InlineData(4, 4, 128, "none", 0f, 0)]
    [InlineData(6, 2, 128, "none", 0f, 0)]
    [InlineData(16, 1, 64, "none", 0f, 0)]
    [InlineData(8, 2, 128, "none", 30f, 0)]
    [InlineData(8, 2, 64, "none", 0f, 48)]
    [InlineData(8, 2, 128, "offload", 30f, 80)]
    public void DecodeReadsThePagesAndMatchesAContiguousCacheBitForBit(int heads, int kvHeads, int headDim, string release, float softcap, int window)
    {
        using Rig rig = new(heads, kvHeads, headDim, release);
        // Prefills are multi-row calls: gathered, as before.
        rig.Step(rig.A, 70, softcap, window);
        rig.Step(rig.B, 65, softcap, window);

        const int rounds = 40;
        AttentionEvidenceEvent[] events;
        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(8192))
        {
            for (int round = 0; round < rounds; round++)
            {
                rig.Step(rig.A, 1, softcap, window);
                rig.Step(rig.B, 1, softcap, window);
            }
            rig.Backend.Sync();
            events = evidence.Snapshot();
            Assert.Equal(0, evidence.Dropped);
        }
        Assert.True(Scattered(rig.A.Paged.PageTable) && Scattered(rig.B.Paged.PageTable), "The sequences' pages were meant to interleave.");
        // Every decode call over the paged cache ran the paged kernel, its twin ran the contiguous grouped decode, and no
        // row was copied into a contiguous buffer for it.
        Assert.Equal(2 * rounds * Layers, Count(events, "flash.decode.paged"));
        Assert.Equal(2 * rounds * Layers, events.Count(e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "flash.decode" && e.Branch == "F.4"));
        Assert.DoesNotContain(events, e => e.Kind == AttentionEvidenceKind.PrefixGather);

        // A rollback frees pages the other sequence then takes; both keep decoding on reused pages.
        rig.Truncate(rig.A, 75);
        for (int round = 0; round < 24; round++)
        {
            rig.Step(rig.B, 1, softcap, window);
            rig.Step(rig.A, 1, softcap, window);
        }
        Assert.Empty(rig.A.Paged.PageTable.Intersect(rig.B.Paged.PageTable));
        // A multi-row call after decode is gathered again: the contiguous copy is rebuilt from the pages.
        rig.Step(rig.A, 20, softcap, window);
        rig.Step(rig.A, 1, softcap, window);
        rig.AssertGatheredRowsMatch(rig.A);
        rig.AssertGatheredRowsMatch(rig.B);
    }

    [CudaFact]
    public void ACallTheKernelDoesNotTakeIsGatheredAndStillMatches()
    {
        const int heads = 8, kvHeads = 2, headDim = 64;
        using Rig rig = new(heads, kvHeads, headDim);
        using Tensor sinks = new(new TensorShape(heads), DType.F32);
        sinks.AsSpan<float>().Fill(0.25f);
        rig.Step(rig.A, 70, sinks: sinks);
        rig.Step(rig.B, 70, sinks: sinks);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(4096);
        for (int round = 0; round < 6; round++)
        {
            rig.Step(rig.A, 1, sinks: sinks);
            rig.Step(rig.B, 1, sinks: sinks);
        }
        rig.Backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();
        // Sinks are outside the paged kernel: every call reads the contiguous copy, which each step extends by its row.
        Assert.Equal(0, Count(events, "flash.decode.paged"));
        Assert.Equal(2 * 6 * Layers * 2, events.Count(e => e.Kind == AttentionEvidenceKind.PrefixGather));
    }

    [CudaFact]
    public void AShortSequenceIsGatheredUntilItHasEnoughKeysForTheKernel()
    {
        const int heads = 8, kvHeads = 2, headDim = 64;
        using Rig rig = new(heads, kvHeads, headDim);
        rig.Step(rig.A, 20);
        rig.Step(rig.B, 30);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8192);
        // A decodes from 21 valid keys to 90: the grouped decode kernels start at 64.
        for (int round = 0; round < 70; round++) rig.Step(rig.A, 1);
        rig.Backend.Sync();
        AttentionEvidenceEvent[] events = evidence.Snapshot();
        Assert.Equal(0, evidence.Dropped);
        Assert.Equal((90 - 63) * Layers, Count(events, "flash.decode.paged"));
        Assert.All(events.Where(e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "flash.decode.paged"),
            e => Assert.True(e.ValidKeys >= 64));
        rig.AssertGatheredRowsMatch(rig.A);
    }
}
