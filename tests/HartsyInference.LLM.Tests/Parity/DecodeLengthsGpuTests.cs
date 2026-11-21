using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The CUDA and Vulkan decodes that take a key length per sequence and return the LSE (row
/// <c>flash.decode.lengths</c> on each), against the host reference for the same call. Tolerances were fixed before either
/// kernel first ran: 2e-5 on the output, and 2e-5 relative to the larger of 1 and the value on the LSE.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class DecodeLengthsGpuTests
{
    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    /// <summary><paramref name="source"/> stored as <paramref name="dtype"/>: F16, or BF16 rounded to nearest even.</summary>
    private static Tensor Stored(Tensor source, DType dtype)
    {
        if (dtype == DType.F32) return source;
        Tensor stored = new(source.Shape, dtype);
        ReadOnlySpan<float> values = source.AsReadOnlySpan<float>();
        Span<ushort> bits = stored.AsSpan<ushort>();
        for (int i = 0; i < values.Length; i++)
        {
            uint u = BitConverter.SingleToUInt32Bits(values[i]);
            bits[i] = dtype == DType.F16 ? BitConverter.HalfToUInt16Bits((Half)values[i]) : (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
        source.Dispose();
        return stored;
    }

    private sealed class Case : IDisposable
    {
        public readonly Tensor Query, Key, Value, Output, Lse, Expected, ExpectedLse;
        public readonly int Batch, Heads, Rows;

        public Case(int batch, int heads, int kvHeads, int dim, int rows, int capacity, DType? cache = null)
        {
            (Batch, Heads, Rows) = (batch, heads, rows);
            Query = Random(new TensorShape(batch, heads, rows, dim), 51);
            Key = Stored(Random(new TensorShape(batch, kvHeads, capacity, dim), 52), cache ?? DType.F32);
            Value = Stored(Random(Key.Shape, 53), cache ?? DType.F32);
            Output = new Tensor(new TensorShape(batch, rows, heads * dim), DType.F32);
            Expected = new Tensor(Output.Shape, DType.F32);
            Lse = new Tensor(new TensorShape(batch, heads, rows), DType.F32);
            ExpectedLse = new Tensor(Lse.Shape, DType.F32);
        }

        public AttentionCall Call(int[] lengths, bool causal, bool lse) =>
            AttentionCall.FromDecodeWithLengths(Output, Query, Key, Value, lengths, causal, 1f / MathF.Sqrt(Query.Shape[3]), lse ? Lse : null);

        /// <summary>Runs the call on the backend and on the host reference, and compares.</summary>
        public AttentionEvidenceEvent[] RunAndCompare(IBackend backend, int[] lengths, bool causal, bool lse)
        {
            AttentionCall call = Call(lengths, causal, lse);
            AttentionCall reference = call with { Output = call.Output with { Storage = Expected }, Lse = lse ? ExpectedLse : null };
            AttentionHostReference.Run(in reference);
            Output.AsSpan<float>().Fill(-7f);
            Lse.AsSpan<float>().Fill(-7f);

            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
            backend.Attention(in call);
            backend.Sync();
            AttentionEvidenceEvent[] events = evidence.Snapshot();

            ReadOnlySpan<float> expected = Expected.AsReadOnlySpan<float>(), actual = Output.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-5f, $"output {i}: {actual[i]}, expected {expected[i]}");
            if (!lse) return events;
            ReadOnlySpan<float> expectedLse = ExpectedLse.AsReadOnlySpan<float>(), actualLse = Lse.AsReadOnlySpan<float>();
            for (int i = 0; i < expectedLse.Length; i++)
            {
                if (float.IsNegativeInfinity(expectedLse[i])) Assert.True(float.IsNegativeInfinity(actualLse[i]), $"LSE {i}: {actualLse[i]}, expected negative infinity");
                else Assert.True(Math.Abs(expectedLse[i] - actualLse[i]) <= 2e-5f * Math.Max(1f, Math.Abs(expectedLse[i])), $"LSE {i}: {actualLse[i]}, expected {expectedLse[i]}");
            }
            return events;
        }

        public void Dispose()
        {
            foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[Query, Key, Value, Output, Lse, Expected, ExpectedLse]) tensor.Dispose();
        }
    }

    public static TheoryData<int, int, int, int, int, int, int[], bool> Shapes => new()
    {
        // batch, heads, kv heads, head dim, step rows, capacity, lengths, causal
        { 2, 8, 2, 64, 1, 96, new[] { 80, 70 }, true },                // plain decode, two sequences
        { 3, 8, 2, 128, 1, 300, new[] { 300, 1, 0 }, true },           // full, one key, empty; several chunks
        { 2, 4, 4, 64, 5, 200, new[] { 150, 3 }, true },               // a staircase, and a sequence shorter than the step
        { 2, 8, 1, 64, 4, 130, new[] { 0, 129 }, false },              // prefix-only, one sequence empty, group of 8
        { 1, 16, 1, 128, 3, 2000, new[] { 1999 }, true },              // group of 16 over a long cache
        { 2, 4, 2, 64, 2, 100, new[] { 1000, -5 }, true },             // lengths outside the cache are clamped
        { 1, 8, 8, 64, 32, 64, new[] { 40 }, false },                  // the most rows a verify step holds
        { 4, 4, 2, 64, 1, 256, new[] { 256, 128, 64, 192 }, true },    // four chunks of 64: each sequence ends on a chunk boundary
    };

    [CudaTheory]
    [MemberData(nameof(Shapes))]
    public void TheCudaKernelMatchesTheHostReferenceWithAndWithoutTheLse(int batch, int heads, int kvHeads, int dim, int rows, int capacity,
        int[] lengths, bool causal)
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        MatchesTheHostReference(backend, batch, heads, kvHeads, dim, rows, capacity, lengths, causal);
    }

    /// <summary>An F16 or BF16 cache is widened as it is read: the result is the host reference's over the same stored
    /// values, to the F32 tolerance.</summary>
    [CudaTheory]
    [InlineData("F16")]
    [InlineData("BF16")]
    public void TheCudaKernelReadsAHalfTypeCache(string type)
    {
        DType cache = type == "F16" ? DType.F16 : DType.BF16;
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        foreach ((int batch, int heads, int kvHeads, int dim, int rows, int capacity, int[] lengths, bool causal) in
            ((int, int, int, int, int, int, int[], bool)[])[(2, 8, 2, 64, 1, 96, [80, 70], true), (2, 4, 4, 128, 5, 200, [150, 3], true),
                (2, 8, 1, 64, 4, 130, [0, 129], false), (1, 16, 1, 128, 3, 700, [699], true),
                // lengths outside the cache are clamped; each sequence ends on a chunk boundary
                (2, 4, 2, 64, 2, 100, [1000, -5], true), (4, 4, 2, 64, 1, 256, [256, 128, 64, 192], true)])
        {
            using Case shape = new(batch, heads, kvHeads, dim, rows, capacity, cache);
            foreach (bool lse in (ReadOnlySpan<bool>)[true, false])
            {
                AttentionEvidenceEvent[] events = shape.RunAndCompare(backend, lengths, causal, lse);
                Assert.Equal("flash.decode.lengths", Assert.Single(events, e => e.Kind == AttentionEvidenceKind.Execution).Implementation);
                Assert.DoesNotContain(events, e => e.Kind == AttentionEvidenceKind.Cast);
            }
        }
    }

    /// <summary>Every shape the Vulkan split shader is built for: its group limit is 8 query heads a key head.</summary>
    [VulkanFact]
    public void TheVulkanKernelMatchesTheHostReferenceWithAndWithoutTheLse()
    {
        using VulkanBackend backend = new();
        int ran = 0;
        foreach (object[] shape in Shapes)
        {
            int heads = (int)shape[1], kvHeads = (int)shape[2];
            if (heads / kvHeads > 8) continue;
            MatchesTheHostReference(backend, (int)shape[0], heads, kvHeads, (int)shape[3], (int)shape[4], (int)shape[5], (int[])shape[6], (bool)shape[7]);
            ran++;
        }
        Assert.Equal(7, ran);
    }

    /// <summary>The row's head dims run from one subgroup width to eight, and the shapes above reach two and four where a
    /// subgroup is 32 lanes wide. This runs one, three, six and eight widths. At eight (head dim 256 at 32 lanes): full,
    /// one-key and empty sequences over several chunks with a group of 4; a five-row staircase with a sequence shorter
    /// than the step, one query head a key head; and a group of 8 over 899 of 900 slots, which leaves a one-slot tail.
    /// Half a width and nine are refused whole, and so is anything past head dim 256 on a wider subgroup.</summary>
    [VulkanFact]
    public void TheVulkanKernelRunsTheHeadDimsItsRowDeclaresAndRefusesTheOthers()
    {
        using VulkanBackend backend = new();
        int lanes = (int)backend.Vk.SubgroupSize;
        MatchesTheHostReference(backend, 2, 8, 2, lanes, 2, 150, [150, 41], causal: true);
        MatchesTheHostReference(backend, 2, 6, 2, 3 * lanes, 1, 200, [199, 64], causal: false);
        MatchesTheHostReference(backend, 2, 4, 2, 6 * lanes, 1, 180, [180, 9], causal: true);
        MatchesTheHostReference(backend, 3, 8, 2, 8 * lanes, 1, 300, [300, 1, 0], causal: true);
        MatchesTheHostReference(backend, 2, 4, 4, 8 * lanes, 5, 200, [150, 3], causal: true);
        MatchesTheHostReference(backend, 1, 8, 1, 8 * lanes, 3, 900, [899], causal: false);
        foreach (int dim in (ReadOnlySpan<int>)[lanes / 2, 9 * lanes, 256 + lanes])
        {
            using Case outside = new(1, 4, 2, dim, 1, 96);
            outside.Output.AsSpan<float>().Fill(-7f);
            AttentionCall call = outside.Call([80], causal: true, lse: false);
            Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in call));
            backend.Sync();
            Assert.Equal(-1, outside.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        }
    }

    private static void MatchesTheHostReference(IBackend backend, int batch, int heads, int kvHeads, int dim, int rows, int capacity,
        int[] lengths, bool causal)
    {
        using Case shape = new(batch, heads, kvHeads, dim, rows, capacity);
        foreach (bool lse in (ReadOnlySpan<bool>)[true, false])
        {
            AttentionEvidenceEvent[] events = shape.RunAndCompare(backend, lengths, causal, lse);
            AttentionEvidenceEvent execution = Assert.Single(events, e => e.Kind == AttentionEvidenceKind.Execution);
            Assert.Equal("flash.decode.lengths", execution.Implementation);
            Assert.Equal(batch * sizeof(int), Assert.Single(events, e => e.Kind == AttentionEvidenceKind.StagingCopy).Bytes);
            Assert.DoesNotContain(events, e => e.Kind is AttentionEvidenceKind.Cast or AttentionEvidenceKind.Repeat or AttentionEvidenceKind.PrefixGather);
            if (!lse) Assert.Equal(-1, shape.Lse.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        }
        AttentionCall call = shape.Call(lengths, causal, lse: true);
        AttentionPlan plan = backend.PlanAttention(in call);
        Assert.Equal("flash.decode.lengths", plan.RowId);
        Assert.Equal(0, plan.Preparation!.Value.Total);
    }

    [CudaFact]
    public void OnCudaRowsPastASequencesLengthAreNeverRead()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        RowsPastASequencesLengthAreNeverRead(backend);
    }

    [CudaTheory]
    [InlineData("F16")]
    [InlineData("BF16")]
    public void OnCudaRowsPastASequencesLengthOfAHalfTypeCacheAreNeverRead(string type)
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        RowsPastASequencesLengthAreNeverRead(backend, type == "F16" ? DType.F16 : DType.BF16);
    }

    [VulkanFact]
    public void OnVulkanRowsPastASequencesLengthAreNeverRead()
    {
        using VulkanBackend backend = new();
        RowsPastASequencesLengthAreNeverRead(backend);
    }

    private static void RowsPastASequencesLengthAreNeverRead(IBackend backend, DType? cache = null)
    {
        int[] lengths = [130, 7];
        using Case shape = new(2, 8, 2, 64, 3, 200, cache);
        for (int b = 0; b < 2; b++)
            for (int kh = 0; kh < 2; kh++)
            {
                int tail = ((b * 2 + kh) * 200 + lengths[b]) * 64, count = (200 - lengths[b]) * 64;
                foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[shape.Key, shape.Value])
                {
                    if (tensor.DType == DType.F32) tensor.AsSpan<float>().Slice(tail, count).Fill(float.NaN);
                    // A quiet NaN in either half type: every exponent bit and the top fraction bit.
                    else tensor.AsSpan<ushort>().Slice(tail, count).Fill(tensor.DType == DType.F16 ? (ushort)0x7E00 : (ushort)0x7FC0);
                }
            }
        foreach (bool causal in (ReadOnlySpan<bool>)[true, false])
        {
            shape.RunAndCompare(backend, lengths, causal, lse: true);
            Assert.DoesNotContain(shape.Output.AsReadOnlySpan<float>().ToArray(), float.IsNaN);
            Assert.DoesNotContain(shape.Lse.AsReadOnlySpan<float>().ToArray(), float.IsNaN);
        }
    }

    /// <summary>The Vulkan row's limits: a group above 8, an F16 cache and a strided view are refused whole.</summary>
    [VulkanFact]
    public void OnVulkanACallOutsideTheKernelIsRefusedAndWritesNothing()
    {
        using VulkanBackend backend = new();
        using Case wide = new(1, 16, 1, 64, 1, 96);
        wide.Output.AsSpan<float>().Fill(-7f);
        AttentionCall call = wide.Call([80], causal: true, lse: false);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in call));
        using Case shape = new(2, 8, 2, 64, 1, 96);
        using Tensor halfKey = new(shape.Key.Shape, DType.F16);
        using Tensor halfValue = new(shape.Key.Shape, DType.F16);
        shape.Output.AsSpan<float>().Fill(-7f);
        AttentionCall half = AttentionCall.FromDecodeWithLengths(shape.Output, shape.Query, halfKey, halfValue, [80, 70], true, 0.125f);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in half));
        using Tensor longer = Random(new TensorShape(2, 8, 2, 64), 54);
        AttentionCall whole = shape.Call([80, 70], causal: true, lse: false);
        AttentionCall view = whole with { Query = new AttentionOperand(longer, AttentionLayout.HeadMajor, 2, 8, 1, 64, 64, 8 * 2 * 64, 2 * 64, 64) };
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in view));
        backend.Sync();
        Assert.Equal(-1, wide.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        Assert.Equal(-1, shape.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
    }

    /// <summary>The row takes only calls with sequence lengths: a call it is not built for is refused whole.</summary>
    [CudaFact]
    public void OnCudaACallOutsideTheKernelIsRefusedAndWritesNothing()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        // Head dim 32 has no decode kernel.
        using Case narrow = new(2, 4, 2, 32, 1, 96);
        narrow.Output.AsSpan<float>().Fill(-7f);
        AttentionCall call = narrow.Call([80, 70], causal: true, lse: false);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in call));
        // Keys and values of two types are not this kernel's: it reads one cache type.
        using Case shape = new(2, 8, 2, 64, 1, 96);
        using Tensor halfKey = new(shape.Key.Shape, DType.F16);
        shape.Output.AsSpan<float>().Fill(-7f);
        AttentionCall mixed = AttentionCall.FromDecodeWithLengths(shape.Output, shape.Query, halfKey, shape.Value, [80, 70], true, 0.125f);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in mixed));
        // Neither is an F16 query.
        using Tensor halfQuery = new(shape.Query.Shape, DType.F16);
        AttentionCall halfQ = AttentionCall.FromDecodeWithLengths(shape.Output, halfQuery, shape.Key, shape.Value, [80, 70], true, 0.125f);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in halfQ));
        // The row reads whole tensors: a view of the queries (rows 1 and up of a longer tensor) is not its call.
        using Tensor longer = Random(new TensorShape(2, 8, 2, 64), 54);
        AttentionCall whole = shape.Call([80, 70], causal: true, lse: false);
        AttentionCall view = whole with { Query = new AttentionOperand(longer, AttentionLayout.HeadMajor, 2, 8, 1, 64, 64, 8 * 2 * 64, 2 * 64, 64) };
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in view));
        backend.Sync();
        Assert.Equal(-1, narrow.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        Assert.Equal(-1, shape.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
    }
}
