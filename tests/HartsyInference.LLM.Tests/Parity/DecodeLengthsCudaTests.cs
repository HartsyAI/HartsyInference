using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The CUDA decode that takes a key length per sequence and returns the LSE (row <c>flash.decode.lengths</c>),
/// against the host reference for the same call. Tolerances were fixed before the kernel first ran: 2e-5 on the output,
/// and 2e-5 relative to the larger of 1 and the value on the LSE.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class DecodeLengthsCudaTests
{
    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    private sealed class Case : IDisposable
    {
        public readonly Tensor Query, Key, Value, Output, Lse, Expected, ExpectedLse;
        public readonly int Batch, Heads, Rows;

        public Case(int batch, int heads, int kvHeads, int dim, int rows, int capacity)
        {
            (Batch, Heads, Rows) = (batch, heads, rows);
            Query = Random(new TensorShape(batch, heads, rows, dim), 51);
            Key = Random(new TensorShape(batch, kvHeads, capacity, dim), 52);
            Value = Random(Key.Shape, 53);
            Output = new Tensor(new TensorShape(batch, rows, heads * dim), DType.F32);
            Expected = new Tensor(Output.Shape, DType.F32);
            Lse = new Tensor(new TensorShape(batch, heads, rows), DType.F32);
            ExpectedLse = new Tensor(Lse.Shape, DType.F32);
        }

        public AttentionCall Call(int[] lengths, bool causal, bool lse) =>
            AttentionCall.FromDecodeWithLengths(Output, Query, Key, Value, lengths, causal, 1f / MathF.Sqrt(Query.Shape[3]), lse ? Lse : null);

        /// <summary>Runs the call on the backend and on the host reference, and compares.</summary>
        public AttentionEvidenceEvent[] RunAndCompare(CudaBackend backend, int[] lengths, bool causal, bool lse)
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
    };

    [CudaTheory]
    [MemberData(nameof(Shapes))]
    public void TheKernelMatchesTheHostReferenceWithAndWithoutTheLse(int batch, int heads, int kvHeads, int dim, int rows, int capacity,
        int[] lengths, bool causal)
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
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
    public void RowsPastASequencesLengthAreNeverRead()
    {
        int[] lengths = [130, 7];
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Case shape = new(2, 8, 2, 64, 3, 200);
        for (int b = 0; b < 2; b++)
            for (int kh = 0; kh < 2; kh++)
            {
                int tail = ((b * 2 + kh) * 200 + lengths[b]) * 64, count = (200 - lengths[b]) * 64;
                shape.Key.AsSpan<float>().Slice(tail, count).Fill(float.NaN);
                shape.Value.AsSpan<float>().Slice(tail, count).Fill(float.NaN);
            }
        foreach (bool causal in (ReadOnlySpan<bool>)[true, false])
        {
            shape.RunAndCompare(backend, lengths, causal, lse: true);
            Assert.DoesNotContain(shape.Output.AsReadOnlySpan<float>().ToArray(), float.IsNaN);
            Assert.DoesNotContain(shape.Lse.AsReadOnlySpan<float>().ToArray(), float.IsNaN);
        }
    }

    /// <summary>The row takes only calls with sequence lengths: a call it is not built for is refused whole.</summary>
    [CudaFact]
    public void ACallOutsideTheKernelIsRefusedAndWritesNothing()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        // Head dim 32 has no decode kernel.
        using Case narrow = new(2, 4, 2, 32, 1, 96);
        narrow.Output.AsSpan<float>().Fill(-7f);
        AttentionCall call = narrow.Call([80, 70], causal: true, lse: false);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in call));
        // An F16 cache is not this kernel's.
        using Case shape = new(2, 8, 2, 64, 1, 96);
        using Tensor halfKey = new(shape.Key.Shape, DType.F16);
        using Tensor halfValue = new(shape.Key.Shape, DType.F16);
        shape.Output.AsSpan<float>().Fill(-7f);
        AttentionCall half = AttentionCall.FromDecodeWithLengths(shape.Output, shape.Query, halfKey, halfValue, [80, 70], true, 0.125f);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in half));
        backend.Sync();
        Assert.Equal(-1, narrow.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        Assert.Equal(-1, shape.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
    }
}
