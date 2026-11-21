using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>General Sol-Attn on CUDA against <see cref="SolAttentionReference"/>. Two things were fixed before the kernels
/// first ran: the routing (which key blocks each query block reads exactly) must equal the reference's entry for entry,
/// because the routing stages are written to reproduce its scores bit for bit; and the output must agree to 1e-4
/// relative to the larger of 1 and the value.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class SolAttentionCudaTests
{
    /// <summary>Operands with structure for the routing to find: noise, plus a component shared by every third block.</summary>
    private sealed class Operands : IDisposable
    {
        public readonly Tensor Query, Key, Value, Output, Expected;
        public readonly int Batch, Tokens, Heads, Dim;

        public Operands(int batch, int tokens, int heads, int dim, int seed)
        {
            (Batch, Tokens, Heads, Dim) = (batch, tokens, heads, dim);
            TensorShape shape = new(batch, tokens, heads, dim);
            Random random = new(seed);
            float[] basis = new float[3 * dim];
            for (int i = 0; i < basis.Length; i++) basis[i] = (random.NextSingle() * 2f - 1f) * 2f;
            Query = new Tensor(shape, DType.F32);
            Key = new Tensor(shape, DType.F32);
            Value = new Tensor(shape, DType.F32);
            Output = new Tensor(shape, DType.F32);
            Expected = new Tensor(shape, DType.F32);
            Span<float> q = Query.AsSpan<float>(), k = Key.AsSpan<float>(), v = Value.AsSpan<float>();
            for (int i = 0; i < q.Length; i++)
            {
                int token = i / (heads * dim) % tokens, channel = i % dim;
                float shared = basis[token / 64 % 3 * dim + channel];
                q[i] = random.NextSingle() * 2f - 1f + shared;
                k[i] = random.NextSingle() * 2f - 1f + shared;
                v[i] = random.NextSingle() * 2f - 1f;
            }
        }

        public int Blocks => SolAttentionReference.Blocks(Tokens);

        /// <summary>Runs the reference and the backend with <paramref name="options"/>; the routing must be identical and the
        /// output within tolerance. Returns how many (query block, key block) pairs were read exactly, of how many.</summary>
        public (int Read, int Pairs) RunAndCompare(IBackend backend, SolAttentionOptions options)
        {
            int pairs = Batch * Heads * Blocks * Blocks;
            bool[] expectedRouting = new bool[pairs], routing = new bool[pairs];
            SolAttentionReference.Run(Expected, Query, Key, Value, options, expectedRouting);
            Output.AsSpan<float>().Fill(-7f);
            backend.SolAttention(Output, Query, Key, Value, options, routing);
            backend.Sync();
            for (int i = 0; i < pairs; i++)
                Assert.True(expectedRouting[i] == routing[i], $"routing differs at pair {i} (query block {i / Blocks % Blocks}, key block {i % Blocks}): reference {expectedRouting[i]}");
            ReadOnlySpan<float> expected = Expected.AsReadOnlySpan<float>(), actual = Output.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])), $"output {i}: {actual[i]}, expected {expected[i]}");
            return (routing.Count(static read => read), pairs);
        }

        public Tensor Bias(bool perBatch, bool mask, int seed)
        {
            Tensor bias = perBatch ? new Tensor(new TensorShape(Batch, Tokens), DType.F32) : new Tensor(new TensorShape(Tokens), DType.F32);
            Random random = new(seed);
            Span<float> values = bias.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
            {
                float draw = random.NextSingle() * 2f - 1f;
                values[i] = mask ? (draw > -0.6f ? 0f : float.NegativeInfinity) : draw * 2f;
            }
            return bias;
        }

        public void Dispose()
        {
            foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[Query, Key, Value, Output, Expected]) tensor.Dispose();
        }
    }

    private static CudaBackend Backend()
    {
        CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        Assert.True(backend.HasSolAttention);
        return backend;
    }

    [CudaTheory]
    [InlineData(1f)]
    [InlineData(0.2f)]
    [InlineData(3f)]
    [InlineData(-1e6f)]
    public void TauRoutingOverARaggedLastBlockMatchesTheReference(float tau)
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(1, 64 * 5 + 17, 2, 128, 91);
        (int read, int pairs) = operands.RunAndCompare(backend, new SolAttentionOptions { Tau = tau });
        // A threshold far below every score reads everything; the others read some blocks and pool the rest.
        if (tau < -1e5f) Assert.Equal(pairs, read);
        else Assert.InRange(read, 1, pairs - 1);
    }

    [CudaTheory]
    [InlineData(0.4, true)]
    [InlineData(0.4, false)]
    [InlineData(0.1, true)]
    public void TopKRoutingMatchesTheReferenceWithAndWithoutTheTail(double ratio, bool tail)
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(1, 64 * 7, 2, 128, 92);
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = ratio, Tail = tail });
    }

    [CudaFact]
    public void SinkBlocksAndSinkQueriesMatchTheReference()
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(1, 64 * 6 + 5, 2, 128, 93);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 2f, SinkBlocks = (1, 3), SinkQueries = (0, 1) });
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.3, SinkBlocks = (1, 3) });
        // Sink ranges past the last block are clamped, as in the reference.
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 2f, SinkBlocks = (5, 99), SinkQueries = (6, 40) });
    }

    [CudaFact]
    public void AKeyBiasAndAKeyMaskMatchTheReference()
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(2, 64 * 4 + 30, 2, 128, 94);
        using Tensor perBatch = operands.Bias(perBatch: true, mask: false, 1);
        using Tensor shared = operands.Bias(perBatch: false, mask: false, 2);
        using Tensor mask = operands.Bias(perBatch: false, mask: true, 3);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1.5f, SinkBlocks = (0, 2), KeyBias = perBatch });
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1.5f, SinkBlocks = (0, 2), KeyBias = shared });
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1.5f, SinkBlocks = (0, 2), KeyBias = mask });
    }

    [CudaFact]
    public void BlockLengthsWithDeadRowsMatchTheReference()
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(1, 64 * 5, 2, 128, 95);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1f, BlockLengths = [64, 40, 64, 1, 17] });
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.5, Tail = false, BlockLengths = [10, 64, 64, 64, 33] });
    }

    [CudaTheory]
    [InlineData(2, 3, 128, 64 * 3 + 1)]
    [InlineData(1, 4, 64, 64 * 6)]
    [InlineData(1, 2, 256, 64 * 4 + 9)]
    [InlineData(1, 5, 40, 64 * 5)]
    [InlineData(1, 2, 128, 50)]
    public void OtherShapesAndACustomScaleMatchTheReference(int batch, int heads, int dim, int tokens)
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(batch, tokens, heads, dim, 96);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1f, Scale = 0.07f });
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.34 });
    }

    [CudaFact]
    public void WithoutTheRoutingReportTheOutputIsTheSame()
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(1, 64 * 4, 2, 128, 97);
        SolAttentionOptions options = new() { Tau = 1f };
        operands.RunAndCompare(backend, options);
        float[] reported = operands.Output.AsReadOnlySpan<float>().ToArray();
        backend.SolAttention(operands.Output, operands.Query, operands.Key, operands.Value, options);
        backend.Sync();
        Assert.True(reported.AsSpan().SequenceEqual(operands.Output.AsReadOnlySpan<float>()));
    }

    [CudaFact]
    public void WhatTheKernelsDoNotTakeIsRefusedAndWritesNothing()
    {
        using CudaBackend backend = Backend();
        using Operands operands = new(1, 64 * 3, 2, 128, 98);
        operands.Output.AsSpan<float>().Fill(-7f);
        using Tensor gate = new(operands.Query.Shape, DType.F32);
        Assert.Throws<NotSupportedException>(() =>
            backend.SolAttention(operands.Output, operands.Query, operands.Key, operands.Value, new SolAttentionOptions { CoarseGate = gate }));
        using Operands wide = new(1, 64 * 2, 1, 320, 99);
        wide.Output.AsSpan<float>().Fill(-7f);
        Assert.Throws<NotSupportedException>(() => backend.SolAttention(wide.Output, wide.Query, wide.Key, wide.Value, new SolAttentionOptions()));
        Assert.Throws<ArgumentException>(() =>
            backend.SolAttention(operands.Output, operands.Query, operands.Key, operands.Value, new SolAttentionOptions { TopKRatio = 1 }));
        Assert.Throws<ArgumentException>(() =>
            backend.SolAttention(operands.Output, operands.Query, operands.Key, operands.Value, new SolAttentionOptions(), new bool[3]));
        backend.Sync();
        Assert.Equal(-1, operands.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        Assert.Equal(-1, wide.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
    }
}
