using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>General Sol-Attn on CUDA and on Vulkan against <see cref="SolAttentionReference"/>. Two things were fixed
/// before either backend's kernels first ran: the routing (which key blocks each query block reads exactly) must equal
/// the reference's entry for entry, because the routing stages are written to reproduce its scores; and the output must
/// agree to 1e-4 relative to the larger of 1 and the value.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class SolAttentionGpuTests
{
    /// <summary>Operands with structure for the routing to find: noise, plus a component shared by every third block.</summary>
    private sealed class Operands : IDisposable
    {
        public readonly Tensor Query, Key, Value, Output, Expected;
        public readonly int Batch, Tokens, Heads, Dim;

        public Operands(int batch, int tokens, int heads, int dim, int seed, float shared = 2f)
        {
            (Batch, Tokens, Heads, Dim) = (batch, tokens, heads, dim);
            TensorShape shape = new(batch, tokens, heads, dim);
            Random random = new(seed);
            float[] basis = new float[3 * dim];
            for (int i = 0; i < basis.Length; i++) basis[i] = (random.NextSingle() * 2f - 1f) * shared;
            Query = new Tensor(shape, DType.F32);
            Key = new Tensor(shape, DType.F32);
            Value = new Tensor(shape, DType.F32);
            Output = new Tensor(shape, DType.F32);
            Expected = new Tensor(shape, DType.F32);
            Span<float> q = Query.AsSpan<float>(), k = Key.AsSpan<float>(), v = Value.AsSpan<float>();
            for (int i = 0; i < q.Length; i++)
            {
                int token = i / (heads * dim) % tokens, channel = i % dim;
                float common = basis[token / 64 % 3 * dim + channel];
                q[i] = random.NextSingle() * 2f - 1f + common;
                k[i] = random.NextSingle() * 2f - 1f + common;
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

    private static CudaBackend Cuda()
    {
        CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        Assert.True(backend.HasSolAttention);
        return backend;
    }

    private static VulkanBackend Vulkan()
    {
        VulkanBackend backend = new();
        Assert.True(backend.HasSolAttention);
        return backend;
    }

    [CudaTheory]
    [InlineData(1f)]
    [InlineData(0.2f)]
    [InlineData(3f)]
    [InlineData(-1e6f)]
    public void OnCudaTauRoutingOverARaggedLastBlockMatchesTheReference(float tau)
    {
        using CudaBackend backend = Cuda();
        TauRoutingOverARaggedLastBlockMatchesTheReference(backend, tau);
    }

    [VulkanFact]
    public void OnVulkanTauRoutingOverARaggedLastBlockMatchesTheReference()
    {
        using VulkanBackend backend = Vulkan();
        TauRoutingOverARaggedLastBlockMatchesTheReference(backend, 1f);
        TauRoutingOverARaggedLastBlockMatchesTheReference(backend, 0.2f);
        TauRoutingOverARaggedLastBlockMatchesTheReference(backend, 3f);
        TauRoutingOverARaggedLastBlockMatchesTheReference(backend, -1e6f);
    }

    private static void TauRoutingOverARaggedLastBlockMatchesTheReference(IBackend backend, float tau)
    {
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
    public void OnCudaTopKRoutingMatchesTheReferenceWithAndWithoutTheTail(double ratio, bool tail)
    {
        using CudaBackend backend = Cuda();
        TopKRoutingMatchesTheReferenceWithAndWithoutTheTail(backend, ratio, tail);
    }

    [VulkanFact]
    public void OnVulkanTopKRoutingMatchesTheReferenceWithAndWithoutTheTail()
    {
        using VulkanBackend backend = Vulkan();
        TopKRoutingMatchesTheReferenceWithAndWithoutTheTail(backend, 0.4, true);
        TopKRoutingMatchesTheReferenceWithAndWithoutTheTail(backend, 0.4, false);
        TopKRoutingMatchesTheReferenceWithAndWithoutTheTail(backend, 0.1, true);
    }

    private static void TopKRoutingMatchesTheReferenceWithAndWithoutTheTail(IBackend backend, double ratio, bool tail)
    {
        using Operands operands = new(1, 64 * 7, 2, 128, 92);
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = ratio, Tail = tail });
    }

    [CudaFact]
    public void OnCudaSinkBlocksAndSinkQueriesMatchTheReference()
    {
        using CudaBackend backend = Cuda();
        SinkBlocksAndSinkQueriesMatchTheReference(backend);
    }

    [VulkanFact]
    public void OnVulkanSinkBlocksAndSinkQueriesMatchTheReference()
    {
        using VulkanBackend backend = Vulkan();
        SinkBlocksAndSinkQueriesMatchTheReference(backend);
    }

    private static void SinkBlocksAndSinkQueriesMatchTheReference(IBackend backend)
    {
        using Operands operands = new(1, 64 * 6 + 5, 2, 128, 93);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 2f, SinkBlocks = (1, 3), SinkQueries = (0, 1) });
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.3, SinkBlocks = (1, 3) });
        // Sink ranges past the last block are clamped, as in the reference.
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 2f, SinkBlocks = (5, 99), SinkQueries = (6, 40) });
    }

    [CudaFact]
    public void OnCudaAKeyBiasAndAKeyMaskMatchTheReference()
    {
        using CudaBackend backend = Cuda();
        AKeyBiasAndAKeyMaskMatchTheReference(backend);
    }

    [VulkanFact]
    public void OnVulkanAKeyBiasAndAKeyMaskMatchTheReference()
    {
        using VulkanBackend backend = Vulkan();
        AKeyBiasAndAKeyMaskMatchTheReference(backend);
    }

    private static void AKeyBiasAndAKeyMaskMatchTheReference(IBackend backend)
    {
        using Operands operands = new(2, 64 * 4 + 30, 2, 128, 94);
        using Tensor perBatch = operands.Bias(perBatch: true, mask: false, 1);
        using Tensor shared = operands.Bias(perBatch: false, mask: false, 2);
        using Tensor mask = operands.Bias(perBatch: false, mask: true, 3);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1.5f, SinkBlocks = (0, 2), KeyBias = perBatch });
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1.5f, SinkBlocks = (0, 2), KeyBias = shared });
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1.5f, SinkBlocks = (0, 2), KeyBias = mask });
    }

    [CudaFact]
    public void OnCudaBlockLengthsWithDeadRowsMatchTheReference()
    {
        using CudaBackend backend = Cuda();
        BlockLengthsWithDeadRowsMatchTheReference(backend);
    }

    [VulkanFact]
    public void OnVulkanBlockLengthsWithDeadRowsMatchTheReference()
    {
        using VulkanBackend backend = Vulkan();
        BlockLengthsWithDeadRowsMatchTheReference(backend);
    }

    private static void BlockLengthsWithDeadRowsMatchTheReference(IBackend backend)
    {
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
    public void OnCudaOtherShapesAndACustomScaleMatchTheReference(int batch, int heads, int dim, int tokens)
    {
        using CudaBackend backend = Cuda();
        OtherShapesAndACustomScaleMatchTheReference(backend, batch, heads, dim, tokens);
    }

    [VulkanFact]
    public void OnVulkanOtherShapesAndACustomScaleMatchTheReference()
    {
        using VulkanBackend backend = Vulkan();
        OtherShapesAndACustomScaleMatchTheReference(backend, 2, 3, 128, 64 * 3 + 1);
        OtherShapesAndACustomScaleMatchTheReference(backend, 1, 4, 64, 64 * 6);
        OtherShapesAndACustomScaleMatchTheReference(backend, 1, 2, 256, 64 * 4 + 9);
        OtherShapesAndACustomScaleMatchTheReference(backend, 1, 5, 40, 64 * 5);
        OtherShapesAndACustomScaleMatchTheReference(backend, 1, 2, 128, 50);
    }

    private static void OtherShapesAndACustomScaleMatchTheReference(IBackend backend, int batch, int heads, int dim, int tokens)
    {
        using Operands operands = new(batch, tokens, heads, dim, 96);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 1f, Scale = 0.07f });
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.34 });
    }

    [CudaFact]
    public void OnCudaWhereTheTailCarriesWeightItMatchesTheReference()
    {
        using CudaBackend backend = Cuda();
        WhereTheTailCarriesWeightItMatchesTheReference(backend);
    }

    [VulkanFact]
    public void OnVulkanWhereTheTailCarriesWeightItMatchesTheReference()
    {
        using VulkanBackend backend = Vulkan();
        WhereTheTailCarriesWeightItMatchesTheReference(backend);
    }

    /// <summary>A weak shared component keeps the pooled scores near each row's maximum, so the unread blocks' pooled
    /// entries carry real weight: the reference's own output moves by more than 0.05 when the tail is turned off.</summary>
    private static void WhereTheTailCarriesWeightItMatchesTheReference(IBackend backend)
    {
        using Operands operands = new(1, 64 * 8, 2, 128, 101, shared: 0.15f);
        using Tensor withTail = new(operands.Query.Shape, DType.F32);
        using Tensor withoutTail = new(operands.Query.Shape, DType.F32);
        SolAttentionReference.Run(withTail, operands.Query, operands.Key, operands.Value, new SolAttentionOptions { TopKRatio = 0.15 });
        SolAttentionReference.Run(withoutTail, operands.Query, operands.Key, operands.Value, new SolAttentionOptions { TopKRatio = 0.15, Tail = false });
        float moved = 0f;
        ReadOnlySpan<float> on = withTail.AsReadOnlySpan<float>(), off = withoutTail.AsReadOnlySpan<float>();
        for (int i = 0; i < on.Length; i++) moved = MathF.Max(moved, MathF.Abs(on[i] - off[i]));
        Assert.True(moved > 0.05f, $"the tail moves this fixture by only {moved}");

        (int read, int pairs) = operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.15 });
        Assert.InRange(read, 1, pairs - 1);
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.15, Tail = false });
    }

    [CudaFact]
    public void OnCudaMoreKeyBlocksThanOneWorkgroupHoldsMatchTheReference()
    {
        using CudaBackend backend = Cuda();
        MoreKeyBlocksThanOneWorkgroupHoldsMatchTheReference(backend);
    }

    [VulkanFact]
    public void OnVulkanMoreKeyBlocksThanOneWorkgroupHoldsMatchTheReference()
    {
        using VulkanBackend backend = Vulkan();
        MoreKeyBlocksThanOneWorkgroupHoldsMatchTheReference(backend);
    }

    /// <summary>257 blocks: the pooled and routing stages give a thread or invocation more than one key block.</summary>
    private static void MoreKeyBlocksThanOneWorkgroupHoldsMatchTheReference(IBackend backend)
    {
        using Operands operands = new(1, 64 * 257, 1, 16, 102);
        operands.RunAndCompare(backend, new SolAttentionOptions { TopKRatio = 0.01 });
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = 2.5f });
    }

    [CudaFact]
    public void OnCudaAFullyMaskedBlockAndFullyMaskedRowsMatchTheReference()
    {
        using CudaBackend backend = Cuda();
        AFullyMaskedBlockAndFullyMaskedRowsMatchTheReference(backend);
    }

    [VulkanFact]
    public void OnVulkanAFullyMaskedBlockAndFullyMaskedRowsMatchTheReference()
    {
        using VulkanBackend backend = Vulkan();
        AFullyMaskedBlockAndFullyMaskedRowsMatchTheReference(backend);
    }

    /// <summary>A mask over one whole read block, and a mask over every key with the tail off, which leaves each row
    /// nothing to attend: the output is zero, as in the reference.</summary>
    private static void AFullyMaskedBlockAndFullyMaskedRowsMatchTheReference(IBackend backend)
    {
        const int tokens = 64 * 3;
        using Operands operands = new(1, tokens, 2, 128, 103);
        using Tensor oneBlock = new(new TensorShape(tokens), DType.F32);
        oneBlock.AsSpan<float>().Slice(64, 64).Fill(float.NegativeInfinity);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = -1e6f, KeyBias = oneBlock });
        using Tensor everyKey = new(new TensorShape(tokens), DType.F32);
        everyKey.AsSpan<float>().Fill(float.NegativeInfinity);
        operands.RunAndCompare(backend, new SolAttentionOptions { Tau = -1e6f, Tail = false, KeyBias = everyKey });
        Assert.Equal(-1, operands.Output.AsReadOnlySpan<float>().IndexOfAnyExcept(0f));
    }

    [CudaFact]
    public void OnCudaWithoutTheRoutingReportTheOutputIsTheSame()
    {
        using CudaBackend backend = Cuda();
        WithoutTheRoutingReportTheOutputIsTheSame(backend);
    }

    [VulkanFact]
    public void OnVulkanWithoutTheRoutingReportTheOutputIsTheSame()
    {
        using VulkanBackend backend = Vulkan();
        WithoutTheRoutingReportTheOutputIsTheSame(backend);
    }

    private static void WithoutTheRoutingReportTheOutputIsTheSame(IBackend backend)
    {
        using Operands operands = new(1, 64 * 4, 2, 128, 97);
        SolAttentionOptions options = new() { Tau = 1f };
        operands.RunAndCompare(backend, options);
        float[] reported = operands.Output.AsReadOnlySpan<float>().ToArray();
        backend.SolAttention(operands.Output, operands.Query, operands.Key, operands.Value, options);
        backend.Sync();
        Assert.True(reported.AsSpan().SequenceEqual(operands.Output.AsReadOnlySpan<float>()));
    }

    [CudaFact]
    public void OnCudaWhatTheKernelsDoNotTakeIsRefusedAndWritesNothing()
    {
        using CudaBackend backend = Cuda();
        WhatTheKernelsDoNotTakeIsRefusedAndWritesNothing(backend);
    }

    [VulkanFact]
    public void OnVulkanWhatTheKernelsDoNotTakeIsRefusedAndWritesNothing()
    {
        using VulkanBackend backend = Vulkan();
        WhatTheKernelsDoNotTakeIsRefusedAndWritesNothing(backend);
    }

    private static void WhatTheKernelsDoNotTakeIsRefusedAndWritesNothing(IBackend backend)
    {
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
