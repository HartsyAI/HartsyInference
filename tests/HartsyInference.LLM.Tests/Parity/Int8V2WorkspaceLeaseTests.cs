using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A captured graph bakes the INT8 v2 workspace's address. A caller-owned graph leases that buffer, so a later
/// growth keeps it until the graph is disposed; the step graph is reset before growth, as it always was.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class Int8V2WorkspaceLeaseTests
{
    private const int Heads = 2, HeadDim = 128, SmallTokens = 64, LargeTokens = 192;
    private const float Scale = 0.09f;

    [CudaFact]
    public void ACallerOwnedGraphKeepsItsWorkspaceThroughGrowthAndReplaysCorrectly()
    {
        using IDisposable policy = Policy().Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Operands small = new(backend, SmallTokens, seed: 3), large = new(backend, LargeTokens, seed: 4);
        using Tensor eager = new(small.Query.Shape, DType.F32);
        using Tensor replayed = new(small.Query.Shape, DType.F32);
        using Tensor grown = new(large.Query.Shape, DType.F32);

        backend.ScaledDotProductAttention(eager, small.Query, small.Key, small.Value, null, Scale);
        backend.Sync();
        float[] expected = eager.AsReadOnlySpan<float>().ToArray();
        long capturedBytes = backend.Int8V2WorkspaceBytes;
        Assert.True(capturedBytes > 0);
        Assert.Equal(1, backend.Int8AttentionV2ExecutionCount);

        object graph = backend.CaptureGraph(() =>
            backend.ScaledDotProductAttention(replayed, small.Query, small.Key, small.Value, null, Scale))!;
        try
        {
            Assert.Equal(2, backend.Int8AttentionV2ExecutionCount);
            Assert.Equal(0, backend.Int8V2WorkspaceRetainedBytes);

            backend.ScaledDotProductAttention(grown, large.Query, large.Key, large.Value, null, Scale);
            backend.Sync();

            Assert.Equal(3, backend.Int8AttentionV2ExecutionCount);
            Assert.True(backend.Int8V2WorkspaceBytes > capturedBytes);
            Assert.Equal(capturedBytes, backend.Int8V2WorkspaceRetainedBytes);

            backend.LaunchGraph(graph);
            backend.Sync();
            Assert.True(replayed.AsReadOnlySpan<float>().SequenceEqual(expected));
        }
        finally
        {
            backend.DisposeGraph(graph);
        }
        Assert.Equal(0, backend.Int8V2WorkspaceRetainedBytes);
    }

    [CudaFact]
    public void GrowthResetsTheStepGraphAndRetainsNothing()
    {
        using IDisposable policy = Policy().Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Operands small = new(backend, SmallTokens, seed: 5), large = new(backend, LargeTokens, seed: 6);
        using Tensor warm = new(small.Query.Shape, DType.F32);
        using Tensor captured = new(small.Query.Shape, DType.F32);
        using Tensor grown = new(large.Query.Shape, DType.F32);
        backend.ScaledDotProductAttention(warm, small.Query, small.Key, small.Value, null, Scale);
        backend.Sync();

        backend.StepGraphBegin();
        backend.ScaledDotProductAttention(captured, small.Query, small.Key, small.Value, null, Scale);
        backend.StepGraphEndAndLaunch();
        backend.Sync();
        Assert.True(backend.StepGraphReady);
        Assert.Equal(2, backend.Int8AttentionV2ExecutionCount);

        backend.ScaledDotProductAttention(grown, large.Query, large.Key, large.Value, null, Scale);
        backend.Sync();

        Assert.False(backend.StepGraphReady);
        Assert.Equal(0, backend.Int8V2WorkspaceRetainedBytes);
        Assert.Equal(3, backend.Int8AttentionV2ExecutionCount);
    }

    private static KnobProfile Policy() => KnobProfile.Create("int8v2-workspace-lease")
        .With(EngineKnobs.Int8AttnV2, true).With(EngineKnobs.Int8AttnV2MinTokens, SmallTokens).With(EngineKnobs.SageAttn, false);

    /// <summary>Device-resident Q, K and V: a capture cannot upload from the host, so each is an op's output.</summary>
    private sealed class Operands : IDisposable
    {
        public readonly Tensor Query, Key, Value;

        public Operands(CudaBackend backend, int tokens, int seed)
        {
            Random random = new(seed);
            TensorShape shape = new(1, Heads, tokens, HeadDim);
            Query = Resident(backend, shape, random);
            Key = Resident(backend, shape, random);
            Value = Resident(backend, shape, random);
            backend.Sync();
        }

        private static Tensor Resident(CudaBackend backend, TensorShape shape, Random random)
        {
            using Tensor host = new(shape, DType.F32);
            Span<float> values = host.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() - 0.5);
            Tensor resident = new(shape, DType.F32);
            backend.Tanh(resident, host);
            return resident;
        }

        public void Dispose()
        {
            Query.Dispose();
            Key.Dispose();
            Value.Dispose();
        }
    }
}
