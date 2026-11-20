using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Eager split decode keeps its partial states in one high-water buffer: it allocates once, then nothing per
/// step, and computes the same bits whatever the buffer held before. A call recorded into a graph keeps its own scratch.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class DecodePartialsWorkspaceTests
{
    private const int Heads = 2, HeadDim = 64, ShortKeys = 128, LongKeys = 512;
    private const float Scale = 0.1f;

    [CudaFact]
    public void EagerDecodeGrowsTheWorkspaceOnceThenAllocatesNothing()
    {
        using IDisposable policy = Policy().Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Operands operands = new(backend, ShortKeys, seed: 7);
        using Tensor first = new(operands.Query.Shape, DType.F32);
        using Tensor second = new(operands.Query.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        Decode(backend, first, operands);
        backend.Sync();
        AttentionEvidenceEvent growth = Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Workspace);
        Assert.Equal("cuda.flash.partials.workspace.grow", growth.Site);
        Assert.Equal(growth.Bytes, backend.DecodePartialsWorkspaceBytes);

        Decode(backend, second, operands);
        backend.Sync();

        Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Workspace);
        Assert.Equal(2, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Execution));
        Assert.True(first.AsReadOnlySpan<float>().SequenceEqual(second.AsReadOnlySpan<float>()));
    }

    [CudaFact]
    public void ALargerBufferLeftByAnotherCallDoesNotChangeTheResult()
    {
        using IDisposable policy = Policy().Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Operands small = new(backend, ShortKeys, seed: 8), large = new(backend, LongKeys, seed: 9);
        using Tensor fresh = new(small.Query.Shape, DType.F32);
        using Tensor reused = new(small.Query.Shape, DType.F32);
        using Tensor other = new(large.Query.Shape, DType.F32);
        Decode(backend, fresh, small);
        backend.Sync();
        float[] expected = fresh.AsReadOnlySpan<float>().ToArray();
        long smallBytes = backend.DecodePartialsWorkspaceBytes;

        Decode(backend, other, large);
        backend.Sync();
        Assert.True(backend.DecodePartialsWorkspaceBytes > smallBytes);
        Decode(backend, reused, small);
        backend.Sync();

        Assert.True(reused.AsReadOnlySpan<float>().SequenceEqual(expected));
    }

    [CudaFact]
    public void ACapturedDecodeKeepsItsOwnScratchAndReplaysAfterTheWorkspaceGrows()
    {
        using IDisposable policy = Policy().Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Operands small = new(backend, ShortKeys, seed: 10), large = new(backend, LongKeys, seed: 11);
        using Tensor eager = new(small.Query.Shape, DType.F32);
        using Tensor replayed = new(small.Query.Shape, DType.F32);
        using Tensor other = new(large.Query.Shape, DType.F32);
        Decode(backend, eager, small);
        backend.Sync();
        float[] expected = eager.AsReadOnlySpan<float>().ToArray();
        long workspaceBytes = backend.DecodePartialsWorkspaceBytes;

        object graph;
        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(64))
        {
            graph = backend.CaptureGraph(() => Decode(backend, replayed, small))!;
            AttentionEvidenceEvent scratch = Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Workspace);
            Assert.EndsWith(".partials", scratch.Site);
            Assert.Equal(3, scratch.Count);
        }
        try
        {
            Assert.Equal(workspaceBytes, backend.DecodePartialsWorkspaceBytes);

            Decode(backend, other, large);
            backend.Sync();
            Assert.True(backend.DecodePartialsWorkspaceBytes > workspaceBytes);

            backend.LaunchGraph(graph);
            backend.Sync();
            Assert.True(replayed.AsReadOnlySpan<float>().SequenceEqual(expected));
        }
        finally
        {
            backend.DisposeGraph(graph);
        }
    }

    private static void Decode(CudaBackend backend, Tensor output, Operands operands) =>
        backend.FlashAttention(output, operands.Query, operands.Key, operands.Value, operands.Keys, 1, causal: true,
            qOffset: operands.Keys - 1, Scale);

    private static KnobProfile Policy() => KnobProfile.Create("decode-partials-workspace").With(EngineKnobs.FlashSplitForce, true);

    /// <summary>Device-resident Q, K and V: a capture cannot upload from the host, so each is an op's output.</summary>
    private sealed class Operands : IDisposable
    {
        public readonly Tensor Query, Key, Value;
        public readonly int Keys;

        public Operands(CudaBackend backend, int keys, int seed)
        {
            Keys = keys;
            Random random = new(seed);
            Query = Resident(backend, new TensorShape(1, Heads, 1, HeadDim), random);
            Key = Resident(backend, new TensorShape(1, Heads, keys, HeadDim), random);
            Value = Resident(backend, new TensorShape(1, Heads, keys, HeadDim), random);
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
