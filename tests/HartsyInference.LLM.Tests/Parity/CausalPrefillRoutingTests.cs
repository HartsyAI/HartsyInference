using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Causal prefill over an F32 KV cache: the engine's causal kernel takes every call it can, so those build no
/// <c>[Sq, Skv]</c> mask, and cuDNN with a materialized mask keeps the rest.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class CausalPrefillRoutingTests
{
    private const int Heads = 4, KvHeads = 2;

    /// <summary>The routes: head dims 64 and 128 with at least 16 query rows run mask-free; anything else stays on cuDNN.</summary>
    [CudaTheory]
    [InlineData(64, 64, true, "fa2.causal")]
    [InlineData(128, 64, true, "fa2.causal")]
    [InlineData(256, 64, true, "cudnn.causal")]
    [InlineData(64, 8, true, "cudnn.causal")]
    [InlineData(64, 64, false, "cudnn.causal")]
    public void EligiblePrefillRunsWithoutAMask(int headDim, int tokens, bool fa2Prefill, string implementation)
    {
        using IDisposable policy = KnobProfile.Create("causal-prefill-routing").With(EngineKnobs.SdpaCudnn, true)
            .With(EngineKnobs.Fa2Prefill, fa2Prefill).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        Random random = new(31);
        using Tensor query = RandomTensor(new TensorShape(1, Heads, tokens, headDim), random);
        using Tensor key = RandomTensor(new TensorShape(1, KvHeads, tokens, headDim), random);
        using Tensor value = RandomTensor(key.Shape, random);
        using Tensor output = new(query.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.FlashAttention(output, query, key, value, tokens, Heads / KvHeads, causal: true, qOffset: 0, 1f / MathF.Sqrt(headDim));
        backend.Sync();

        AttentionEvidenceEvent execution = Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Execution);
        Assert.Equal(implementation, execution.Implementation);
        long maskBytes = evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.MaterializeMask).Sum(e => e.Bytes);
        Assert.Equal(implementation == "cudnn.causal" ? (long)tokens * tokens * sizeof(float) : 0, maskBytes);
    }

    /// <summary>Both prefill rows compute QK in F16 and stay within the same bound of the host reference, with and without a
    /// sliding window and with a cached prefix. The bound (5e-3 absolute, on outputs that are averages of values in
    /// [-0.5, 0.5]) was fixed before either row's error was looked at.</summary>
    [CudaTheory]
    [InlineData(64, true, 0, 0)]
    [InlineData(128, true, 0, 0)]
    [InlineData(64, true, 24, 0)]
    [InlineData(128, true, 0, 32)]
    [InlineData(64, false, 0, 0)]
    [InlineData(128, false, 24, 0)]
    public void BothPrefillRowsMatchTheHostReference(int headDim, bool fa2Prefill, int window, int cachedPrefix)
    {
        const int queryTokens = 80;
        const float tolerance = 5e-3f;
        int keys = cachedPrefix + queryTokens;
        using IDisposable policy = KnobProfile.Create("causal-prefill-parity").With(EngineKnobs.SdpaCudnn, true)
            .With(EngineKnobs.Fa2Prefill, fa2Prefill).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        Random random = new(37);
        using Tensor query = RandomTensor(new TensorShape(1, Heads, queryTokens, headDim), random);
        using Tensor key = RandomTensor(new TensorShape(1, KvHeads, keys, headDim), random);
        using Tensor value = RandomTensor(key.Shape, random);
        using Tensor output = new(query.Shape, DType.F32);
        using Tensor expected = new(query.Shape, DType.F32);
        float scale = 1f / MathF.Sqrt(headDim);
        AttentionCall reference = AttentionCall.FromFlashAttention(expected, query, key, value, keys, true, cachedPrefix, scale, 0f, null,
            window, null);
        AttentionHostReference.Run(in reference);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.FlashAttention(output, query, key, value, keys, Heads / KvHeads, causal: true, qOffset: cachedPrefix, scale,
            slidingWindow: window);
        backend.Sync();

        Assert.Equal(fa2Prefill ? "fa2.causal" : "cudnn.causal",
            Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Execution).Implementation);
        ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>(), wanted = expected.AsReadOnlySpan<float>();
        float worst = 0;
        for (int i = 0; i < actual.Length; i++) worst = MathF.Max(worst, MathF.Abs(actual[i] - wanted[i]));
        Assert.True(worst <= tolerance, $"max abs error {worst} exceeds {tolerance}");
    }

    private static Tensor RandomTensor(TensorShape shape, Random random)
    {
        Tensor tensor = new(shape, DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() - 0.5);
        return tensor;
    }
}
