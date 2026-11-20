using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>No GPU row takes a key length per sequence or writes an LSE yet. A backend sent such a call says so and writes
/// nothing; it does not run the call with one length, or without the LSE.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class SequenceLengthsRefusalTests
{
    private static void AssertRefused(IBackend backend)
    {
        const int batch = 2, heads = 8, kvHeads = 2, rows = 1, capacity = 96, dim = 64;
        using Tensor query = new(new TensorShape(batch, heads, rows, dim), DType.F32);
        using Tensor key = new(new TensorShape(batch, kvHeads, capacity, dim), DType.F32);
        using Tensor value = new(key.Shape, DType.F32);
        using Tensor output = new(new TensorShape(batch, rows, heads * dim), DType.F32);
        using Tensor lse = new(new TensorShape(batch, heads, rows), DType.F32);
        output.AsSpan<float>().Fill(-7f);
        lse.AsSpan<float>().Fill(-7f);

        AttentionCall lengths = AttentionCall.FromDecodeWithLengths(output, query, key, value, [80, 70], causal: true, 0.125f);
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in lengths));
        AttentionCall both = lengths with { Lse = lse };
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in both));

        // An LSE alone, on an ordinary single-sequence decode the backend otherwise serves.
        using Tensor oneQuery = new(new TensorShape(1, heads, rows, dim), DType.F32);
        using Tensor oneKey = new(new TensorShape(1, kvHeads, capacity, dim), DType.F32);
        using Tensor oneValue = new(oneKey.Shape, DType.F32);
        using Tensor oneOutput = new(oneQuery.Shape, DType.F32);
        using Tensor oneLse = new(new TensorShape(1, heads, rows), DType.F32);
        oneOutput.AsSpan<float>().Fill(-7f);
        oneLse.AsSpan<float>().Fill(-7f);
        AttentionCall plain = AttentionCall.FromFlashAttention(oneOutput, oneQuery, oneKey, oneValue, 80, causal: true, qOffset: 79, scale: 0.125f,
            softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: null);
        AttentionCall withLse = plain with { Lse = oneLse };
        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in withLse));
        backend.Sync();

        Assert.Equal(-1, output.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        Assert.Equal(-1, lse.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        Assert.Equal(-1, oneOutput.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
        Assert.Equal(-1, oneLse.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));

        // The same decode without the LSE is served: the refusal is about the LSE, not the call.
        backend.Attention(in plain);
        backend.Sync();
        Assert.NotEqual(-1, oneOutput.AsReadOnlySpan<float>().IndexOfAnyExcept(-7f));
    }

    [CudaFact]
    public void CudaRefusesSequenceLengthsAndAnLseAndWritesNothing()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        AssertRefused(backend);
    }

    [VulkanFact]
    public void VulkanRefusesSequenceLengthsAndAnLseAndWritesNothing()
    {
        using VulkanBackend backend = new();
        AssertRefused(backend);
    }
}
