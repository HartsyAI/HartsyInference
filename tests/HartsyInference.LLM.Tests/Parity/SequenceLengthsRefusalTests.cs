using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A call with a key length per sequence, or one that asks for the LSE, has two honest outcomes on a GPU backend:
/// it is refused whole and nothing is written, or a row that declares it runs it and the result is the host reference's.
/// What must never happen is the third: the call run with one length, or without its LSE. The checks were written
/// before any GPU row declared either and have not changed since: on a backend with a <c>flash.decode.lengths</c> row
/// the lengths cases now take the second outcome, and an LSE on a call with one shared length is still refused.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class SequenceLengthsRefusalTests
{
    private const float Untouched = -7f;

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    /// <summary>Runs <paramref name="call"/> on <paramref name="backend"/>. A refusal must leave <paramref name="output"/> and
    /// <paramref name="lse"/> as they were; anything else must equal the host reference's result for the same call.</summary>
    private static bool RefusedOrMatchesTheReference(IBackend backend, AttentionCall call, Tensor output, Tensor? lse)
    {
        using Tensor expected = new(output.Shape, DType.F32);
        using Tensor? expectedLse = lse is null ? null : new Tensor(lse.Shape, DType.F32);
        AttentionCall reference = call with { Output = call.Output with { Storage = expected }, Lse = expectedLse };
        AttentionHostReference.Run(in reference);

        output.AsSpan<float>().Fill(Untouched);
        lse?.AsSpan<float>().Fill(Untouched);
        bool refused = false;
        try
        {
            backend.Attention(in call);
        }
        catch (AttentionUnsupportedException)
        {
            refused = true;
        }
        backend.Sync();
        if (refused)
        {
            Assert.Equal(-1, output.AsReadOnlySpan<float>().IndexOfAnyExcept(Untouched));
            if (lse is not null) Assert.Equal(-1, lse.AsReadOnlySpan<float>().IndexOfAnyExcept(Untouched));
            return true;
        }
        AssertClose(expected.AsReadOnlySpan<float>(), output.AsReadOnlySpan<float>(), "output");
        if (lse is not null) AssertClose(expectedLse!.AsReadOnlySpan<float>(), lse.AsReadOnlySpan<float>(), "LSE");
        return false;
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNegativeInfinity(expected[i])) Assert.True(float.IsNegativeInfinity(actual[i]), $"{what} {i}: {actual[i]}, expected negative infinity");
            else Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])), $"{what} {i}: {actual[i]}, expected {expected[i]}");
        }
    }

    private static void AssertNeverRunsWithOneLengthOrWithoutTheLse(IBackend backend)
    {
        const int batch = 2, heads = 8, kvHeads = 2, capacity = 96, dim = 64;
        foreach ((int rows, bool causal, int[] lengths) in ((int, bool, int[])[])[(1, true, [80, 70]), (3, true, [80, 2]), (3, false, [0, 70])])
        {
            using Tensor query = Random(new TensorShape(batch, heads, rows, dim), 31);
            using Tensor key = Random(new TensorShape(batch, kvHeads, capacity, dim), 32);
            using Tensor value = Random(key.Shape, 33);
            using Tensor output = new(new TensorShape(batch, rows, heads * dim), DType.F32);
            using Tensor lse = new(new TensorShape(batch, heads, rows), DType.F32);
            AttentionCall call = AttentionCall.FromDecodeWithLengths(output, query, key, value, lengths, causal, 0.125f);
            RefusedOrMatchesTheReference(backend, call, output, null);
            RefusedOrMatchesTheReference(backend, call with { Lse = lse }, output, lse);
        }

        // An LSE alone, on an ordinary single-sequence decode the backend serves without it.
        using Tensor oneQuery = Random(new TensorShape(1, heads, 1, dim), 41);
        using Tensor oneKey = Random(new TensorShape(1, kvHeads, capacity, dim), 42);
        using Tensor oneValue = Random(oneKey.Shape, 43);
        using Tensor oneOutput = new(oneQuery.Shape, DType.F32);
        using Tensor oneLse = new(new TensorShape(1, heads, 1), DType.F32);
        AttentionCall plain = AttentionCall.FromFlashAttention(oneOutput, oneQuery, oneKey, oneValue, 80, causal: true, qOffset: 79, scale: 0.125f,
            softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: null);
        RefusedOrMatchesTheReference(backend, plain with { Lse = oneLse }, oneOutput, oneLse);
        Assert.False(RefusedOrMatchesTheReference(backend, plain, oneOutput, null), "the plain decode was refused");
    }

    [CudaFact]
    public void CudaNeverRunsSuchACallWithOneLengthOrWithoutItsLse()
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        AssertNeverRunsWithOneLengthOrWithoutTheLse(backend);
    }

    [VulkanFact]
    public void VulkanNeverRunsSuchACallWithOneLengthOrWithoutItsLse()
    {
        using VulkanBackend backend = new();
        AssertNeverRunsWithOneLengthOrWithoutTheLse(backend);
    }
}
