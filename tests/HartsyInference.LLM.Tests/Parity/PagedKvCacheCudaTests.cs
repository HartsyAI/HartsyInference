using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>On the device the paged cache's contiguous prefix is kept between calls, so it must survive what a KV buffer
/// survives: activations freed or offloaded between steps. It is compared with a contiguous cache fed the same rows and
/// with the rows themselves.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class PagedKvCacheCudaTests
{
    private const int Layers = 2, KvHeads = 2, HeadDim = 64, PageSize = 16, MaxPages = 32;

    [CudaTheory]
    [InlineData("none")]
    [InlineData("offload")]
    [InlineData("free")]
    public void ThePrefixMatchesAContiguousCacheAcrossActivationRelease(string release)
    {
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using PagedKvPool pool = new(Layers, KvHeads, HeadDim, PageSize, MaxPages);
        using PagedKvCache paged = new(pool);
        using FixedKvCache contiguous = new(Layers, 1, KvHeads, HeadDim, PageSize * MaxPages, DType.F32);
        Random random = new(99);
        List<float[]>[] keys = [[], []], values = [[], []];

        void Step(int tokens)
        {
            for (int layer = 0; layer < Layers; layer++)
            {
                using Tensor k = Rows(random, tokens), v = Rows(random, tokens);
                Remember(keys[layer], k, tokens);
                Remember(values[layer], v, tokens);
                paged.AppendStep(backend, layer, k, v);
                contiguous.AppendStep(backend, layer, k, v);
                int valid = paged.CurrentLength + tokens;
                AssertRows(paged.KeyPrefix(layer), contiguous.KeyPrefix(layer), keys[layer], valid, $"layer {layer} keys");
                AssertRows(paged.ValuePrefix(layer), contiguous.ValuePrefix(layer), values[layer], valid, $"layer {layer} values");
            }
            paged.AdvanceLength(tokens);
            contiguous.AdvanceLength(tokens);
            if (release == "offload") backend.OffloadActivations(long.MaxValue);
            else if (release == "free") backend.FreeActivations(trimPool: false);
        }

        Step(40);
        for (int step = 0; step < 40; step++) Step(1);
        paged.Truncate(50);
        contiguous.Truncate(50);
        foreach (List<float[]> rows in keys) rows.RemoveRange(50, rows.Count - 50);
        foreach (List<float[]> rows in values) rows.RemoveRange(50, rows.Count - 50);
        for (int step = 0; step < 20; step++) Step(1);
    }

    private static Tensor Rows(Random random, int tokens)
    {
        Tensor tensor = new(new TensorShape(1, KvHeads, tokens, HeadDim), DType.F32);
        Span<float> data = tensor.AsSpan<float>();
        for (int i = 0; i < data.Length; i++) data[i] = (float)random.NextDouble();
        return tensor;
    }

    /// <summary>Keeps a host copy of each appended row as <c>[head][headDim]</c> flattened head-major.</summary>
    private static void Remember(List<float[]> rows, Tensor appended, int tokens)
    {
        ReadOnlySpan<float> data = appended.AsReadOnlySpan<float>();
        for (int token = 0; token < tokens; token++)
        {
            float[] row = new float[KvHeads * HeadDim];
            for (int head = 0; head < KvHeads; head++)
                data.Slice((head * tokens + token) * HeadDim, HeadDim).CopyTo(row.AsSpan(head * HeadDim, HeadDim));
            rows.Add(row);
        }
    }

    private static void AssertRows(Tensor paged, Tensor contiguous, List<float[]> expected, int validRows, string what)
    {
        ReadOnlySpan<float> actual = paged.AsReadOnlySpan<float>(), reference = contiguous.AsReadOnlySpan<float>();
        int pagedRows = (int)paged.Shape[2], contiguousRows = (int)contiguous.Shape[2];
        for (int head = 0; head < KvHeads; head++)
            for (int row = 0; row < validRows; row++)
            {
                ReadOnlySpan<float> want = expected[row].AsSpan(head * HeadDim, HeadDim);
                Assert.True(reference.Slice((head * contiguousRows + row) * HeadDim, HeadDim).SequenceEqual(want),
                    $"{what}: the contiguous cache lost head {head}, row {row}.");
                Assert.True(actual.Slice((head * pagedRows + row) * HeadDim, HeadDim).SequenceEqual(want),
                    $"{what}: the paged prefix differs at head {head}, row {row} of {validRows}.");
            }
    }
}
