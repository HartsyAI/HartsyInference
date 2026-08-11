using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Regression coverage for native-F16 SDPA's materialized fallback with an F32 additive mask.</summary>
[Collection("CudaSerial")]
public sealed unsafe class F16MaskedSdpaTests
{
    private readonly ITestOutputHelper _output;

    public F16MaskedSdpaTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void NativeF16Sdpa_F32BatchMask_CudnnDisabled_MatchesHalfRoundedReference()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int Batch = 2;
        const int Heads = 3;
        const int QueryLength = 5;
        const int KeyLength = 7;
        const int HeadDim = 16;
        const float Tolerance = 8e-3f;
        float scale = 1f / MathF.Sqrt(HeadDim);

        using Tensor query = new(new TensorShape(Batch, Heads, QueryLength, HeadDim), DType.F16);
        using Tensor key = new(new TensorShape(Batch, Heads, KeyLength, HeadDim), DType.F16);
        using Tensor value = new(new TensorShape(Batch, Heads, KeyLength, HeadDim), DType.F16);
        using Tensor mask = new(new TensorShape(Batch, 1, QueryLength, KeyLength), DType.F32);
        using Tensor actual = new(new TensorShape(Batch, Heads, QueryLength, HeadDim), DType.F16);

        Half[] queryValues = FillHalf(query, seed: 0x51A7, magnitude: 0.7f);
        Half[] keyValues = FillHalf(key, seed: 0xA771, magnitude: 0.7f);
        Half[] valueValues = FillHalf(value, seed: 0xC0DE, magnitude: 0.7f);
        float[] maskValues = FillBatchMask(mask, Batch, QueryLength, KeyLength);
        const int FullyMaskedBatch = 1, FullyMaskedQuery = 2;
        float* maskPointer = (float*)mask.DataPointer;
        for (int keyIndex = 0; keyIndex < KeyLength; keyIndex++)
        {
            int index = (FullyMaskedBatch * QueryLength + FullyMaskedQuery) * KeyLength + keyIndex;
            maskValues[index] = -1e30f;
            maskPointer[index] = -1e30f;
        }
        float[] expected = ReferenceSdpa(
            queryValues, keyValues, valueValues, maskValues,
            Batch, Heads, QueryLength, KeyLength, HeadDim, scale);

        string? previousCudnn = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "0");
            using CudaBackend backend = new(0, ptxDir);
            backend.ScaledDotProductAttention(actual, query, key, value, mask, scale);
            backend.Sync();

            Half* actualValues = (Half*)actual.DataPointer;
            float maxAbsDifference = 0f;
            long worstIndex = -1;
            for (long i = 0; i < actual.ElementCount; i++)
            {
                float observed = (float)actualValues[i];
                Assert.True(float.IsFinite(observed), $"native-F16 masked SDPA produced a non-finite value at {i}: {observed}");
                float difference = MathF.Abs(observed - expected[i]);
                if (difference > maxAbsDifference)
                {
                    maxAbsDifference = difference;
                    worstIndex = i;
                }
            }

            _output.WriteLine($"max |F16 CUDA - F32 reference| = {maxAbsDifference:E3} at index {worstIndex}");
            Assert.False(backend.CudnnSdpaEngaged, "cuDNN must remain disabled so this test exercises the materialized fallback.");
            Assert.Equal(0, backend.CudnnSdpaExecutionCount);
            // Q/K/V and the output are binary16, and the fallback also stores scores and probabilities in
            // binary16. With every operand bounded by 0.7, 8e-3 allows the accumulated rounding from those
            // four F16 boundaries while remaining far below the error caused by dropping or misreading the mask.
            Assert.True(maxAbsDifference <= Tolerance,
                $"native-F16 masked SDPA differs from the half-rounded F32 reference by {maxAbsDifference:E3} " +
                $"(tolerance {Tolerance:E3}, worst index {worstIndex}).");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", previousCudnn);
        }
    }

    [Theory]
    [InlineData(3, false)] // B != H, rank-3 [H,Sq,Skv]
    [InlineData(2, true)]  // B == H, rank-4 [1,H,Sq,Skv] must not be mistaken for [B,1,...]
    public void MaterializedSdpa_PerHeadMask_BroadcastsByShape(int heads, bool rankFour)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int Batch = 2, QueryLength = 4, KeyLength = 6, HeadDim = 16;
        float scale = 1f / MathF.Sqrt(HeadDim);
        using Tensor query = RandomF32(new TensorShape(Batch, heads, QueryLength, HeadDim), 701 + heads);
        using Tensor key = RandomF32(new TensorShape(Batch, heads, KeyLength, HeadDim), 711 + heads);
        using Tensor value = RandomF32(new TensorShape(Batch, heads, KeyLength, HeadDim), 721 + heads);
        TensorShape maskShape = rankFour
            ? new TensorShape(1, heads, QueryLength, KeyLength)
            : new TensorShape(heads, QueryLength, KeyLength);
        using Tensor mask = new(maskShape, DType.F32);
        float* maskValues = (float*)mask.DataPointer;
        for (int head = 0; head < heads; head++)
            for (int queryIndex = 0; queryIndex < QueryLength; queryIndex++)
                for (int keyIndex = 0; keyIndex < KeyLength; keyIndex++)
                {
                    long index = ((long)head * QueryLength + queryIndex) * KeyLength + keyIndex;
                    maskValues[index] = 0.19f * (((head + 1) * 13 + queryIndex * 5 + keyIndex * 3) % 11 - 5);
                }

        using Tensor expected = new(query.Shape, DType.F32);
        using (CpuBackend cpu = new())
            ((IBackend)cpu).ScaledDotProductAttention(expected, query, key, value, mask, scale);

        using Tensor actual = new(query.Shape, DType.F32);
        string? previousCudnn = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        string? previousSage = Environment.GetEnvironmentVariable("HARTSY_SAGE_ATTN");
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "0");
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "0");
            using CudaBackend backend = new(0, ptxDir);
            backend.ScaledDotProductAttention(actual, query, key, value, mask, scale);
            backend.Sync();
            _ = *(float*)actual.DataPointer;
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", previousCudnn);
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", previousSage);
        }

        float* expectedValues = (float*)expected.DataPointer;
        float* actualValues = (float*)actual.DataPointer;
        float maxDifference = 0f;
        for (long index = 0; index < actual.ElementCount; index++)
        {
            Assert.True(float.IsFinite(actualValues[index]), $"non-finite output at {index}: {actualValues[index]}");
            maxDifference = MathF.Max(maxDifference, MathF.Abs(expectedValues[index] - actualValues[index]));
        }
        _output.WriteLine($"heads={heads} rank4={rankFour}: max |CUDA - CPU| = {maxDifference:E3}");
        Assert.True(maxDifference <= 2e-3f, $"per-head mask broadcast diverged by {maxDifference:E3}");
    }

    private static Half[] FillHalf(Tensor tensor, int seed, float magnitude)
    {
        Random random = new(seed);
        Half[] values = new Half[checked((int)tensor.ElementCount)];
        Half* destination = (Half*)tensor.DataPointer;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (Half)((float)(random.NextDouble() * 2.0 - 1.0) * magnitude);
            destination[i] = values[i];
        }
        return values;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        float* values = (float*)tensor.DataPointer;
        Random random = new(seed);
        for (long index = 0; index < tensor.ElementCount; index++)
            values[index] = (float)(random.NextDouble() * 1.4 - 0.7);
        return tensor;
    }

    private static float[] FillBatchMask(Tensor mask, int batch, int queryLength, int keyLength)
    {
        float[] values = new float[batch * queryLength * keyLength];
        float* destination = (float*)mask.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < queryLength; q++)
            {
                for (int k = 0; k < keyLength; k++)
                {
                    int index = (b * queryLength + q) * keyLength + k;
                    float bias = 0.137f * (((b + 2) * 11 + q * 7 + k * 5) % 9 - 4);
                    if (k == (q + b) % keyLength) bias += 2.25f;
                    if (k == (q + 2 * b + 3) % keyLength) bias -= 3.75f;
                    values[index] = bias;
                    destination[index] = bias;
                }
            }
        }
        return values;
    }

    private static float[] ReferenceSdpa(
        Half[] query, Half[] key, Half[] value, float[] mask,
        int batch, int heads, int queryLength, int keyLength, int headDim, float scale)
    {
        float[] output = new float[batch * heads * queryLength * headDim];
        float[] scores = new float[keyLength];
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int q = 0; q < queryLength; q++)
                {
                    float maxScore = float.NegativeInfinity;
                    for (int k = 0; k < keyLength; k++)
                    {
                        int queryBase = ((b * heads + h) * queryLength + q) * headDim;
                        int keyBase = ((b * heads + h) * keyLength + k) * headDim;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                            dot += (float)query[queryBase + d] * (float)key[keyBase + d];

                        float score = dot * scale + mask[(b * queryLength + q) * keyLength + k];
                        scores[k] = score;
                        maxScore = MathF.Max(maxScore, score);
                    }

                    float denominator = 0f;
                    for (int k = 0; k < keyLength; k++)
                    {
                        scores[k] = MathF.Exp(scores[k] - maxScore);
                        denominator += scores[k];
                    }

                    int outputBase = ((b * heads + h) * queryLength + q) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < keyLength; k++)
                        {
                            int valueBase = ((b * heads + h) * keyLength + k) * headDim;
                            sum += scores[k] * (float)value[valueBase + d];
                        }
                        output[outputBase + d] = sum / denominator;
                    }
                }
            }
        }
        return output;
    }
}
