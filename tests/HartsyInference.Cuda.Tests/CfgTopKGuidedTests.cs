using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>MiniMax Music 3's guided semantic logits on the device (lm_cfg_topk_guided.cu) against the host BlendGuided it
/// replaces: same threshold, same candidate rule and same operation order, so the outputs must match bit for bit. Inputs are
/// quantized to a coarse grid, so many entries tie at the threshold and the tie rule is exercised.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CfgTopKGuidedTests
{
    private const int SemanticVocab = 16385;   // 16384 codes plus end-of-audio
    private const float CfgScale = 1.5f;

    private readonly ITestOutputHelper _output;

    public CfgTopKGuidedTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(SemanticVocab, 50, 11)]   // the MiniMax semantic head with its top-50 candidate set
    [InlineData(1000, 1, 12)]             // single candidate: the threshold is the maximum
    [InlineData(1000, 64, 13)]            // the largest K the sampler accepts
    public void CfgTopKGuided_MatchesHostBlendGuided_BitForBit(int count, int topK, int seed)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        Random rng = new(seed);
        Tensor cond = new(new TensorShape(count), DType.F32);
        Tensor uncond = new(new TensorShape(count), DType.F32);
        Span<float> c = cond.AsSpan<float>();
        Span<float> u = uncond.AsSpan<float>();
        for (int i = 0; i < count; i++)
        {
            c[i] = rng.Next(-40, 41) * 0.25f;      // coarse grid: many exact ties
            u[i] = rng.Next(-40, 41) * 0.25f;
        }
        float threshold = KthLargest(c, topK);
        float[] expected = new float[count];
        for (int i = 0; i < count; i++)
        {
            expected[i] = c[i] < threshold ? float.NegativeInfinity : u[i] + ((c[i] - u[i]) * CfgScale);
        }

        using CudaBackend backend = new(0, ptxDir);
        IBackend b = backend;
        Tensor output = new(new TensorShape(count), DType.F32);
        b.CfgTopKGuided(output, cond, uncond, topK, CfgScale);
        ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>();

        int mismatches = 0;
        int first = -1;
        for (int i = 0; i < count; i++)
        {
            if (BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i]))
            {
                continue;
            }
            if (first < 0)
            {
                first = i;
            }
            mismatches++;
        }
        int candidates = 0;
        for (int i = 0; i < count; i++)
        {
            if (!float.IsNegativeInfinity(expected[i]))
            {
                candidates++;
            }
        }
        _output.WriteLine($"count={count} topK={topK} threshold={threshold} candidates={candidates} mismatches={mismatches}");
        Assert.True(mismatches == 0, $"guided logits differ at {mismatches} entries; first at {first}.");

        output.Dispose();
        cond.Dispose();
        uncond.Dispose();
    }

    /// <summary>The K-th largest value, by the same insertion scheme as the host sampler's KthLargest.</summary>
    private static float KthLargest(ReadOnlySpan<float> values, int k)
    {
        float[] top = new float[k];
        Array.Fill(top, float.NegativeInfinity);
        for (int i = 0; i < values.Length; i++)
        {
            float value = values[i];
            if (value <= top[k - 1])
            {
                continue;
            }
            int position = k - 1;
            while (position > 0 && top[position - 1] < value)
            {
                top[position] = top[position - 1];
                position--;
            }
            top[position] = value;
        }
        return top[k - 1];
    }
}
