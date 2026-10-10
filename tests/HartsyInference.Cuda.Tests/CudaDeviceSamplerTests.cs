using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Sampling;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The device sampler draws from the same distribution as the host <see cref="SamplerChain"/> (top-k, temperature, nucleus, min-p),
/// and consumes a fresh draw per launch.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaDeviceSamplerTests
{
    private readonly ITestOutputHelper _output;
    public CudaDeviceSamplerTests(ITestOutputHelper output) => _output = output;

    public static IEnumerable<object[]> Cases()
    {
        yield return [0.7f, 40, 1.0f, 0.0f];
        yield return [0.8f, 40, 0.9f, 0.0f];
        yield return [1.0f, 40, 1.0f, 0.1f];
        yield return [1.5f, 8, 0.95f, 0.05f];
        yield return [0.5f, 1, 1.0f, 0.0f];
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(Cases))]
    public void DeviceSampler_MatchesTheHostDistribution(float temperature, int topK, float topP, float minP)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir)) ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        const int vocab = 32000;
        Random rng = new(11);
        float[] logits = new float[vocab];
        for (int i = 0; i < vocab; i++) logits[i] = (float)(rng.NextDouble() * 6 - 3);
        for (int i = 0; i < 12; i++) logits[rng.Next(vocab)] += 5f + i * 0.4f;   // a few clear leaders

        SamplingOptions options = new() { Temperature = temperature, TopK = topK, TopP = topP, MinP = minP };
        float[] expected = new float[vocab];
        SamplerChain.FromOptions(options).Distribution((float[])logits.Clone(), [], expected);

        using CudaBackend cuda = new(0, ptxDir);
        Assert.True(cuda.DeviceSamplingSupported);
        using Tensor t = new(new TensorShape(1, 1, vocab), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < vocab; i++) p[i] = logits[i];
        ulong tokenId = cuda.AllocDeviceTokenId();
        ulong state = cuda.AllocDeviceRng(0xC0FFEEul, topK);
        try
        {
            const int draws = 30000;
            int[] counts = new int[vocab];
            HashSet<int> seen = [];
            for (int i = 0; i < draws; i++)
            {
                cuda.SampleTopKInto(tokenId, t, topK, temperature, topP, minP, state);
                int id = cuda.ReadDeviceTokenId(tokenId);
                Assert.InRange(id, 0, vocab - 1);
                counts[id]++;
                seen.Add(id);
            }
            double worst = 0;
            int support = 0;
            for (int i = 0; i < vocab; i++)
            {
                double pe = expected[i];
                if (pe > 0) support++;
                else Assert.True(counts[i] == 0, $"token {i} has host probability 0 but was drawn {counts[i]} times");
                double sigma = Math.Sqrt(Math.Max(pe * (1 - pe) / draws, 1e-12));
                worst = Math.Max(worst, Math.Abs(counts[i] / (double)draws - pe) / sigma);
            }
            _output.WriteLine($"T={temperature} k={topK} p={topP} minP={minP}: support {support} tokens, {seen.Count} drawn, worst deviation {worst:F2} sigma");
            Assert.True(worst < 6.0, $"device draws deviate from the host distribution by {worst:F2} sigma");
        }
        finally
        {
            cuda.FreeDeviceTokenId(tokenId);
            cuda.FreeDeviceRng(state);
        }
    }
}
