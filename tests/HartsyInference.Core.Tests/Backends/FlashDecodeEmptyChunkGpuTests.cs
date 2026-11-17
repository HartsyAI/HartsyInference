using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>A key whose score is exactly -FLT_MAX must still count as accumulated. The decode kernels used to test
/// "nothing accumulated yet" as m &lt;= -FLT_MAX, which also matches that score, so a second such key erased the first.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("SdpaKnobs")]
public sealed class FlashDecodeEmptyChunkGpuTests
{
    private readonly ITestOutputHelper _output;

    public FlashDecodeEmptyChunkGpuTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AllKeysAtFiniteMinimumScore_AverageTheirValues(bool device)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        foreach (bool q8 in new[] { false, true })
        {
            KnobStore.Set(EngineKnobs.QuantAtProducer, q8);
            KnobStore.Set(EngineKnobs.AttnCombineQ8, q8);
            try
            {
                using CudaBackend backend = new(0, PtxDir()) { EnableDp4aGemv = q8 };
                const int d = 64, heads = 8, length = 128;
                using Tensor q = new(new TensorShape(1, heads, 1, d), DType.F32);
                using Tensor k = new(new TensorShape(1, 1, length, d), DType.F32);
                using Tensor v = new(new TensorShape(1, 1, length, d), DType.F32);
                using Tensor actual = new(q.Shape, DType.F32);
                q.AsSpan<float>().Clear();
                k.AsSpan<float>().Clear();
                for (int h = 0; h < heads; h++) q.AsSpan<float>()[h * d] = 1f;
                for (int key = 0; key < length; key++)
                {
                    k.AsSpan<float>()[key * d] = -float.MaxValue;
                    v.AsSpan<float>().Slice(key * d, d).Fill(key % 2 == 0 ? 1f : 3f);
                }
                RunAttention(backend, actual, q, k, v, length, heads, device);
                backend.OffloadActivation(actual);
                foreach (float value in actual.AsReadOnlySpan<float>())
                    Assert.True(value == 2f, $"device={device}, q8={q8}: expected 2, got {value}");
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.QuantAtProducer);
                KnobStore.Clear(EngineKnobs.AttnCombineQ8);
            }
        }
    }

    private static void RunAttention(CudaBackend backend, Tensor output, Tensor q, Tensor k, Tensor v, int length, int group, bool device)
    {
        if (!device)
        {
            backend.FlashAttention(output, q, k, v, length, group, true, length - 1, 1f);
            return;
        }
        ulong pos = backend.AllocDevicePos();
        try
        {
            backend.WriteDevicePos(pos, length, length - 1);
            backend.FlashAttentionDev(output, q, k, v, 0, group, true, 0, 1f, pos, 0f, 0);
        }
        finally
        {
            backend.FreeDevicePos(pos);
        }
    }

    private static string PtxDir()
    {
        string output = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(output)) return output;
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
