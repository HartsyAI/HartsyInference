using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The CUDA and Vulkan per-key-channel delta-rule prefill against the host reference, output and carried state. Each
/// test reports itself as skipped when its device is absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SsmChannelDecayPrefillGpuTests
{
    private const float Tol = 1e-4f;

    private readonly ITestOutputHelper _output;
    public SsmChannelDecayPrefillGpuTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(3, 4, 3, 4, 2)]    // small: grouped key heads, one column per lane
    [InlineData(4, 2, 5, 64, 1)]   // wide key head: two columns per lane
    public void Cuda_MatchesHostReference(int seq, int hv, int sv, int sk, int repeat)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        Compare(backend, seq, hv, sv, sk, repeat);
        // Without the module the backend answers from the host reference and this test would pass without touching the device.
        CudaKernels? kernels = (CudaKernels?)typeof(CudaBackend).GetField("_kernels", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(backend);
        Assert.True(kernels?.HasChannelDecayPrefillKernel == true, "ssm_channel_decay_prefill_f32.ptx did not load; the device path did not run.");
    }

    [Theory]
    [InlineData(3, 4, 3, 4, 2)]
    [InlineData(4, 2, 5, 64, 1)]
    public void Vulkan_MatchesHostReference(int seq, int hv, int sv, int sk, int repeat)
    {
        if (!VulkanContext.IsAvailable()) { _output.WriteLine("SKIPPED: Vulkan unavailable"); return; }
        using VulkanBackend backend = new();
        Compare(backend, seq, hv, sv, sk, repeat);
    }

    private static void Compare(IBackend device, int seq, int hv, int sv, int sk, int repeat)
    {
        int keyHeads = hv / repeat;
        Random rng = new(seq * 131 + sk);
        using Tensor q = Make(rng, seq * keyHeads * sk, -1f, 1f, seq, keyHeads * sk);
        using Tensor k = Make(rng, seq * keyHeads * sk, -1f, 1f, seq, keyHeads * sk);
        using Tensor v = Make(rng, seq * hv * sv, -1f, 1f, seq, hv * sv);
        using Tensor logDecay = Make(rng, seq * hv * sk, -2f, 0f, seq, hv * sk);
        using Tensor beta = Make(rng, seq * hv, -2f, 2f, seq, hv);
        float[] state0 = Floats(rng, hv * sv * sk, -0.5f, 0.5f);

        using Tensor expectedOut = new(new TensorShape(seq, hv * sv), DType.F32);
        using Tensor expectedState = Make(state0, hv, sv, sk);
        GatedDeltaNetReference.ChannelDecayScan(expectedOut, expectedState, q, k, v, logDecay, beta, hv, sv, sk, repeat);

        using Tensor actualOut = new(new TensorShape(seq, hv * sv), DType.F32);
        using Tensor actualState = Make(state0, hv, sv, sk);
        device.SsmChannelDecayPrefill(actualOut, actualState, q, k, v, logDecay, beta, hv, sv, sk, repeat);
        device.OffloadActivation(actualOut);
        device.OffloadActivation(actualState);

        AssertClose(expectedOut.AsReadOnlySpan<float>(), actualOut.AsReadOnlySpan<float>(), "output");
        AssertClose(expectedState.AsReadOnlySpan<float>(), actualState.AsReadOnlySpan<float>(), "state");
    }

    private static void AssertClose(ReadOnlySpan<float> want, ReadOnlySpan<float> got, string what)
    {
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= Tol, $"{what}[{i}] want {want[i]} got {got[i]}");
    }

    private static Tensor Make(Random rng, int n, float lo, float hi, params long[] shape)
        => Make(Floats(rng, n, lo, hi), shape);

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static float[] Floats(Random rng, int n, float lo, float hi)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return values;
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        return Path.Combine(AppContext.BaseDirectory, "Ptx");
    }
}
