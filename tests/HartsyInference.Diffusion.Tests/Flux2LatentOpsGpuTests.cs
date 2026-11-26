using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FLUX.2 latent layouts on CUDA against the CPU: the permuted strided views, the fused BatchNorm and the pack. The device
/// sync counter proves that the device-to-device layout ops read nothing back to the host. Returns without asserting when CUDA is not
/// available, so the suite stays green on a CPU-only machine.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class Flux2LatentOpsGpuTests
{
    private const int Batch = 2, Channels = 3, Patch = 2, GridH = 3, GridW = 5;
    private const int TokenChannels = Channels * Patch * Patch;
    private const float Eps = 1e-4f;

    [Fact]
    public void TheCudaLayoutsMatchTheCpuAndTheDeviceOpsReadNothingBack()
    {
        if (!CudaContext.IsAvailable())
        {
            return;
        }

        float[] patched = RandomValues(Batch * TokenChannels * GridH * GridW, 11);
        float[] mean = RandomValues(TokenChannels, 12);
        float[] var = RandomValues(TokenChannels, 13).Select(v => MathF.Abs(v) + 0.5f).ToArray();
        using CpuBackend cpu = new();
        using CudaBackend cuda = new(0, PtxDirectory());
        using Tensor cpuPatched = Make(new TensorShape(Batch, TokenChannels, GridH, GridW), patched);
        using Tensor cpuMean = Make(new TensorShape(TokenChannels), mean);
        using Tensor cpuVar = Make(new TensorShape(TokenChannels), var);

        // Host input, device output: the upload is host-to-device, so the layouts must match the CPU elementwise.
        using Tensor devicePatched = Make(new TensorShape(Batch, TokenChannels, GridH, GridW), patched);
        using Tensor cudaVae = Flux2LatentOps.Unpatchify(cuda, devicePatched, Channels, Patch);
        using Tensor cpuVae = Flux2LatentOps.Unpatchify(cpu, cpuPatched, Channels, Patch);
        AssertClose(cpuVae.AsSpan<float>().ToArray(), cudaVae.AsSpan<float>().ToArray());

        // Device-to-device chain: the sync counter must not move while the ops run.
        long before = cuda.GetD2hSyncCount();
        using Tensor cudaTokens = Flux2LatentOps.ToPackedTokens(cuda, cudaVae, cpuMean, cpuVar, Eps, Patch);
        using Tensor cudaNormalized = Flux2LatentOps.NormalizePatchified(cuda, cudaVae, cpuMean, cpuVar, Eps, Patch);
        using Tensor cudaBack = Flux2LatentOps.FromPackedTokens(cuda, cudaTokens, GridH, GridW, cpuMean, cpuVar, Eps, Channels, Patch);
        using Tensor cudaPacked = Flux2LatentOps.Pack(cuda, cudaNormalized);
        using Tensor cudaUnpacked = Flux2LatentOps.Unpack(cuda, cudaPacked, GridH * Patch, GridW * Patch);
        Assert.Equal(before, cuda.GetD2hSyncCount());

        // Downloads happen only now, for the comparison.
        using Tensor cpuTokens = Flux2LatentOps.ToPackedTokens(cpu, cpuVae, cpuMean, cpuVar, Eps, Patch);
        AssertClose(cpuTokens.AsSpan<float>().ToArray(), cudaTokens.AsSpan<float>().ToArray());
        AssertClose(cpuVae.AsSpan<float>().ToArray(), cudaBack.AsSpan<float>().ToArray());
        using Tensor cpuNormalized = Flux2LatentOps.NormalizePatchified(cpu, cpuVae, cpuMean, cpuVar, Eps, Patch);
        AssertClose(cpuNormalized.AsSpan<float>().ToArray(), cudaNormalized.AsSpan<float>().ToArray());
        Assert.Equal(cpuNormalized.Shape, cudaUnpacked.Shape);
    }

    private static float[] RandomValues(int count, int seed)
    {
        Random generator = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = generator.NextSingle() * 2f - 1f;
        return values;
    }

    private static Tensor Make(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-5f, $"element {i}: expected {expected[i]}, got {actual[i]}");
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }

        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
