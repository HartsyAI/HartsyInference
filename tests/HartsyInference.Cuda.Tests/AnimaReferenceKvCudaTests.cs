using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Tests.Common.AnimaSyntheticWeights;

namespace HartsyInference.Cuda.Tests;

/// <summary>The reference K/V join on CUDA against the CPU: a tiny Anima transformer builds a reference cache, and a forward joins it into
/// every block. The device output must match the host output. The inputs are written on the device, so the comparison does not depend on
/// a host upload of them.</summary>
[Collection("CudaSerial")]
public sealed unsafe class AnimaReferenceKvCudaTests
{
    private const int TargetH = 6, TargetW = 10, RefH = 4, RefW = 6, Text = 7;

    private readonly ITestOutputHelper _output;

    public AnimaReferenceKvCudaTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ReferenceJoin_CudaMatchesCpu()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA device");
            return;
        }
        AnimaConfig config = TinyConfig();
        using AnimaTransformer transformer = new(config);
        transformer.LoadWeights(Weights(config));
        ReferencePlacement placement = new(1, 1.0f, 0);
        Tensor latentHost = Ramp(new TensorShape(1, 2, TargetH, TargetW), 0.3f);
        Tensor textHost = Ramp(new TensorShape(1, Text, TextWidth), 0.7f);
        Tensor referenceHost = Ramp(new TensorShape(1, 2, RefH, RefW), 1.1f);

        using CpuBackend cpu = new();
        using Tensor expected = Join(cpu, transformer, latentHost, textHost, referenceHost, placement);
        latentHost.Dispose();
        textHost.Dispose();
        referenceHost.Dispose();

        using CudaBackend cuda = new(0, PtxDir());
        using Tensor latent = new(new TensorShape(1, 2, TargetH, TargetW), DType.F32);
        using Tensor text = new(new TensorShape(1, Text, TextWidth), DType.F32);
        using Tensor reference = new(new TensorShape(1, 2, RefH, RefW), DType.F32);
        // Device-authored inputs, so a host round trip inside the join would show up.
        using Tensor latentSource = Ramp(latent.Shape, 0.3f);
        using Tensor textSource = Ramp(text.Shape, 0.7f);
        using Tensor referenceSource = Ramp(reference.Shape, 1.1f);
        cuda.Scale(latent, latentSource, 1.0f);
        cuda.Scale(text, textSource, 1.0f);
        cuda.Scale(reference, referenceSource, 1.0f);
        cuda.PreloadWeights(transformer.EnumerateWeights());
        cuda.Sync();

        using Tensor actual = Join(cuda, transformer, latent, text, reference, placement);
        cuda.Sync();
        transformer.ReleaseDeviceCache(cuda);

        AssertClose(expected, actual, cuda, tolerance: 1e-3f);
    }

    /// <summary>Caches the reference on the backend, then runs the target forward with that cache joined into every block.</summary>
    private static Tensor Join(IBackend backend, AnimaTransformer transformer, Tensor latent, Tensor text, Tensor reference,
        ReferencePlacement placement)
    {
        using ReferenceKvCache cache = transformer.BuildReferenceCache(backend, reference, TargetH / 2, placement, identity: 7);
        return transformer.Forward(backend, latent, 0.625f, text, null, cache);
    }

    private static void AssertClose(Tensor expected, Tensor actual, CudaBackend backend, float tolerance)
    {
        backend.Sync();
        Assert.Equal(expected.Shape, actual.Shape);
        int count = (int)expected.ElementCount;
        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(e[i] - a[i]) <= tolerance * MathF.Max(1f, MathF.Abs(e[i])), $"element {i}: expected {e[i]}, got {a[i]}");
        }
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }
}
