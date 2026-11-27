using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Tests.Shared;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A weight the LongCat FP8 selection packs never reaches a native fp8 GEMM, which would quantize its activation: on a device that has one,
/// the Linear gives the 16-bit-cast route's bytes with the native route switched on. Skipped, never passed, on a device without a native route,
/// where there is nothing to keep off.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatFp8FfnRouteTests
{
    private const int Rows = 8, In = 128, Out = 256;

    [CudaNativeFp8Fact]
    public void CudaKeepsAPackedWeightOffItsNativeFp8Gemm()
    {
        using CudaBackend cuda = GpuDevices.OpenCuda();
        Check(cuda, native => cuda.EnableNativeFp8Gemm = native);
    }

    [VulkanNativeFp8Fact]
    public void VulkanKeepsAPackedWeightOffItsFp8CooperativeMatrices()
    {
        using VulkanBackend vulkan = GpuDevices.OpenVulkan();
        Check(vulkan, native => vulkan.EnableFp8Linear = native);
    }

    private static void Check(IBackend backend, Action<bool> setNative)
    {
        List<Tensor> owned = [];
        using Tensor input = LongCatTestData.Random(new TensorShape(Rows, In), 5);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        using Tensor source = LongCatTestData.Random(new TensorShape(Out, In), 6, 0.05f);
        weights["blocks.6.ffn.w1.weight"] = source.CastTo(DType.BF16);
        weights["blocks.6.ffn.w3.weight"] = source.CastTo(DType.BF16);
        owned.AddRange(weights.Values);
        try
        {
            LongCatFp8FfnConverter.Convert(weights, owned, [6]);
            Tensor weight = weights["blocks.6.ffn.w1.weight"];

            float[] Run(bool native, bool marked)
            {
                setNative(native);
                weight.SkipNativeFp8Gemm = marked;
                using Tensor output = new(new TensorShape(Rows, Out), DType.F32);
                backend.Linear(output, input, weight, null);
                return output.AsReadOnlySpan<float>().ToArray();
            }

            float[] cast = Run(native: false, marked: true);
            float[] marked = Run(native: true, marked: true);
            float[] unmarked = Run(native: true, marked: false);

            Assert.Equal(cast, marked);
            // The control: without the mark this device does take its native route, so the equality above shows something.
            Assert.NotEqual(cast, unmarked);
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }
}

/// <summary>A fact reported as skipped unless the CUDA device has a native fp8 GEMM (SM 8.9 or later).</summary>
internal sealed class CudaNativeFp8FactAttribute : FactAttribute
{
    public CudaNativeFp8FactAttribute()
    {
        if (GpuDevices.CudaFailure is { } failure)
        {
            Skip = $"SKIPPED: no usable CUDA device ({failure}).";
            return;
        }
        using CudaBackend cuda = GpuDevices.OpenCuda();
        if (!cuda.NativeFp8Gemm)
            Skip = "SKIPPED: this CUDA device has no native fp8 GEMM.";
    }
}

/// <summary>A fact reported as skipped unless the Vulkan device runs fp8 Linears on cooperative matrices.</summary>
internal sealed class VulkanNativeFp8FactAttribute : FactAttribute
{
    public VulkanNativeFp8FactAttribute()
    {
        if (GpuDevices.VulkanFailure is { } failure)
        {
            Skip = $"SKIPPED: no usable Vulkan device ({failure}).";
            return;
        }
        using VulkanBackend vulkan = GpuDevices.OpenVulkan();
        if (!vulkan.EnableFp8Linear)
            Skip = "SKIPPED: this Vulkan device has no fp8 cooperative matrices.";
    }
}
