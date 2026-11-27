using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat recipe bodies on a device against the CPU backend: the same tiny checkpoints and stand-ins, run once on the CPU and once on
/// CUDA or Vulkan, and the outputs compared. The bodies are the shared helpers of <see cref="LongCatRecipeEndToEndTests.RunBaseClip"/> and
/// <see cref="LongCatAvatarRecipeEndToEndTests.Run"/>; nothing is copied. Each test runs only where its device opens and is reported as skipped
/// otherwise.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatRecipeGpuParityTests
{
    /// <summary>The largest allowed difference, as a fraction of the CPU output's largest magnitude (floats), and in 8-bit levels for frames.</summary>
    private const double RelativeTolerance = 2e-3;
    private const int FrameLevels = 2;

    [CudaDeviceFact]
    public void Cuda_BaseRecipe_MatchesCpu() => BaseParity(() => GpuDevices.OpenCuda(), int8: false);

    [CudaDeviceFact]
    public void Cuda_BaseRecipeInt8_MatchesCpu() => BaseParity(() => GpuDevices.OpenCuda(), int8: true);

    [VulkanDeviceFact]
    public void Vulkan_BaseRecipe_MatchesCpu() => BaseParity(() => GpuDevices.OpenVulkan(), int8: false);

    [VulkanDeviceFact]
    public void Vulkan_BaseRecipeInt8_MatchesCpu() => BaseParity(() => GpuDevices.OpenVulkan(), int8: true);

    [CudaDeviceFact]
    public void Cuda_AvatarChain15_MatchesCpu() => AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: false);

    [CudaDeviceFact]
    public void Cuda_AvatarChain16_MatchesCpu() => AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: true);

    [VulkanDeviceFact]
    public void Vulkan_AvatarChain15_MatchesCpu() => AvatarParity(() => GpuDevices.OpenVulkan(), onePointSix: false);

    [VulkanDeviceFact]
    public void Vulkan_AvatarChain16_MatchesCpu() => AvatarParity(() => GpuDevices.OpenVulkan(), onePointSix: true);

    private static void BaseParity(Func<IBackend> device, bool int8)
    {
        VideoGenerationResult cpu = LongCatRecipeEndToEndTests.RunBaseClip(int8, () => new CpuBackend());
        VideoGenerationResult gpu = LongCatRecipeEndToEndTests.RunBaseClip(int8, device);

        Assert.Equal(cpu.Frames.Count, gpu.Frames.Count);
        int worst = 0;
        for (int f = 0; f < cpu.Frames.Count; f++)
        {
            Assert.Equal(cpu.Frames[f].Rgb.Length, gpu.Frames[f].Rgb.Length);
            for (int i = 0; i < cpu.Frames[f].Rgb.Length; i++)
                worst = Math.Max(worst, Math.Abs(cpu.Frames[f].Rgb[i] - gpu.Frames[f].Rgb[i]));
        }
        Assert.True(worst <= FrameLevels, $"largest 8-bit frame difference {worst} levels, allowed {FrameLevels}");
    }

    private static void AvatarParity(Func<IBackend> device, bool onePointSix)
    {
        List<(int Frames, float[] Values)> cpu = LongCatAvatarRecipeEndToEndTests.Run(onePointSix, audioSeed: 1).Vae.Decoded;
        List<(int Frames, float[] Values)> gpu = LongCatAvatarRecipeEndToEndTests.Run(onePointSix, audioSeed: 1, device).Vae.Decoded;

        Assert.Equal(cpu.Count, gpu.Count);
        for (int segment = 0; segment < cpu.Count; segment++)
        {
            float scale = Math.Max(1f, cpu[segment].Values.Max(Math.Abs));
            float worst = 0f;
            for (int i = 0; i < cpu[segment].Values.Length; i++)
                worst = Math.Max(worst, Math.Abs(cpu[segment].Values[i] - gpu[segment].Values[i]));
            Assert.True(worst <= RelativeTolerance * scale,
                $"segment {segment}: largest difference {worst:G4}, allowed {RelativeTolerance * scale:G4} (scale {scale:G4})");
        }
    }
}
