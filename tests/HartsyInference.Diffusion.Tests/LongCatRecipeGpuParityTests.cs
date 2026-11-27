using HartsyInference.Core.Tensors;
using HartsyInference.Core.Configuration;
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

    /// <summary>The INT8 lanes' bound, as a fraction of scale, written before the rerun that uses it. Derived from
    /// <c>LongCatInt8LinearOpTests</c>: over all 40 avatar linears the CUDA INT8 output is 3.7 to 4.0e-3 relative RMS from exact q·s with the
    /// same F32 activations, and matches the per-row int8 activation model to 5e-8, so the gap is the activation quantization of the resident path
    /// (a stated precision limit). The largest segment figure measured on the 1.6 chain is 0.80% of scale (CPU as widened), and this bound is that
    /// figure plus about 25%. The dense lanes keep <see cref="RelativeTolerance"/>.</summary>
    private const double Int8RelativeTolerance = 1.0e-2;
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

    /// <summary>Diagnosis of the CUDA 1.6 chain: the same case with exact GEMMs (HighPrecisionGemm on), to see whether the difference is TF32 accumulation.</summary>
    [CudaDeviceFact]
    public void Cuda_AvatarChain16_HighPrecisionGemm_MatchesCpu() => AvatarParity(() =>
    {
        IBackend backend = GpuDevices.OpenCuda();
        backend.HighPrecisionGemm = true;
        return backend;
    }, onePointSix: true);

    /// <summary>Diagnosis: the 1.6 INT8 chain with F32 DiT activations (numerics.ditF16 = false, pinned by the reference profile) instead of the CUDA default F16.</summary>
    [CudaDeviceFact]
    public void Cuda_AvatarChain16_F32Activations_MatchesCpu()
    {
        using IDisposable scope = KnobProfile.Create("longcat-ditf16-off").With(EngineKnobs.DitF16, false).Push();
        AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: true);
    }

    /// <summary>Diagnosis: the 1.6 INT8 chain with the dp4a decode GEMV off (numerics.dp4aOn = false, pinned by the reference profile). That GEMV quantizes
    /// activations to Q8_1, which the CPU widened path does not, so a gap here names activation quantization.</summary>
    [CudaDeviceFact]
    public void Cuda_AvatarChain16_Dp4aOff_MatchesCpu()
    {
        using IDisposable scope = KnobProfile.Create("longcat-dp4a-off").With(EngineKnobs.Dp4aOn, false).Push();
        AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: true);
    }

    /// <summary>Diagnosis: the 1.5 chain with INT8 weights and the single DMD LoRA (no mix), to split INT8 from the 1.6 LoRA mix.</summary>
    [CudaDeviceFact]
    public void Cuda_AvatarChain15_Int8Weights_MatchesCpu() => AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: false, int8: true);

    /// <summary>Diagnosis: the 1.6 chain with dense (not INT8) weights and the same two LoRAs, to split INT8 from the LoRA mix.</summary>
    [CudaDeviceFact]
    public void Cuda_AvatarChain16_DenseWeights_MatchesCpu() => AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: true, int8: false);

    /// <summary>Diagnosis of the INT8 gap: the CPU side runs the 1.6 chain on exact F32 q*s weights (the INT8 values times their scale, no BF16
    /// truncation), and the CUDA side runs the INT8 release. If the gap closes, the CPU widening's BF16 truncation is the difference.</summary>
    [CudaDeviceFact]
    public void Cuda_AvatarChain16_CpuExactDequant_MatchesCuda() => AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: true, int8: true,
        cpuDense: ExactDequantWeights());

    /// <summary>The 1.6 dense weights with every linear replaced by its INT8 values times their scale, in exact F32 (same seed as the INT8 release).</summary>
    private static Dictionary<string, Tensor> ExactDequantWeights(bool upstream = false)
    {
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(LongCatTinyAvatarRelease.Config, seed: 71);
        Dictionary<string, Tensor> quantized = LongCatTestData.QuantizePerRow(weights);
        Dictionary<string, Tensor> dense = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in weights)
        {
            if (key.EndsWith(".weight", StringComparison.Ordinal) && quantized.TryGetValue(key[..^".weight".Length] + ".weight_int8", out Tensor? q)
                && quantized.TryGetValue(key[..^".weight".Length] + ".weight_scale", out Tensor? scale))
            {
                Tensor exact = new(tensor.Shape, DType.F32);
                long cols = tensor.Shape[1];
                ReadOnlySpan<sbyte> values = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, sbyte>(q.AsSpan<byte>());
                ReadOnlySpan<float> scales = scale.AsSpan<float>();
                Span<float> outValues = exact.AsSpan<float>();
                for (long r = 0; r < tensor.Shape[0]; r++)
                    for (long c = 0; c < cols; c++)
                    {
                        float qv = values[(int)(r * cols + c)], sc = scales[(int)r];
                        // Upstream QuantizedLinear.forward: weight = weight_int8.to(x.dtype) * weight_scale.to(x.dtype), with x.dtype BF16.
                        outValues[(int)(r * cols + c)] = upstream ? Bf16Rne(Bf16Rne(sc) * qv) : qv * sc;
                    }
                dense[key] = exact;
            }
            else
            {
                dense[key] = tensor;
            }
        }
        return dense;
    }

    /// <summary>Diagnosis: the CPU side runs the 1.6 chain on the weights upstream's QuantizedLinear forms, bf16(bf16(scale) * q) in BF16 (the
    /// reference's dtype rule), against the CUDA INT8 release.</summary>
    [CudaDeviceFact]
    public void Cuda_AvatarChain16_CpuUpstreamDequant_MatchesCuda() => AvatarParity(() => GpuDevices.OpenCuda(), onePointSix: true, int8: true,
        cpuDense: ExactDequantWeights(upstream: true));

    private static float Bf16Rne(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint lsb = (bits >> 16) & 1u;
        return BitConverter.UInt32BitsToSingle((bits + 0x7FFFu + lsb) & 0xFFFF0000u);
    }

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

    private static void AvatarParity(Func<IBackend> device, bool onePointSix, bool? int8 = null, Dictionary<string, Tensor>? cpuDense = null)
    {
        // INT8 weights on the device are the INT8 lane; a dense checkpoint (or cpuDense, which is dense) is the dense lane.
        double tolerance = (int8 ?? onePointSix) && cpuDense is null ? Int8RelativeTolerance : RelativeTolerance;
        List<(int Frames, float[] Values)> cpu = LongCatAvatarRecipeEndToEndTests.Run(onePointSix, audioSeed: 1, int8: int8, dense: cpuDense).Vae.Decoded;
        List<(int Frames, float[] Values)> gpu = LongCatAvatarRecipeEndToEndTests.Run(onePointSix, audioSeed: 1, device, int8).Vae.Decoded;

        Assert.Equal(cpu.Count, gpu.Count);
        for (int segment = 0; segment < cpu.Count; segment++)
        {
            float scale = Math.Max(1f, cpu[segment].Values.Max(Math.Abs));
            float worst = 0f;
            for (int i = 0; i < cpu[segment].Values.Length; i++)
                worst = Math.Max(worst, Math.Abs(cpu[segment].Values[i] - gpu[segment].Values[i]));
            Assert.True(worst <= tolerance * scale,
                $"segment {segment}: largest difference {worst:G4}, allowed {tolerance * scale:G4} (scale {scale:G4}, tolerance {tolerance:G3})");
        }
    }
}
