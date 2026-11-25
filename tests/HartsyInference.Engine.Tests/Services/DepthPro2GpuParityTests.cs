using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.DepthPro2;
using HartsyInference.Vulkan;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>
/// Depth Pro 2 S and its focal estimator on a GPU backend against the CPU F32 lane, on the saved inputs of
/// <c>tests/python-reference/dump_depth_pro2.py</c>: every backbone tap and head stage, the focal length and the metric depth of one image,
/// three views run jointly, and buffered and unbuffered video. Needs <c>HARTSY_DEPTHPRO2_WEIGHTS</c>, <c>HARTSY_DEPTHPRO2_FOCAL_WEIGHTS</c>
/// and <c>HARTSY_DEPTHPRO2_REF</c> (the dump's <c>meta.json</c>) and the device; otherwise each test is reported as skipped.
/// </summary>
[Trait("Category", "GpuIntegration")]
[Collection(DepthPro2GpuKnobsCollection.Name)]
public sealed class DepthPro2GpuParityTests(ITestOutputHelper output)
{
    private const string WeightsVariable = "HARTSY_DEPTHPRO2_WEIGHTS";
    private const string FocalVariable = "HARTSY_DEPTHPRO2_FOCAL_WEIGHTS";
    private const string ReferenceVariable = "HARTSY_DEPTHPRO2_REF";
    private const double Tolerance = 1e-4;
    private const double Tf32Tolerance = 5e-3;

    private static readonly Lazy<string?> CudaFailure = new(() => Probe(Gpu.Cuda));
    private static readonly Lazy<string?> VulkanFailure = new(() => Probe(Gpu.Vulkan));

    [DepthPro2GpuFact(Gpu.CudaFp32)]
    public void CudaFp32_Image_MatchesCpu_256x384() => Image(Gpu.CudaFp32, "a");

    [DepthPro2GpuFact(Gpu.CudaFp32)]
    public void CudaFp32_Image_MatchesCpu_Padded250x370() => Image(Gpu.CudaFp32, "b");

    [DepthPro2GpuFact(Gpu.CudaFp32)]
    public void CudaFp32_ThreeViews_MatchCpu() => Views(Gpu.CudaFp32);

    [DepthPro2GpuFact(Gpu.CudaFp32)]
    public void CudaFp32_BufferedVideo_MatchesCpu() => Video(Gpu.CudaFp32, DepthPro2VideoMode.Buffered);

    [DepthPro2GpuFact(Gpu.CudaFp32)]
    public void CudaFp32_UnbufferedVideo_MatchesCpu() => Video(Gpu.CudaFp32, DepthPro2VideoMode.Unbuffered);

    // The default CUDA lane multiplies F32 in TF32 (10-bit mantissa), so it is held to the TF32 bound, not the F32 one.
    [DepthPro2GpuFact(Gpu.Cuda)]
    public void Cuda_Tf32_Image_WithinTf32Bound_256x384() => Image(Gpu.Cuda, "a");

    [DepthPro2GpuFact(Gpu.Cuda)]
    public void Cuda_Tf32_Image_WithinTf32Bound_Padded250x370() => Image(Gpu.Cuda, "b");

    [DepthPro2GpuFact(Gpu.Cuda)]
    public void Cuda_Tf32_BufferedVideo_WithinTf32Bound() => Video(Gpu.Cuda, DepthPro2VideoMode.Buffered);

    [DepthPro2GpuFact(Gpu.Vulkan)]
    public void Vulkan_Image_MatchesCpu_256x384() => Image(Gpu.Vulkan, "a");

    [DepthPro2GpuFact(Gpu.Vulkan)]
    public void Vulkan_Image_MatchesCpu_Padded250x370() => Image(Gpu.Vulkan, "b");

    [DepthPro2GpuFact(Gpu.Vulkan)]
    public void Vulkan_ThreeViews_MatchCpu() => Views(Gpu.Vulkan);

    [DepthPro2GpuFact(Gpu.Vulkan)]
    public void Vulkan_BufferedVideo_MatchesCpu() => Video(Gpu.Vulkan, DepthPro2VideoMode.Buffered);

    [DepthPro2GpuFact(Gpu.Vulkan)]
    public void Vulkan_UnbufferedVideo_MatchesCpu() => Video(Gpu.Vulkan, DepthPro2VideoMode.Unbuffered);

    // Each stage on both backends, then the whole model twice on the device: the first run uploads, the second is timed per stage.
    private void Image(Gpu gpu, string name)
    {
        (byte[] rgb, int height, int width) = ReferenceRgb(name);
        using SafeTensorsLoader depthLoader = Load(WeightsVariable);
        using SafeTensorsLoader focalLoader = Load(FocalVariable);
        DepthPro2Model model = new(new DepthPro2Network(DepthPro2Preset.Small, depthLoader.GetAllTensors()));
        DepthPro2FocalEstimator estimator = new(focalLoader.GetAllTensors());
        using CpuBackend cpu = new();
        using IBackend device = Open(gpu);

        Dictionary<string, float[]> expected = Stages(cpu, model.Network, estimator, rgb, width, height);
        Dictionary<string, float[]> actual = Stages(device, model.Network, estimator, rgb, width, height);
        foreach ((string stage, float[] values) in expected)
        {
            double stageError = RelRms(actual[stage], values);
            output.WriteLine($"{gpu} {name} {stage} relRMS {stageError:E2}");
            Assert.True(stageError <= Bound(gpu), $"{stage} relRMS {stageError:E2}");
        }

        DepthPro2Prediction cpuResult = model.Predict(cpu, [rgb], width, height, focalLengthPx: null, estimator);
        DepthPro2Prediction gpuResult = model.Predict(device, [rgb], width, height, focalLengthPx: null, estimator);
        Timed(device, model.Network, estimator, rgb, width, height, $"{gpu} {name}");
        double focalError = Math.Abs(gpuResult.FocalLengthPx - cpuResult.FocalLengthPx) / cpuResult.FocalLengthPx;
        output.WriteLine($"{gpu} {name} f_px {gpuResult.FocalLengthPx:F4} vs CPU {cpuResult.FocalLengthPx:F4} (rel {focalError:E2})");
        Assert.True(focalError <= Bound(gpu), $"f_px rel {focalError:E2}");
        CompareMaps($"{gpu} {name}", gpuResult.Maps, cpuResult.Maps, Bound(gpu));
    }

    // Three overlapping crops of the 256x384 input attend to each other as one set.
    private void Views(Gpu gpu)
    {
        (byte[] rgb, int height, int width) = ReferenceRgb("a");
        const int viewH = 224, viewW = 352;
        byte[][] views = [Crop(rgb, width, 0, 0, viewW, viewH), Crop(rgb, width, 16, 16, viewW, viewH), Crop(rgb, width, 32, 32, viewW, viewH)];
        using SafeTensorsLoader depthLoader = Load(WeightsVariable);
        using SafeTensorsLoader focalLoader = Load(FocalVariable);
        DepthPro2Model model = new(new DepthPro2Network(DepthPro2Preset.Small, depthLoader.GetAllTensors()));
        DepthPro2FocalEstimator estimator = new(focalLoader.GetAllTensors());
        using CpuBackend cpu = new();
        using IBackend device = Open(gpu);

        DepthPro2Prediction cpuResult = model.Predict(cpu, views, viewW, viewH, focalLengthPx: null, estimator);
        DepthPro2Prediction gpuResult = model.Predict(device, views, viewW, viewH, focalLengthPx: null, estimator);
        double focalError = Math.Abs(gpuResult.FocalLengthPx - cpuResult.FocalLengthPx) / cpuResult.FocalLengthPx;
        output.WriteLine($"{gpu} views f_px {gpuResult.FocalLengthPx:F4} vs CPU {cpuResult.FocalLengthPx:F4} (rel {focalError:E2})");
        Assert.True(focalError <= Bound(gpu), $"f_px rel {focalError:E2}");
        CompareMaps($"{gpu} views", gpuResult.Maps, cpuResult.Maps, Bound(gpu));
    }

    // Eight sliding 128x160 crops: buffered clips of 4 overlapping by 2 (the last clip pulled back), or a 3-frame ring that wraps.
    private void Video(Gpu gpu, DepthPro2VideoMode mode)
    {
        (byte[] rgb, int _, int width) = ReferenceRgb("a");
        const int frameH = 128, frameW = 160, count = 8;
        byte[][] frames = new byte[count][];
        for (int i = 0; i < count; i++) frames[i] = Crop(rgb, width, 24 * i, 8 * i, frameW, frameH);
        using SafeTensorsLoader depthLoader = Load(WeightsVariable);
        DepthPro2Network network = new(DepthPro2Preset.Small, depthLoader.GetAllTensors());
        using CpuBackend cpu = new();
        using IBackend device = Open(gpu);
        (int window, int overlap) = mode == DepthPro2VideoMode.Buffered ? (4, 2) : (3, 0);

        List<DepthPro2Map> expected = RunVideo(network, cpu, frames, frameW, frameH, window, overlap, mode);
        List<DepthPro2Map> actual = RunVideo(network, device, frames, frameW, frameH, window, overlap, mode);
        Assert.Equal(count, actual.Count);
        CompareMaps($"{gpu} {mode.ToString().ToLowerInvariant()} video", actual, expected, Bound(gpu));
    }

    private static List<DepthPro2Map> RunVideo(DepthPro2Network network, IBackend backend, byte[][] frames, int width, int height, int window,
        int overlap, DepthPro2VideoMode mode)
    {
        using DepthPro2VideoStream stream = new(network, backend, width, height, 300.0, window, overlap, mode);
        List<DepthPro2Map> maps = [];
        foreach (byte[] frame in frames) maps.AddRange(stream.Step(frame));
        maps.AddRange(stream.Flush());
        return maps;
    }

    // The padded input, the four backbone taps, every head stage and the estimator's stages, copied to the host.
    private static Dictionary<string, float[]> Stages(IBackend backend, DepthPro2Network network, DepthPro2FocalEstimator estimator, byte[] rgb,
        int width, int height)
    {
        Dictionary<string, float[]> stages = new(StringComparer.Ordinal);
        using Tensor pixels = DepthPro2Preprocessor.Normalize(rgb, width, height);
        using Tensor? padded = DepthPro2Network.PadToMultiple(backend, pixels);
        Tensor input = padded ?? pixels;
        stages["padded"] = input.AsReadOnlySpan<float>().ToArray();
        Tensor[] taps = network.EncodeTaps(backend, input);
        try
        {
            for (int i = 0; i < taps.Length; i++) stages[$"tap{network.Preset.ReassembledBlocks[i]}"] = taps[i].AsReadOnlySpan<float>().ToArray();
            int patch = network.Preset.Backbone.PatchSize, prefix = 1 + network.Preset.Backbone.NumRegisterTokens;
            using Tensor map = network.Head.Forward(backend, taps, prefix, (int)input.Shape[2] / patch, (int)input.Shape[3] / patch,
                (stage, tensor) => stages[stage] = tensor.AsReadOnlySpan<float>().ToArray());
            stages["inverse_depth"] = map.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            foreach (Tensor tap in taps) tap.Dispose();
        }

        estimator.Estimate(backend, pixels, (stage, tensor) => stages[stage] = tensor.AsReadOnlySpan<float>().ToArray());
        return stages;
    }

    // Wall time of the backbone, the head and the estimator on a warm device; reading a result back waits for the device.
    private void Timed(IBackend backend, DepthPro2Network network, DepthPro2FocalEstimator estimator, byte[] rgb, int width, int height,
        string label)
    {
        using Tensor pixels = DepthPro2Preprocessor.Normalize(rgb, width, height);
        using Tensor? padded = DepthPro2Network.PadToMultiple(backend, pixels);
        Tensor input = padded ?? pixels;
        Stopwatch clock = Stopwatch.StartNew();
        Tensor[] taps = network.EncodeTaps(backend, input);
        try
        {
            _ = taps[^1].AsReadOnlySpan<float>()[0];
            double encoder = clock.Elapsed.TotalMilliseconds;
            clock.Restart();
            int patch = network.Preset.Backbone.PatchSize, prefix = 1 + network.Preset.Backbone.NumRegisterTokens;
            using Tensor map = network.Head.Forward(backend, taps, prefix, (int)input.Shape[2] / patch, (int)input.Shape[3] / patch);
            _ = map.AsReadOnlySpan<float>()[0];
            double head = clock.Elapsed.TotalMilliseconds;
            clock.Restart();
            estimator.Estimate(backend, pixels);
            double focal = clock.Elapsed.TotalMilliseconds;
            output.WriteLine($"{label} warm: backbone {encoder:F1} ms, head {head:F1} ms, focal estimator {focal:F1} ms");
        }
        finally
        {
            foreach (Tensor tap in taps) tap.Dispose();
        }
    }

    private void CompareMaps(string label, IReadOnlyList<DepthPro2Map> actual, IReadOnlyList<DepthPro2Map> expected, double bound)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int f = 0; f < expected.Count; f++)
        {
            float[] gpuDepth = actual[f].DepthMeters, cpuDepth = expected[f].DepthMeters;
            Assert.Equal(cpuDepth.Length, gpuDepth.Length);
            double diff2 = 0, ref2 = 0;
            int infinite = 0;
            for (int i = 0; i < cpuDepth.Length; i++)
            {
                Assert.Equal(float.IsPositiveInfinity(cpuDepth[i]), float.IsPositiveInfinity(gpuDepth[i]));
                if (float.IsPositiveInfinity(cpuDepth[i]))
                {
                    infinite++;
                    continue;
                }

                double d = gpuDepth[i] - (double)cpuDepth[i];
                diff2 += d * d;
                ref2 += (double)cpuDepth[i] * cpuDepth[i];
            }

            double depth = Math.Sqrt(diff2 / Math.Max(ref2, 1e-30));
            Assert.Equal(expected[f].ValidMask, actual[f].ValidMask);
            Assert.NotNull(actual[f].InverseDepth);
            Assert.NotNull(expected[f].InverseDepth);
            double inverse = RelRms(actual[f].InverseDepth!, expected[f].InverseDepth!);
            output.WriteLine($"{label} frame {f}: depth relRMS {depth:E2}, inverse depth relRMS {inverse:E2}, {infinite} infinite pixels");
            Assert.True(depth <= bound, $"{label} frame {f}: depth relRMS {depth:E2}");
            Assert.True(inverse <= bound, $"{label} frame {f}: inverse depth relRMS {inverse:E2}");
        }
    }

    private static double Bound(Gpu gpu) => gpu == Gpu.Cuda ? Tf32Tolerance : Tolerance;

    private static double RelRms(float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double diff2 = 0, ref2 = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - (double)expected[i];
            diff2 += d * d;
            ref2 += (double)expected[i] * expected[i];
        }

        return Math.Sqrt(diff2 / Math.Max(ref2, 1e-30));
    }

    private static byte[] Crop(byte[] rgb, int width, int x0, int y0, int cropW, int cropH)
    {
        byte[] crop = new byte[cropW * cropH * 3];
        for (int y = 0; y < cropH; y++) Array.Copy(rgb, ((y0 + y) * width + x0) * 3, crop, y * cropW * 3, cropW * 3);
        return crop;
    }

    private static (byte[] Rgb, int Height, int Width) ReferenceRgb(string name)
    {
        string meta = Environment.GetEnvironmentVariable(ReferenceVariable)!;
        int[] shape = [.. JsonDocument.Parse(File.ReadAllText(meta)).RootElement.GetProperty($"{name}_rgb_u8").EnumerateArray().Select(e => e.GetInt32())];
        return (File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(meta)!, $"{name}_rgb_u8.bin")), shape[0], shape[1]);
    }

    private static SafeTensorsLoader Load(string variable)
    {
        SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable(variable)!);
        return loader;
    }

    // Both CUDA lanes pin numerics.noTf32 and numerics.sdpaNoTf32 (read at construction), so a settings file cannot swap them, and put back
    // whatever override was there before.
    private static IBackend Open(Gpu gpu)
    {
        if (gpu == Gpu.Vulkan) return new VulkanBackend();
        bool fp32 = gpu == Gpu.CudaFp32;
        bool? noTf32 = KnobStore.HasOverride(EngineKnobs.NoTf32) ? EngineKnobs.NoTf32.Value : null;
        bool? sdpaNoTf32 = KnobStore.HasOverride(EngineKnobs.SdpaNoTf32) ? EngineKnobs.SdpaNoTf32.Value : null;
        KnobStore.Set(EngineKnobs.NoTf32, fp32);
        KnobStore.Set(EngineKnobs.SdpaNoTf32, fp32);
        try
        {
            return new CudaBackend(0, Ernie5ModelGpuParityTests.PtxDirectory());
        }
        finally
        {
            Restore(EngineKnobs.NoTf32, noTf32);
            Restore(EngineKnobs.SdpaNoTf32, sdpaNoTf32);
        }
    }

    private static void Restore(Knob<bool> knob, bool? previous)
    {
        if (previous is { } value) KnobStore.Set(knob, value);
        else KnobStore.Clear(knob);
    }

    private static string? Probe(Gpu gpu)
    {
        try
        {
            using IBackend backend = Open(gpu);
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    /// <summary>A fact reported as skipped (never passed) when the weights, the reference inputs or the device are missing.</summary>
    private sealed class DepthPro2GpuFactAttribute : FactAttribute
    {
        public DepthPro2GpuFactAttribute(Gpu gpu)
        {
            foreach (string name in new[] { WeightsVariable, FocalVariable, ReferenceVariable })
            {
                string? path = Environment.GetEnvironmentVariable(name);
                if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                {
                    Skip = $"SKIPPED: {name} is unset or not an existing file; the GPU comparison did not run.";
                    return;
                }
            }

            string? failure = (gpu == Gpu.Vulkan ? VulkanFailure : CudaFailure).Value;
            if (failure is not null) Skip = $"SKIPPED: no usable {gpu} device ({failure}); the GPU comparison did not run.";
        }
    }

    /// <summary>The device lane: CUDA with its default TF32 GEMMs, CUDA in plain FP32, or Vulkan.</summary>
    public enum Gpu
    {
        Cuda,
        CudaFp32,
        Vulkan,
    }
}

/// <summary>Runs the Depth Pro 2 GPU tests alone: the CUDA FP32 lane sets process-wide numerics knobs while it builds its backend.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DepthPro2GpuKnobsCollection
{
    public const string Name = "DepthPro2GpuKnobs";
}
