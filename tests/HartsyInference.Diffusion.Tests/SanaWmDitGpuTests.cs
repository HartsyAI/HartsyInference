using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Full SANA-WM DiT, one CFG step on the CUDA backend, against the upstream <c>dit_out_raw_step0</c> dump. GPU lane:
/// needs a CUDA device, <c>HARTSYINFERENCE_MODELS_DIR</c> and <c>SANAWM_REFERENCE_RUN_DIR</c>; skipped otherwise. Filter:
/// <c>Category=GpuIntegration&amp;FullyQualifiedName~SanaWmDitGpu</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SanaWmDitGpuTests
{
    private const double CosineTarget = 0.999;
    private readonly ITestOutputHelper _output;

    public SanaWmDitGpuTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SanaWmBlockReferenceFact]
    public void FullDit_OneCfgStep_MatchesUpstreamVelocity()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        string runDirectory = SanaWmBlockReferenceFactAttribute.ReferenceDirectory();
        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(SanaWmRealWeightsFactAttribute.ResolveDitPath());
        using CudaBackend backend = new(0, PtxDirectory());
        using SanaWmDit dit = new(backend, checkpoint);
        using SafeTensorsLoader sampler = Load(Path.Combine(runDirectory, "sampler.safetensors"));
        using SafeTensorsLoader text = Load(Path.Combine(runDirectory, "text.safetensors"));
        using SafeTensorsLoader camera = Load(Path.Combine(runDirectory, "camera.safetensors"));
        List<Tensor> owned = [];
        try
        {
            Tensor x = Device(backend, sampler.GetTensor("dit_in_latents_step0"), owned);
            Tensor caption = Device(backend, text.GetTensor("dit_y_step0"), owned);
            Tensor mask = Device(backend, MaskAsFloat(text.GetTensor("dit_mask_step0"), owned), owned);
            Tensor chunk = Device(backend, camera.GetTensor("dit_chunk_plucker_step0"), owned);
            float[] timesteps = Floats(sampler.GetTensor("dit_timestep_step0"), owned);
            float[] cameraConditions = Floats(camera.GetTensor("dit_camera_conditions_step0"), owned);

            using Tensor velocity = dit.Forward(x, timesteps, caption, mask, cameraConditions, chunk);
            using Tensor actual = SanaWmTensors.Readback(backend, velocity);
            Tensor expectedHost = Host32(sampler.GetTensor("dit_out_raw_step0"), owned);
            double cosine = Cosine(actual.AsReadOnlySpan<float>(), expectedHost.AsReadOnlySpan<float>());
            _output.WriteLine($"full DiT step 0: cosine {cosine:F6}");
            Assert.True(cosine >= CosineTarget, $"Full DiT cosine {cosine:F6} is below {CosineTarget}.");
        }
        finally
        {
            SanaWmTensors.DisposeAll(owned);
        }
    }

    private static SafeTensorsLoader Load(string path)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return loader;
    }

    // Dtype conversion happens on the host, where the loader's tensors live; the device copy is then uploaded.
    private static Tensor Device(IBackend backend, Tensor source, List<Tensor> owned)
    {
        Tensor host = Host32(source, owned);
        Tensor device = new(host.Shape, DType.F32, backend.Device);
        backend.CopyTo(device, host);
        owned.Add(device);
        return device;
    }

    private static Tensor Host32(Tensor source, List<Tensor> owned)
    {
        if (source.DType == DType.F32)
            return source;
        Tensor converted = source.CastTo(DType.F32);
        owned.Add(converted);
        return converted;
    }

    private static float[] Floats(Tensor source, List<Tensor> owned) => Host32(source, owned).AsReadOnlySpan<float>().ToArray();

    private static Tensor MaskAsFloat(Tensor mask, List<Tensor> owned)
    {
        Tensor converted = new(mask.Shape, DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<long> source = mask.AsReadOnlySpan<long>();
        Span<float> destination = converted.AsSpan<float>();
        for (int i = 0; i < destination.Length; i++)
            destination[i] = source[i];
        owned.Add(converted);
        return converted;
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }

        return dot / Math.Sqrt(Math.Max(normA * normB, double.Epsilon));
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
