using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Full SANA-WM DiT, one CFG step on the CUDA backend, against the upstream <c>dit_out_raw_step0</c> dump, then the
/// conditional item alone (sequential CFG) against its half of that step; logs the step time and device memory. GPU lane:
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
        IBackend device = backend;
        long before = UsedBytes(device);
        using SanaWmDit dit = new(backend, checkpoint);
        long resident = UsedBytes(device);
        using SafeTensorsLoader sampler = Load(Path.Combine(runDirectory, "sampler.safetensors"));
        using SafeTensorsLoader text = Load(Path.Combine(runDirectory, "text.safetensors"));
        using SafeTensorsLoader camera = Load(Path.Combine(runDirectory, "camera.safetensors"));
        List<Tensor> owned = [];
        try
        {
            // Host tensors: the backend uploads what an op reads, so no device-tagged copy is made here.
            Tensor x = Host32(sampler.GetTensor("dit_in_latents_step0"), owned);
            Tensor caption = Host32(text.GetTensor("dit_y_step0"), owned);
            Tensor mask = MaskAsFloat(text.GetTensor("dit_mask_step0"), owned);
            Tensor chunk = Host32(camera.GetTensor("dit_chunk_plucker_step0"), owned);
            float[] timesteps = Floats(sampler.GetTensor("dit_timestep_step0"), owned);
            float[] cameraConditions = Floats(camera.GetTensor("dit_camera_conditions_step0"), owned);

            Stopwatch clock = Stopwatch.StartNew();
            using Tensor velocity = dit.Forward(x, timesteps, caption, mask, cameraConditions, chunk);
            long afterForward = UsedBytes(device);
            device.OffloadActivation(velocity);
            clock.Stop();
            Tensor expectedHost = Host32(sampler.GetTensor("dit_out_raw_step0"), owned);
            double cosine = Cosine(velocity.AsReadOnlySpan<float>(), expectedHost.AsReadOnlySpan<float>());
            _output.WriteLine($"full DiT step 0: cosine {cosine:F6}, {clock.Elapsed.TotalSeconds:F1} s for batch {x.Shape[0]}");
            // Device-wide cuMemGetInfo, other processes included; the stream-ordered pool keeps freed blocks reserved, so the
            // figure after the forward approximates this process's high-water mark.
            _output.WriteLine($"VRAM used: {Gib(before)} before, {Gib(resident)} with the weights preloaded, {Gib(afterForward)} after the forward");
            Assert.True(cosine >= CosineTarget, $"Full DiT cosine {cosine:F6} is below {CosineTarget}.");

            // Sequential CFG: the conditional item alone, as a batch-1 call, against its half of the batch-2 result.
            int frames = (int)x.Shape[2];
            using Tensor single = dit.Forward(Item(x, 1, owned), timesteps.AsSpan(frames, frames), Item(caption, 1, owned),
                Item(mask, 1, owned), cameraConditions.AsSpan(cameraConditions.Length / 2), Item(chunk, 1, owned));
            device.OffloadActivation(single);
            ReadOnlySpan<float> joint = velocity.AsReadOnlySpan<float>();
            double itemCosine = Cosine(single.AsReadOnlySpan<float>(), joint[(joint.Length / 2)..]);
            _output.WriteLine($"batch-1 conditional item vs its batch-2 half: cosine {itemCosine:F8}");
            Assert.True(itemCosine >= 0.99999, $"Batch-1 forward cosine {itemCosine:F8} against the batch-2 half.");
        }
        finally
        {
            SanaWmTensors.DisposeAll(owned);
        }
    }

    /// <summary>Batch item <paramref name="item"/> of a host F32 tensor, as a batch-1 host tensor.</summary>
    private static Tensor Item(Tensor source, int item, List<Tensor> owned)
    {
        Span<long> dims = stackalloc long[source.Shape.Rank];
        source.Shape.CopyDimsTo(dims);
        int length = (int)(source.ElementCount / dims[0]);
        dims[0] = 1;
        Tensor slice = new(new TensorShape(dims), DType.F32);
        source.AsReadOnlySpan<float>().Slice(item * length, length).CopyTo(slice.AsSpan<float>());
        owned.Add(slice);
        return slice;
    }

    private static long UsedBytes(IBackend backend)
    {
        (long free, long total) = backend.GetVramInfo();
        return total - free;
    }

    private static string Gib(long bytes) => $"{bytes / (double)(1L << 30):F2} GiB";

    private static SafeTensorsLoader Load(string path)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return loader;
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
