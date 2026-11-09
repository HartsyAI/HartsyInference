using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The streaming SANA-WM DiT on CUDA against every DiT call of one upstream streaming run
/// (<c>dump_streaming_ref.py</c>: 73 frames, chunks of 4, 3 and 3 latent frames, four denoise passes and a t=0 save pass per
/// chunk, sink plus one cached chunk). Each call is replayed with upstream's own inputs (its latents, per-frame timesteps,
/// caption, and the chunk's slice of the camera tensors) through one <see cref="SanaWmStreamCache"/>, committing after each save
/// pass, so a later chunk's output also checks the state the earlier chunks left behind. Reports cosine and relative error per
/// call and the time of each chunk's five passes. Filter: <c>Category=GpuIntegration&amp;FullyQualifiedName~SanaWmStreamingDitGpu</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SanaWmStreamingDitGpuTests(ITestOutputHelper output)
{
    /// <summary>Names the upstream streaming run directory (index.json, streaming_ref.safetensors).</summary>
    public const string ReferenceEnvironmentVariable = "SANAWM_STREAM_REFERENCE_DIR";

    private const double CosineTarget = 0.999;
    private const int LatentHeight = 22, LatentWidth = 40;

    [SanaWmStreamingReferenceFact]
    public void EveryUpstreamCall_ReplaysThroughTheStreamCache()
    {
        Assert.True(CudaContext.IsAvailable(), "The streaming DiT replay needs a CUDA device; this is a failure, not a skip.");
        string run = Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable)!;
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(run, "index.json")));
        JsonElement[] calls = [.. index.RootElement.GetProperty("calls").EnumerateArray()];
        int cachedChunks = index.RootElement.GetProperty("num_cached_blocks").GetInt32();
        bool sink = index.RootElement.GetProperty("sink_token").GetBoolean();

        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(SanaWmStreamingReferenceFactAttribute.DitPath());
        using CudaBackend backend = new(0, PtxDirectory());
        IBackend device = backend;
        using SanaWmDit dit = new(backend, checkpoint);
        using SanaWmStreamCache cache = new(backend, checkpoint.Config, LatentHeight, LatentWidth,
            TokenWindowPolicy.RecentChunks(cachedChunks, sink));
        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(run, "streaming_ref.safetensors"));
        List<Tensor> owned = [];
        try
        {
            Tensor caption = Host32(reference.GetTensor("cond"), owned);
            Tensor mask = Host32(reference.GetTensor("cond_mask"), owned);
            Tensor raymap = Host32(reference.GetTensor("raymap"), owned);
            Tensor plucker = Host32(reference.GetTensor("chunk_plucker"), owned);
            double worstCosine = 1;
            Stopwatch chunkClock = Stopwatch.StartNew();
            for (int i = 0; i < calls.Length; i++)
            {
                int start = calls[i].GetProperty("start_f").GetInt32(), end = calls[i].GetProperty("end_f").GetInt32();
                bool save = calls[i].GetProperty("save_kv").GetBoolean();
                int frames = end - start;
                Tensor x = Host32(reference.GetTensor($"call{i:D2}_x"), owned);
                Assert.Equal(frames, (int)x.Shape[2]);
                float[] timesteps = FrameTimesteps(reference.GetTensor($"call{i:D2}_t"), frames, owned);
                float[] conditions = raymap.AsReadOnlySpan<float>().Slice(start * 20, frames * 20).ToArray();
                Tensor chunkPlucker = Frames(plucker, start, frames, owned);

                Stopwatch clock = Stopwatch.StartNew();
                using Tensor velocity = dit.ForwardChunk(x, timesteps, caption, mask, conditions, chunkPlucker, cache, commit: save);
                device.OffloadActivation(velocity);
                clock.Stop();
                ReadOnlySpan<float> expected = Host32(reference.GetTensor($"call{i:D2}_out"), owned).AsReadOnlySpan<float>();
                (double cosine, double relative) = Compare(velocity.AsReadOnlySpan<float>(), expected);
                worstCosine = Math.Min(worstCosine, cosine);
                output.WriteLine($"call {i:D2} frames {start}..{end} t[{string.Join(",", timesteps.Select(t => t.ToString("F0")))}]"
                    + $"{(save ? " save" : "")}: cosine {cosine:F6}, relative L2 {relative:E3}, {clock.Elapsed.TotalMilliseconds:F0} ms");
                if (save)
                {
                    cache.Commit(frames);
                    output.WriteLine($"chunk {start}..{end}: {chunkClock.Elapsed.TotalSeconds:F2} s for its passes; "
                        + $"VRAM used {UsedGib(device):F2} GiB");
                    chunkClock.Restart();
                }
            }

            output.WriteLine($"worst cosine {worstCosine:F6}");
            Assert.True(worstCosine >= CosineTarget, $"Worst per-call cosine {worstCosine:F6} is below {CosineTarget}.");
        }
        finally
        {
            SanaWmTensors.DisposeAll(owned);
        }
    }

    // Denoise passes record per-frame timesteps [1, 1, F]; the save pass records one zero per batch item.
    private static float[] FrameTimesteps(Tensor recorded, int frames, List<Tensor> owned)
    {
        ReadOnlySpan<float> values = Host32(recorded, owned).AsReadOnlySpan<float>();
        float[] timesteps = new float[frames];
        if (values.Length == frames)
            values.CopyTo(timesteps);
        else if (values.Length == 1)
            Array.Fill(timesteps, values[0]);
        else
            throw new InvalidDataException($"A recorded timestep holds {values.Length} values for {frames} frames.");
        return timesteps;
    }

    // Frames [start, start + count) of a host F32 [1, C, T, H, W] tensor.
    private static Tensor Frames(Tensor source, int start, int count, List<Tensor> owned)
    {
        int channels = (int)source.Shape[1], total = (int)source.Shape[2];
        int plane = (int)(source.Shape[3] * source.Shape[4]);
        Tensor slice = new(new TensorShape([1, channels, count, source.Shape[3], source.Shape[4]]), DType.F32);
        ReadOnlySpan<float> from = source.AsReadOnlySpan<float>();
        Span<float> to = slice.AsSpan<float>();
        for (int c = 0; c < channels; c++)
            from.Slice((c * total + start) * plane, count * plane).CopyTo(to.Slice(c * count * plane, count * plane));
        owned.Add(slice);
        return slice;
    }

    private static Tensor Host32(Tensor source, List<Tensor> owned)
    {
        if (source.DType == DType.F32)
            return source;
        Tensor converted = source.CastTo(DType.F32);
        owned.Add(converted);
        return converted;
    }

    private static (double Cosine, double Relative) Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double dot = 0, aa = 0, ee = 0, diff = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            dot += (double)actual[i] * expected[i];
            aa += (double)actual[i] * actual[i];
            ee += (double)expected[i] * expected[i];
            double d = actual[i] - expected[i];
            diff += d * d;
        }

        return (dot / Math.Sqrt(aa * ee), Math.Sqrt(diff / ee));
    }

    private static double UsedGib(IBackend backend)
    {
        (long free, long total) = backend.GetVramInfo();
        return (total - free) / (double)(1L << 30);
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

/// <summary>Skips at discovery unless <c>SANAWM_STREAM_REFERENCE_DIR</c> names an upstream streaming run and
/// <c>HARTSYINFERENCE_MODELS_DIR</c> holds the converted streaming DiT at <c>World/SanaWm/streaming/sana_dit/model.safetensors</c>.
/// With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it is never skipped.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SanaWmStreamingReferenceFactAttribute : FactAttribute
{
    private const string DitRelativePath = "World/SanaWm/streaming/sana_dit/model.safetensors";

    public SanaWmStreamingReferenceFactAttribute()
    {
        if (Environment.GetEnvironmentVariable(SanaWmRealWeightsFactAttribute.RequireEnvironmentVariable) == "1")
            return;
        string? run = Environment.GetEnvironmentVariable(SanaWmStreamingDitGpuTests.ReferenceEnvironmentVariable);
        bool hasRun = !string.IsNullOrWhiteSpace(run) && File.Exists(Path.Combine(run, "streaming_ref.safetensors"))
            && File.Exists(Path.Combine(run, "index.json"));
        if (!hasRun || DitPathOrNull() is null)
            Skip = $"SKIPPED: needs {SanaWmStreamingDitGpuTests.ReferenceEnvironmentVariable} (an upstream streaming run) and "
                + $"{SanaWmRealWeightsFactAttribute.ModelsDirEnvironmentVariable} holding {DitRelativePath}.";
    }

    /// <summary>The converted streaming DiT under the models root.</summary>
    public static string DitPath() => DitPathOrNull()
        ?? throw new InvalidOperationException($"{SanaWmRealWeightsFactAttribute.ModelsDirEnvironmentVariable} must hold {DitRelativePath}.");

    private static string? DitPathOrNull()
    {
        string? root = Environment.GetEnvironmentVariable(SanaWmRealWeightsFactAttribute.ModelsDirEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(root))
            return null;
        string path = Path.Combine(root, DitRelativePath);
        return File.Exists(path) ? path : null;
    }
}
