using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Codec;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests.Worlds;

/// <summary>SANA-WM streaming end to end on CUDA, along the CLI's path (options, planner, <c>WorldService.Open</c>, the chunked
/// session), against one upstream streaming run (<c>dump_streaming_ref.py</c>): the demo image, prompt, pose and intrinsics,
/// the run's frame count, guidance 1.0 and seed. Upstream's first latent and noise volume are injected chunk by chunk, so the
/// run differs from upstream only in the DiT, the sampler arithmetic and the causal VAE decode. Reports per-frame PSNR and SSIM
/// against the upstream frames, when each chunk's frames arrived, and the device-memory trace with <c>SANAWM_E2E_VRAM_TRACE</c>.
/// Gates apply only when <c>SANAWM_STREAM_MIN_PSNR</c> / <c>SANAWM_STREAM_MIN_SSIM</c> are set; <c>SANAWM_E2E_OUTPUT_DIR</c>
/// also writes the frames and a 16 fps MP4. Filter: <c>Category=GpuIntegration&amp;FullyQualifiedName~SanaWmStreamingEndToEnd</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SanaWmStreamingEndToEndGpuTests(ITestOutputHelper output)
{
    private const string MinPsnrVariable = "SANAWM_STREAM_MIN_PSNR";
    private const string MinSsimVariable = "SANAWM_STREAM_MIN_SSIM";
    private const string OutputVariable = "SANAWM_E2E_OUTPUT_DIR";
    private const string VramTraceVariable = "SANAWM_E2E_VRAM_TRACE";

    [SanaWmStreamingEndToEndFact]
    public async Task DemoPoseStream_WithTheRecordedNoise_MatchesTheUpstreamFrames()
    {
        Assert.True(CudaContext.IsAvailable(), "SANA-WM streaming end to end needs a CUDA device; this is a failure, not a skip.");
        string run = SanaWmStreamingEndToEndFactAttribute.ReferenceDirectory();
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(run, "index.json")));
        JsonElement root = index.RootElement;
        JsonElement assets = root.GetProperty("assets");
        int[] bounds = [.. root.GetProperty("chunk_bounds").EnumerateArray().Select(e => e.GetInt32())];
        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(assets.GetProperty("image").GetString()!);
        // A run driven by an action script streams through the live camera path (the script queued into the stream, one set
        // of intrinsics), as `hartsy world --actions` does; a pose run follows the recorded camera file.
        bool actionRun = root.TryGetProperty("camera_source", out JsonElement source) && source.GetString() == "action";
        WorldRequestOptions options = new()
        {
            Prompt = root.GetProperty("prompt").GetString()!,
            Frames = root.GetProperty("num_frames").GetInt32(),
            Guidance = root.GetProperty("cfg_scale").GetDouble(),
            Seed = root.GetProperty("seed").GetInt64(),
            CameraPath = actionRun ? null : assets.GetProperty("pose").GetString(),
            CameraScript = actionRun ? root.GetProperty("action").GetString() : null,
            Intrinsics = actionRun ? root.GetProperty("intrinsics_px").GetString() : assets.GetProperty("intrinsics").GetString(),
        };
        WorldRequest request = options.ToRequest(new ImageData { Rgb = rgb, Width = width, Height = height });
        ModelSpec spec = ModelResolver.Resolve("sana-wm-streaming", SanaWmStreamingEndToEndFactAttribute.DitPath(), Modality.World);
        spec = spec with { Aux = SanaWmStreamingEndToEndFactAttribute.ComponentPaths(spec.Aux) };

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(run, "streaming_ref.safetensors"));
        using Tensor z0 = reference.GetTensor("z0").CastTo(DType.F32);
        List<Tensor> noise = [];
        try
        {
            for (int c = 0; c + 1 < bounds.Length; c++)
                noise.Add(Frames(z0, bounds[c], bounds[c + 1] - bounds[c]));

            using InferenceEngine engine = new("cuda");
            WorldService service = (WorldService)engine.World;
            service.SanaWmStreamInjection = _ => new SanaWmStreamInjection
            {
                FirstLatent = reference.GetTensor("first_latent").CastTo(DType.F32),
                ChunkNoise = chunk => chunk < noise.Count ? noise[chunk] : null,
            };
            WorldPlan plan = WorldPlanResolver.Resolve(spec, request, engine.BackendSelector);
            output.WriteLine(plan.Snapshot());
            Assert.True(plan.Streaming);

            string? trace = Environment.GetEnvironmentVariable(VramTraceVariable);
            if (!string.IsNullOrWhiteSpace(trace))
                KnobStore.Set(EngineKnobs.VramTrace, trace);
            List<VideoFrame> frames = [];
            List<double> arrivals = [];
            Stopwatch clock = Stopwatch.StartNew();
            try
            {
                using IWorldSession session = service.Open(spec, plan.Request);
                await foreach (VideoFrame frame in session.StreamAsync(CancellationToken.None))
                {
                    frames.Add(frame);
                    arrivals.Add(clock.Elapsed.TotalSeconds);
                }
            }
            finally
            {
                if (!string.IsNullOrWhiteSpace(trace))
                    KnobStore.Clear(EngineKnobs.VramTrace);
            }

            Assert.Equal(0, service.RunLedger.Outstanding);
            Assert.Equal(plan.Frames, frames.Count);
            ReportChunkArrivals(bounds, arrivals);
            WriteOutputs(frames, plan.Fps);
            CompareFrames(run, frames);
        }
        finally
        {
            foreach (Tensor tensor in noise)
                tensor.Dispose();
        }
    }

    // A chunk of latent frames [start, end) decodes to 8 pixel frames per latent frame, the first latent frame to one.
    private void ReportChunkArrivals(int[] bounds, List<double> arrivals)
    {
        double previous = 0;
        for (int c = 0; c + 1 < bounds.Length; c++)
        {
            int lastPixel = (bounds[c + 1] - 1) * 8;
            double at = arrivals[Math.Min(lastPixel, arrivals.Count - 1)];
            output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"chunk {c} (latent {bounds[c]}..{bounds[c + 1]}): complete at {at:F2} s, {at - previous:F2} s after the previous chunk"));
            previous = at;
        }
    }

    private void CompareFrames(string run, List<VideoFrame> frames)
    {
        double minPsnr = double.PositiveInfinity, minSsim = double.PositiveInfinity, sumPsnr = 0, sumSsim = 0;
        StringBuilder table = new("frame,psnr_db,ssim\n");
        for (int i = 0; i < frames.Count; i++)
        {
            (byte[] expected, int expectedWidth, int expectedHeight) =
                PngDecoder.DecodeFromFile(Path.Combine(run, "frames_png", $"frame_{i:D3}.png"));
            Assert.Equal((expectedWidth, expectedHeight), (frames[i].Width, frames[i].Height));
            double psnr = FrameMetrics.Psnr(expected, frames[i].Rgb);
            double ssim = FrameMetrics.Ssim(expected, frames[i].Rgb, expectedWidth, expectedHeight);
            minPsnr = Math.Min(minPsnr, psnr);
            minSsim = Math.Min(minSsim, ssim);
            sumPsnr += Math.Min(psnr, 100.0);
            sumSsim += ssim;
            table.Append(CultureInfo.InvariantCulture, $"{i},{psnr:F3},{ssim:F5}\n");
        }

        output.WriteLine(table.ToString());
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"PSNR min {minPsnr:F3} dB mean {sumPsnr / frames.Count:F3} dB; SSIM min {minSsim:F5} mean {sumSsim / frames.Count:F5}"));
        double? psnrGate = Gate(MinPsnrVariable), ssimGate = Gate(MinSsimVariable);
        if (psnrGate is null && ssimGate is null)
            output.WriteLine($"UNGATED: set {MinPsnrVariable} and {MinSsimVariable} to enforce the frozen limits.");
        if (psnrGate is double p)
            Assert.True(minPsnr >= p, $"Minimum per-frame PSNR {minPsnr:F3} dB is below {p} dB.");
        if (ssimGate is double s)
            Assert.True(minSsim >= s, $"Minimum per-frame SSIM {minSsim:F5} is below {s}.");
    }

    // Frames [start, start + count) of a host F32 [1, C, T, H, W] tensor, as a new host tensor.
    private static Tensor Frames(Tensor source, int start, int count)
    {
        int channels = (int)source.Shape[1], total = (int)source.Shape[2];
        int plane = (int)(source.Shape[3] * source.Shape[4]);
        Tensor slice = new(new TensorShape([1, channels, count, source.Shape[3], source.Shape[4]]), DType.F32);
        ReadOnlySpan<float> from = source.AsReadOnlySpan<float>();
        Span<float> to = slice.AsSpan<float>();
        for (int c = 0; c < channels; c++)
            from.Slice((c * total + start) * plane, count * plane).CopyTo(to.Slice(c * count * plane, count * plane));
        return slice;
    }

    private void WriteOutputs(IReadOnlyList<VideoFrame> frames, int fps)
    {
        string? directory = Environment.GetEnvironmentVariable(OutputVariable);
        if (string.IsNullOrWhiteSpace(directory))
            return;
        byte[][] pixels = frames.Select(frame => frame.Rgb).ToArray();
        VideoOutputWriter.Written written = VideoOutputWriter.Write(pixels, frames[0].Width, frames[0].Height, directory,
            "sana-wm-streaming-demo-pose", null, fps);
        output.WriteLine($"frames: {written.Directory}; mp4: {written.Mp4Path ?? "(mux failed)"}");
    }

    private static double? Gate(string variable) =>
        double.TryParse(Environment.GetEnvironmentVariable(variable), NumberStyles.Float, CultureInfo.InvariantCulture, out double value)
            ? value : null;
}

/// <summary>Skips at discovery unless <c>SANAWM_STREAM_REFERENCE_DIR</c> names an upstream streaming run (index.json,
/// streaming_ref.safetensors, frames_png) and <c>HARTSYINFERENCE_MODELS_DIR</c> holds the streaming release under
/// <c>World/SanaWm/streaming</c> (the converted DiT and the causal VAE) and the stage-1 Gemma under <c>World/SanaWm/text_encoder</c>.
/// With <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it is never skipped.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class SanaWmStreamingEndToEndFactAttribute : FactAttribute
{
    /// <summary>Names the upstream streaming run directory.</summary>
    public const string ReferenceEnvironmentVariable = "SANAWM_STREAM_REFERENCE_DIR";

    private const string DitRelative = "World/SanaWm/streaming/sana_dit/model.safetensors";
    private const string VaeRelative = "World/SanaWm/streaming/ltx2_causal_vae/diffusion_pytorch_model.safetensors";
    private const string TextEncoderRelative = "World/SanaWm/text_encoder/gemma-2-2b-it/gemma-2-2b-it.safetensors";

    public SanaWmStreamingEndToEndFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1")
            return;
        string? run = Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable);
        bool hasRun = !string.IsNullOrWhiteSpace(run) && File.Exists(Path.Combine(run, "streaming_ref.safetensors"))
            && Directory.Exists(Path.Combine(run, "frames_png"));
        string? root = ModelsRoot();
        bool hasWeights = root is not null && File.Exists(Path.Combine(root, DitRelative)) && File.Exists(Path.Combine(root, VaeRelative))
            && File.Exists(Path.Combine(root, TextEncoderRelative));
        if (!hasRun || !hasWeights)
            Skip = $"SKIPPED: needs {ReferenceEnvironmentVariable} (an upstream streaming run) and "
                + $"{SanaWmEndToEndFactAttribute.ModelsDirEnvironmentVariable} holding the streaming release and Gemma.";
    }

    /// <summary>The upstream streaming run directory.</summary>
    public static string ReferenceDirectory() => Environment.GetEnvironmentVariable(ReferenceEnvironmentVariable)
        ?? throw new InvalidOperationException($"{ReferenceEnvironmentVariable} must name the upstream streaming run.");

    /// <summary>The converted streaming DiT.</summary>
    public static string DitPath() => Path.Combine(RequireRoot(), DitRelative);

    /// <summary><paramref name="aux"/> with the causal VAE and the shared Gemma named explicitly, so the run does not depend on
    /// the resolver's sibling layout.</summary>
    public static IReadOnlyDictionary<string, string> ComponentPaths(IReadOnlyDictionary<string, string> aux)
    {
        Dictionary<string, string> paths = new(aux)
        {
            [WorldComponentResolver.VaeAuxKey] = Path.Combine(RequireRoot(), VaeRelative),
            [WorldComponentResolver.TextEncoderAuxKey] = Path.Combine(RequireRoot(), TextEncoderRelative),
        };
        return paths;
    }

    private static string RequireRoot() => ModelsRoot()
        ?? throw new InvalidOperationException($"{SanaWmEndToEndFactAttribute.ModelsDirEnvironmentVariable} must name the models root.");

    private static string? ModelsRoot()
    {
        string? root = Environment.GetEnvironmentVariable(SanaWmEndToEndFactAttribute.ModelsDirEnvironmentVariable);
        return string.IsNullOrWhiteSpace(root) ? null : root;
    }
}
