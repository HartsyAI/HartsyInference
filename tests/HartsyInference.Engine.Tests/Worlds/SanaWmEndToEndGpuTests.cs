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

/// <summary>SANA-WM stage 1 end to end on CUDA, along the CLI's path (options, planner, <c>WorldService.Open</c>, the one-shot
/// session), with run <c>A_demo_pose</c>'s settings: the demo image, prompt, pose and intrinsics files, 161 frames, 60 steps,
/// CFG 5.0, shift 9.8, seed 42. The recorded upstream <c>z0</c> is injected, so the run differs from upstream only in the
/// DiT, the sampler arithmetic and the VAE decode. Reports per-frame PSNR and SSIM against the upstream frames. Gates apply only
/// when <c>SANAWM_E2E_MIN_PSNR</c> / <c>SANAWM_E2E_MIN_SSIM</c> are set; <c>SANAWM_E2E_OUTPUT_DIR</c> also writes the frames and
/// a 16 fps MP4 for viewing. Filter: <c>Category=GpuIntegration&amp;FullyQualifiedName~SanaWmEndToEnd</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SanaWmEndToEndGpuTests
{
    private const string MinPsnrVariable = "SANAWM_E2E_MIN_PSNR";
    private const string MinSsimVariable = "SANAWM_E2E_MIN_SSIM";
    private const string OutputVariable = "SANAWM_E2E_OUTPUT_DIR";
    private const string VramTraceVariable = "SANAWM_E2E_VRAM_TRACE";
    private readonly ITestOutputHelper _output;

    public SanaWmEndToEndGpuTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SanaWmEndToEndFact]
    public async Task DemoPoseRun_WithTheRecordedZ0_MatchesTheUpstreamFrames()
    {
        Assert.True(CudaContext.IsAvailable(), "SANA-WM end to end needs a CUDA device; this is a failure, not a skip.");

        string run = SanaWmEndToEndFactAttribute.ReferenceDirectory();
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(run, "index.json")));
        JsonElement args = index.RootElement.GetProperty("args");
        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(args.GetProperty("image").GetString()!);
        WorldRequestOptions options = new WorldRequestOptions
        {
            Prompt = File.ReadAllText(args.GetProperty("prompt").GetString()!).Trim(),
            Frames = args.GetProperty("num_frames").GetInt32(),
            Steps = args.GetProperty("step").GetInt32(),
            Guidance = args.GetProperty("cfg_scale").GetDouble(),
            FlowShift = args.GetProperty("flow_shift").GetDouble(),
            Seed = args.GetProperty("seed").GetInt64(),
            CameraPath = args.GetProperty("pose").GetString(),
            Intrinsics = args.GetProperty("intrinsics").GetString(),
        };
        WorldRequest request = options.ToRequest(new ImageData { Rgb = rgb, Width = width, Height = height });
        ModelSpec spec = ModelResolver.Resolve("sana-wm", SanaWmEndToEndFactAttribute.DitPath(), Modality.World);

        using InferenceEngine engine = new InferenceEngine("cuda");
        WorldService service = (WorldService)engine.World;
        service.SanaWmInitialLatent = _ => RecordedZ0(run);
        WorldPlan plan = WorldPlanResolver.Resolve(spec, request, engine.BackendSelector);
        _output.WriteLine(plan.Snapshot());

        // SANAWM_E2E_VRAM_TRACE names a file for the per-phase device-memory trace (diagnostics.vramTrace).
        string? trace = Environment.GetEnvironmentVariable(VramTraceVariable);
        if (!string.IsNullOrWhiteSpace(trace))
            KnobStore.Set(EngineKnobs.VramTrace, trace);
        List<VideoFrame> frames = [];
        Stopwatch clock = Stopwatch.StartNew();
        try
        {
            using IWorldSession session = service.Open(spec, plan.Request);
            await foreach (VideoFrame frame in session.StreamAsync(CancellationToken.None))
                frames.Add(frame);
        }
        finally
        {
            if (!string.IsNullOrWhiteSpace(trace))
                KnobStore.Clear(EngineKnobs.VramTrace);
        }
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"rollout wall: {clock.Elapsed.TotalSeconds:F1} s for {frames.Count} frames ({plan.Steps} steps)"));
        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.Equal(plan.Frames, frames.Count);

        WriteOutputs(frames, plan.Fps);
        double minPsnr = double.PositiveInfinity, minSsim = double.PositiveInfinity, sumPsnr = 0, sumSsim = 0;
        StringBuilder table = new StringBuilder("frame,psnr_db,ssim\n");
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
        _output.WriteLine(table.ToString());
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"PSNR min {minPsnr:F3} dB mean {sumPsnr / frames.Count:F3} dB; SSIM min {minSsim:F5} mean {sumSsim / frames.Count:F5}"));

        double? psnrGate = Gate(MinPsnrVariable), ssimGate = Gate(MinSsimVariable);
        if (psnrGate is null && ssimGate is null)
            _output.WriteLine($"UNGATED: set {MinPsnrVariable} and {MinSsimVariable} to enforce the frozen limits.");
        if (psnrGate is double p)
            Assert.True(minPsnr >= p, $"Minimum per-frame PSNR {minPsnr:F3} dB is below {p} dB.");
        if (ssimGate is double s)
            Assert.True(minSsim >= s, $"Minimum per-frame SSIM {minSsim:F5} is below {s}.");
    }

    private static Tensor RecordedZ0(string run)
    {
        using SafeTensorsLoader image = new SafeTensorsLoader();
        image.Load(Path.Combine(run, "image.safetensors"));
        Tensor z0 = image.GetTensor("z0");
        return z0.CastTo(DType.F32);
    }

    private void WriteOutputs(IReadOnlyList<VideoFrame> frames, int fps)
    {
        string? directory = Environment.GetEnvironmentVariable(OutputVariable);
        if (string.IsNullOrWhiteSpace(directory))
            return;
        byte[][] pixels = frames.Select(frame => frame.Rgb).ToArray();
        VideoOutputWriter.Written written = VideoOutputWriter.Write(pixels, frames[0].Width, frames[0].Height, directory,
            "sana-wm-demo-pose", null, fps);
        _output.WriteLine($"frames: {written.Directory}; mp4: {written.Mp4Path ?? "(mux failed)"}");
    }

    private static double? Gate(string variable) =>
        double.TryParse(Environment.GetEnvironmentVariable(variable), NumberStyles.Float, CultureInfo.InvariantCulture, out double value)
            ? value : null;
}
