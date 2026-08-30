using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Opt-in real-checkpoint reproduction of the dense MiniMax-H3 guide plus AV-mask canary. The first
/// generation supplies an aligned source clip; the second uses its frame 17, soundtrack, and full video as the
/// visual/audio guide and preservation sources. Both outputs and their execution records remain on disk for the
/// manual inspection required before release.</summary>
[Collection("CudaSerial")]
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MiniMaxH3GuideMaskRealWeightTests
{
    private const string RunEnvVar = "HARTSY_RUN_H3_GUIDE_MASK_REAL";
    private const int Width = 512;
    private const int Height = 288;
    private const int Frames = 39;
    private const int Fps = 24;
    private const int Steps = 30;
    private const int GuideFrame = 17;

    private const string TransformerSha256 =
        "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993";
    private const string TextEncoderSha256 =
        "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6";
    private const string VideoVaeSha256 =
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522";
    private const string AudioVaeSha256 =
        "37dddc2f3e6d5d5139d823d5ea283bbf304dadcb885b1ccda818aa13dade5ea2";

    private const string Prompt =
        "A joyful golden retriever puppy splashing through a shallow forest stream, cinematic natural light, "
        + "realistic water droplets, steady tracking camera, coherent motion, ambient rushing water and birds";

    private readonly ITestOutputHelper _output;

    public MiniMaxH3GuideMaskRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DenseGuideAndAvMasks_RealCheckpoint_ProduceInspectableAlignedClip()
    {
        if (Environment.GetEnvironmentVariable(RunEnvVar) != "1")
        {
            _output.WriteLine($"SKIPPED: set {RunEnvVar}=1 and {RealWeightGate.RequireEnvVar}=1 "
                + "to run the exact-artifact MiniMax-H3 guide/AV-mask canary.");
            return;
        }

        Assert.True(Environment.GetEnvironmentVariable(RealWeightGate.RequireEnvVar) == "1",
            $"{RunEnvVar}=1 must be paired with {RealWeightGate.RequireEnvVar}=1 so missing exact assets fail.");
        Assert.True(CudaContext.IsAvailable(), $"{RunEnvVar}=1 requires a CUDA device.");
        Assert.True(FfmpegAvailable(), $"{RunEnvVar}=1 requires ffmpeg on PATH to build the aligned source clip.");
        if (!RealWeightGate.Require(_output.WriteLine,
                TestPaths.MiniMaxH3.DitFp8,
                TestPaths.MiniMaxH3.TextEncoder,
                TestPaths.MiniMaxH3.VideoVae,
                TestPaths.MiniMaxH3.AudioVae))
        {
            return;
        }

        Dictionary<string, string> componentHashes = new(StringComparer.Ordinal)
        {
            ["transformer"] = await RequireSha256Async(
                TestPaths.MiniMaxH3.DitFp8, TransformerSha256),
            ["textEncoder"] = await RequireSha256Async(
                TestPaths.MiniMaxH3.TextEncoder, TextEncoderSha256),
            ["videoVae"] = await RequireSha256Async(
                TestPaths.MiniMaxH3.VideoVae, VideoVaeSha256),
            ["audioVae"] = await RequireSha256Async(
                TestPaths.MiniMaxH3.AudioVae, AudioVaeSha256),
        };

        int gpu = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_GPU"), out int requestedGpu)
            ? requestedGpu : 0;
        Assert.InRange(gpu, 0, CudaContext.GetDeviceCount() - 1);
        ModelSpec spec = ModelResolver.Resolve(
            "minimax-h3", TestPaths.MiniMaxH3.DitFp8, Modality.Video);
        Assert.Equal(Path.GetFullPath(TestPaths.MiniMaxH3.DitFp8), spec.LocalPath);

        ComponentOverrides components = new()
        {
            Qwen = TestPaths.MiniMaxH3.TextEncoder,
            VideoVae = TestPaths.MiniMaxH3.VideoVae,
            AudioVae = TestPaths.MiniMaxH3.AudioVae,
        };
        VideoRequest baselineRequest = Request(seed: 424242, components);

        string outputRoot = Path.Combine(TestPaths.OutputDir, "h3-guide-mask-real-weight");
        using InferenceEngine engine = new("cuda", gpu);
        LoggingProgress progress = new(_output);

        _output.WriteLine("[1/2] Generating the 39-frame dense source clip.");
        VideoPlan baselinePlan = await engine.VideoPlanning.PlanAsync(spec, baselineRequest);
        AssertValidDensePlan(baselinePlan, baselineRequest.Seed);
        Assert.Equal(TransformerSha256, baselinePlan.ArtifactHashes["transformer"]);
        Assert.Equal(VideoVaeSha256, baselinePlan.ArtifactHashes["videoVae"]);
        VideoGenerationResult baseline = await engine.Video.GenerateAsync(
            baselinePlan, baselineRequest, progress);
        VideoOutputWriter.Written baselineWritten = Persist(
            baseline, outputRoot, "minimax-h3-dense-source");
        AssertExecution(baseline, baselineRequest.Seed);
        AssertCoherent(baseline, "baseline");
        Assert.True(baselineWritten.Mp4Path is not null,
            "ffmpeg did not produce the MP4 required as the aligned video-mask source.");
        Assert.NotNull(baselineWritten.AudioPath);

        AudioBuffer sourceAudio = Assert.IsType<AudioBuffer>(baseline.Audio);
        AudioClip fullSourceAudio = Wav(sourceAudio);
        AudioClip guideAudio = Wav(CropAudio(sourceAudio, GuideFrame / (double)Fps));
        VideoFrame guideFrame = baseline.Frames[GuideFrame];

        VideoRequest maskedRequest = Request(seed: 424247, components) with
        {
            Guides =
            [
                new VideoGuide
                {
                    FrameIndex = GuideFrame,
                    Image = new ImageData
                    {
                        Rgb = guideFrame.Rgb,
                        Width = guideFrame.Width,
                        Height = guideFrame.Height,
                    },
                    Audio = guideAudio,
                },
            ],
            VideoDenoiseMask = new VideoDenoiseMask
            {
                MaskImage = HorizontalMask(),
                SourceVideo = new VideoClip
                {
                    Data = await File.ReadAllBytesAsync(baselineWritten.Mp4Path!),
                    Format = "mp4",
                },
            },
            AudioDenoiseMask = new AudioDenoiseMask
            {
                Values = AudioMaskValues(),
                Rate = 40f,
                Source = fullSourceAudio,
            },
        };

        _output.WriteLine("[2/2] Generating with frame-17 visual/audio guide and continuous AV masks.");
        VideoPlan releasePlan = await engine.VideoPlanning.PlanAsync(spec, maskedRequest);
        VideoPlanIssue releaseIssue = Assert.Single(releasePlan.Issues,
            issue => issue.Code == "video.h3_expansion.release_blocked");
        Assert.Equal(VideoPlanIssueSeverity.Error, releaseIssue.Severity);
        Assert.Equal(
            $"Profile '{releasePlan.Profile.Id}' requires validation-pending arbitrary guides, AV denoise masks. "
                + "Its operator-provided real-generation and output-inspection release gate has not passed, "
                + "so this published build cannot execute it.",
            releaseIssue.Message);
        VideoPlan maskedPlan = BindValidationCanaryPlan(releasePlan);
        AssertValidDensePlan(maskedPlan, maskedRequest.Seed);
        VideoGenerationResult masked = await engine.Video.GenerateAsync(maskedPlan, maskedRequest, progress);
        VideoOutputWriter.Written maskedWritten = Persist(
            masked, outputRoot, "minimax-h3-dense-guide-av-mask");
        AssertExecution(masked, maskedRequest.Seed);
        AssertCoherent(masked, "guide/mask");
        Assert.NotNull(maskedWritten.Mp4Path);
        Assert.NotNull(maskedWritten.AudioPath);

        double guideSsim = Ssim.Compute(
            baseline.Frames[GuideFrame].Rgb, masked.Frames[GuideFrame].Rgb, Width, Height);
        double preservedLeftSsim = MeanRegionSsim(
            baseline.Frames, masked.Frames, x: 0, width: Width / 4);
        double generatedRightSsim = MeanRegionSsim(
            baseline.Frames, masked.Frames, x: Width * 3 / 4, width: Width / 4);
        double seamJump = MeanAdjacentColumnJump(masked.Frames, x: 460);
        double earlyAudioDiffDb = DifferenceDb(sourceAudio, masked.Audio!, 0d, 0.45d);
        double lateAudioDiffDb = DifferenceDb(sourceAudio, masked.Audio!, 1.05d, 1.55d);
        double postOnsetRms = Rms(masked.Audio!, 0.35d, masked.Audio!.Seconds);

        string evidencePath = Path.Combine(maskedWritten.Directory, "real-weight-evidence.json");
        JsonSerializerOptions json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
        json.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
        await File.WriteAllTextAsync(evidencePath, JsonSerializer.Serialize(new
        {
            runUtc = DateTimeOffset.UtcNow,
            exactArtifacts = componentHashes,
            baselineExecution = baseline.Execution,
            maskedExecution = masked.Execution,
            metrics = new
            {
                guideFrameIndex = GuideFrame,
                guideSsim,
                preservedLeftSsim,
                generatedRightSsim,
                seamColumn = 460,
                seamJump,
                earlyAudioDiffDb,
                lateAudioDiffDb,
                postOnsetRms,
            },
            outputs = new
            {
                baseline = await OutputEvidenceAsync(baselineWritten),
                masked = await OutputEvidenceAsync(maskedWritten),
            },
        }, json));

        _output.WriteLine($"Baseline output: {baselineWritten.Directory}");
        _output.WriteLine($"Guide/mask output: {maskedWritten.Directory}");
        _output.WriteLine($"Execution and hash evidence: {evidencePath}");
        _output.WriteLine($"guide SSIM={guideSsim:F3}; preserved-left SSIM={preservedLeftSsim:F3}; "
            + $"generated-right SSIM={generatedRightSsim:F3}; seam jump={seamJump:F3}; "
            + $"audio source diff early={earlyAudioDiffDb:F2} dBFS late={lateAudioDiffDb:F2} dBFS");

        // Thresholds leave headroom around this test's final-tree 2026-08-30 exact-artifact run (0.827 guide,
        // 0.866 preserved-left, 0.619 generated-right, 3.066 seam, and -53.96/-32.69 dBFS early/late audio
        // differences). The continuous horizontal mask reaches 0.25 within the measured left quarter, so that
        // region is intentionally not an all-black/exact-preservation assertion.
        Assert.True(guideSsim >= 0.80, $"frame-17 guide adherence regressed: SSIM={guideSsim:F4}.");
        Assert.True(preservedLeftSsim >= 0.82,
            $"video mask no longer preserves its black region: SSIM={preservedLeftSsim:F4}.");
        Assert.True(generatedRightSsim <= 0.75,
            $"video mask's generated region remains too source-bound: SSIM={generatedRightSsim:F4}.");
        Assert.True(preservedLeftSsim >= generatedRightSsim + 0.15,
            "video mask no longer separates preserved and generated regions.");
        Assert.True(seamJump <= 10d,
            $"the former x=460 guide/mask patch seam returned: adjacent-column jump={seamJump:F3}.");
        Assert.True(earlyAudioDiffDb <= -35d,
            $"audio mask's preserved region diverged from its source: {earlyAudioDiffDb:F2} dBFS.");
        Assert.True(lateAudioDiffDb >= earlyAudioDiffDb + 8d,
            $"audio mask no longer separates preserved/generated regions ({earlyAudioDiffDb:F2}/{lateAudioDiffDb:F2} dBFS).");
        Assert.True(postOnsetRms > 1e-4,
            $"soundtrack is silent after the expected H3 onset: RMS={postOnsetRms:E3}.");
    }

    private static VideoRequest Request(long seed, ComponentOverrides components) => new()
    {
        Prompt = Prompt,
        Width = Width,
        Height = Height,
        Frames = Frames,
        Fps = Fps,
        Steps = Steps,
        Seed = seed,
        Components = components,
    };

    /// <summary>Exercises the validation-pending implementation from the friend test assembly while proving the
    /// public service plan remains release-blocked. No shipped assembly can construct this trusted handoff.</summary>
    private static VideoPlan BindValidationCanaryPlan(VideoPlan releasePlan)
    {
        VideoPlanIssue[] retainedIssues = releasePlan.Issues
            .Where(issue => issue.Code is not "video.h3_expansion.release_blocked"
                and not "video.feature.unsupported")
            .ToArray();
        VideoPlan validationPlan = releasePlan with
        {
            Profile = releasePlan.Profile with
            {
                Features = releasePlan.Profile.Features
                    | VideoFeatures.Guides
                    | VideoFeatures.VideoDenoiseMask
                    | VideoFeatures.AudioDenoiseMask,
            },
            Issues = retainedIssues,
        };
        return VideoRequestExecutionBinding.BindPlan(validationPlan);
    }

    private void AssertValidDensePlan(VideoPlan plan, long seed)
    {
        foreach (VideoPlanIssue issue in plan.Issues)
        {
            _output.WriteLine($"plan {issue.Severity}: {issue.Code}: {issue.Message}");
        }
        Assert.True(plan.IsValid, string.Join(Environment.NewLine,
            plan.Issues.Select(issue => $"{issue.Code}: {issue.Message}")));
        Assert.Equal("minimax-h3-fl2va-base-fp8-scaled", plan.Profile.Id);
        Assert.Equal(VideoTaskFamily.Fl2Va, plan.Profile.Task);
        Assert.Equal(VideoAccelerationKind.None, plan.Profile.Acceleration);
        Assert.Equal(VideoAttentionKind.Dense, plan.Profile.Attention);
        Assert.Equal(seed, plan.EffectiveSettings.Seed);
    }

    private static void AssertExecution(VideoGenerationResult result, long seed)
    {
        VideoExecutionSummary execution = Assert.IsType<VideoExecutionSummary>(result.Execution);
        Assert.Equal("minimax-h3-fl2va-base-fp8-scaled", execution.ProfileId);
        Assert.Equal(VideoTaskFamily.Fl2Va, execution.Task);
        Assert.Equal(VideoAccelerationKind.None, execution.Acceleration);
        Assert.Equal(VideoAttentionKind.Dense, execution.Attention);
        Assert.Equal(Width, execution.Width);
        Assert.Equal(Height, execution.Height);
        Assert.Equal(Frames, execution.Frames);
        Assert.Equal(Fps, execution.Fps);
        Assert.Equal(seed, execution.Seed);
        Assert.Equal(Steps, execution.Steps);
        Assert.Equal(1f, execution.CfgScale);
        Assert.Equal(12f, execution.FlowShift);
        Assert.Equal(3f, execution.AudioFlowShift);
        Assert.Equal("euler", execution.Sampler);
        Assert.Equal("normal", execution.Scheduler);
        Assert.Equal("None", execution.ExecutionPath);
        Assert.Contains("transformer", execution.ComponentFormats.Keys);
        Assert.Contains("videoVae", execution.ComponentFormats.Keys);
        Assert.Contains("audioVae", execution.ComponentFormats.Keys);
        Assert.Contains("textEncoder", execution.ComponentFormats.Keys);
    }

    private static void AssertCoherent(VideoGenerationResult result, string label)
    {
        Assert.Equal(Frames, result.Frames.Count);
        Assert.Equal(Frames, result.Frames.Select(frame => Convert.ToHexString(
            SHA256.HashData(frame.Rgb))).Distinct(StringComparer.Ordinal).Count());
        foreach (VideoFrame frame in result.Frames)
        {
            Assert.Equal(Width, frame.Width);
            Assert.Equal(Height, frame.Height);
            Assert.Equal(Width * Height * 3, frame.Rgb.Length);
            Assert.True(frame.Rgb.Count(value => value != 0) > frame.Rgb.Length / 10,
                $"{label} frame {frame.Index} is effectively black.");
            Assert.True(frame.Rgb.Count(value => value != 255) > frame.Rgb.Length / 10,
                $"{label} frame {frame.Index} is effectively white.");
        }
        AudioBuffer audio = Assert.IsType<AudioBuffer>(result.Audio);
        Assert.Equal(2, audio.ChannelCount);
        Assert.Equal(32_000, audio.SampleRate);
        Assert.True(audio.Seconds > 1d, $"{label} clip ends before H3's audio-onset window.");
        Assert.All(audio.Channels, channel => Assert.All(channel, sample => Assert.True(float.IsFinite(sample))));
    }

    private static VideoOutputWriter.Written Persist(
        VideoGenerationResult result, string outputRoot, string slug)
    {
        VideoFrame first = result.Frames[0];
        VideoOutputWriter.Written written = VideoOutputWriter.Write(
            result.Frames.Select(frame => frame.Rgb).ToArray(),
            first.Width,
            first.Height,
            outputRoot,
            slug,
            result.Audio,
            result.Fps ?? Fps);
        JsonSerializerOptions json = new(JsonSerializerDefaults.Web) { WriteIndented = true };
        json.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
        File.WriteAllText(Path.Combine(written.Directory, "execution-summary.json"),
            JsonSerializer.Serialize(result.Execution, json));
        return written;
    }

    private async Task<string> RequireSha256Async(string path, string expected)
    {
        _output.WriteLine($"Resolving exact SHA-256 through the production path/size/mtime cache for "
            + $"{Path.GetFileName(path)} ({new FileInfo(path).Length:N0} bytes).");
        string actual = await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None);
        _output.WriteLine($"SHA-256 {Path.GetFileName(path)}: {actual}");
        Assert.Equal(expected, actual);
        return actual;
    }

    private static async Task<string> Sha256Async(string path)
    {
        await using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            4 * 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        using SHA256 sha = SHA256.Create();
        return Convert.ToHexString(await sha.ComputeHashAsync(stream)).ToLowerInvariant();
    }

    private static ImageData HorizontalMask()
    {
        byte[] rgb = new byte[Width * Height * 3];
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                byte value = (byte)Math.Round(255d * x / (Width - 1));
                int offset = (y * Width + x) * 3;
                rgb[offset] = value;
                rgb[offset + 1] = value;
                rgb[offset + 2] = value;
            }
        }
        return new ImageData { Rgb = rgb, Width = Width, Height = Height };
    }

    private static float[] AudioMaskValues() =>
    [
        .. Enumerable.Repeat(0f, 20),
        .. Enumerable.Range(1, 9).Select(value => value / 10f),
        .. Enumerable.Repeat(1f, 36),
    ];

    private static AudioBuffer CropAudio(AudioBuffer source, double startSeconds)
    {
        int start = Math.Clamp((int)Math.Round(startSeconds * source.SampleRate), 0, source.FrameCount);
        float[][] channels = new float[source.ChannelCount][];
        for (int channel = 0; channel < channels.Length; channel++)
        {
            channels[channel] = source.Channels[channel][start..source.FrameCount];
        }
        return new AudioBuffer { Channels = channels, SampleRate = source.SampleRate };
    }

    private static AudioClip Wav(AudioBuffer audio) => new()
    {
        Data = AudioClipCodec.EncodeWav(audio),
        Format = "wav",
    };

    private static double MeanRegionSsim(
        IReadOnlyList<VideoFrame> source, IReadOnlyList<VideoFrame> output, int x, int width)
    {
        double sum = 0d;
        for (int i = 0; i < source.Count; i++)
        {
            byte[] a = Crop(source[i].Rgb, x, width);
            byte[] b = Crop(output[i].Rgb, x, width);
            sum += Ssim.Compute(a, b, width, Height);
        }
        return sum / source.Count;
    }

    private static byte[] Crop(byte[] rgb, int x, int width)
    {
        byte[] cropped = new byte[width * Height * 3];
        int sourceStride = Width * 3;
        int destinationStride = width * 3;
        for (int y = 0; y < Height; y++)
        {
            Buffer.BlockCopy(rgb, y * sourceStride + x * 3,
                cropped, y * destinationStride, destinationStride);
        }
        return cropped;
    }

    private static double MeanAdjacentColumnJump(IReadOnlyList<VideoFrame> frames, int x)
    {
        double sum = 0d;
        long count = 0;
        foreach (VideoFrame frame in frames)
        {
            for (int y = 0; y < Height; y++)
            {
                int left = (y * Width + x - 1) * 3;
                int right = left + 3;
                for (int channel = 0; channel < 3; channel++)
                {
                    sum += Math.Abs(frame.Rgb[left + channel] - frame.Rgb[right + channel]);
                    count++;
                }
            }
        }
        return sum / count;
    }

    private static double DifferenceDb(AudioBuffer source, AudioBuffer output, double start, double end)
    {
        int sampleRate = Math.Min(source.SampleRate, output.SampleRate);
        int from = Math.Max(0, (int)Math.Floor(start * sampleRate));
        int to = Math.Min(Math.Min(source.FrameCount, output.FrameCount), (int)Math.Ceiling(end * sampleRate));
        double sum = 0d;
        long count = 0;
        int channels = Math.Min(source.ChannelCount, output.ChannelCount);
        for (int channel = 0; channel < channels; channel++)
        {
            for (int i = from; i < to; i++)
            {
                double difference = output.Channels[channel][i] - source.Channels[channel][i];
                sum += difference * difference;
                count++;
            }
        }
        return 20d * Math.Log10(Math.Max(Math.Sqrt(sum / Math.Max(1, count)), 1e-12));
    }

    private static double Rms(AudioBuffer audio, double start, double end)
    {
        int from = Math.Max(0, (int)Math.Floor(start * audio.SampleRate));
        int to = Math.Min(audio.FrameCount, (int)Math.Ceiling(end * audio.SampleRate));
        double sum = 0d;
        long count = 0;
        foreach (float[] channel in audio.Channels)
        {
            for (int i = from; i < to; i++)
            {
                sum += channel[i] * channel[i];
                count++;
            }
        }
        return Math.Sqrt(sum / Math.Max(1, count));
    }

    private static async Task<object> OutputEvidenceAsync(VideoOutputWriter.Written written) => new
    {
        directory = written.Directory,
        mp4 = written.Mp4Path,
        mp4Sha256 = written.Mp4Path is null ? null : await Sha256Async(written.Mp4Path),
        audio = written.AudioPath,
        audioSha256 = written.AudioPath is null ? null : await Sha256Async(written.AudioPath),
    };

    private static bool FfmpegAvailable()
    {
        try
        {
            using Process process = Process.Start(new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = "-version",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            })!;
            if (!process.WaitForExit(5_000))
            {
                process.Kill(entireProcessTree: true);
                return false;
            }
            return process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private sealed class LoggingProgress(ITestOutputHelper output) : IProgress<StepPreview>
    {
        public void Report(StepPreview value) =>
            output.WriteLine($"denoise {value.Step}/{value.TotalSteps}");
    }
}
