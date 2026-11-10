using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>Video on the real MOSS-VL GGUF: a committed five-frame clip (a red disc moving left to right on white) through the video
/// preprocessing, the GGUF tower and the decoder on one GPU. <c>HARTSY_MOSS_VL_INDEX</c> supplies the config and tokenizer;
/// <c>HARTSY_MOSS_VL_BACKEND</c> picks <c>vulkan</c> (default) or <c>cuda</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MossVlVideoTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string GgufVariable = "HARTSY_MOSS_VL_GGUF";
    private const string BackendVariable = "HARTSY_MOSS_VL_BACKEND";
    private const int Frames = 5;

    [RealWeightsFact(CheckpointVariable, GgufVariable)]
    public void TheReplyDescribesTheMotion()
    {
        using Fixture f = Fixture.Load();
        MossVlGenerator generator = new(f.Decoder, f.Config, f.Tokenizer, f.Backend);
        MossVlGeneration reply = generator.GenerateVideo("Describe what happens in this video.", f.Features, maxTokens: 60,
            SamplingOptions.GreedyPreset, onDelta: null, CancellationToken.None);

        Console.WriteLine($"MOSS-VL video ({Frames} frames, {f.BackendName}, greedy): {reply.Text}");
        string lower = reply.Text.ToLowerInvariant();
        Assert.Contains("red", lower);
        Assert.True(lower.Contains("move") || lower.Contains("moving"), $"The reply does not describe motion: {reply.Text}");
    }

    /// <summary>A stream built in segments must equal the same tokens in one forward: the prompt with all frames at once, against the
    /// prompt cut after the second frame and run as two segments. Positions, visibility and the appended vision keys all have to
    /// line up for the last logits to agree.</summary>
    [RealWeightsFact(CheckpointVariable, GgufVariable)]
    public void SegmentsAppendToTheSameStateAsOneForward()
    {
        using Fixture f = Fixture.Load();
        MossVlRealtimeProtocol protocol = new(f.Config, f.Tokenizer);
        int[] ids = f.Tokenizer.Encode(protocol.OfflinePrompt(0, f.Features.Select(x => x.TimestampSeconds).ToArray(),
            "Describe what happens in this video."), addSpecial: true);
        MossVlImageFeatures[] all = [.. f.Features.Select(x => new MossVlImageFeatures(x.MergedTokens, x.MergedHeight, x.MergedWidth))];

        float[] whole;
        using (MossVlDecodeSession one = new(f.Decoder, f.Config, f.Backend))
            whole = one.ForwardLogits(ids, all);

        int cut = Array.IndexOf(ids, f.Config.ImageTokenId, Array.IndexOf(ids, f.Config.ImageTokenId) + 1) + 1;   // after frame 2
        float[] split;
        using (MossVlDecodeSession two = new(f.Decoder, f.Config, f.Backend))
        {
            two.ForwardLogits(ids.AsSpan(0, cut), all[..2]);
            split = two.ForwardLogits(ids.AsSpan(cut), all[2..]);
            Assert.Equal(ids.Length, two.TextTokens);
        }

        double maxDiff = 0, maxRef = 0;
        for (int i = 0; i < whole.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(whole[i] - split[i]));
            maxRef = Math.Max(maxRef, Math.Abs(whole[i]));
        }
        Console.WriteLine($"MOSS-VL segment equivalence ({f.BackendName}): max|diff| {maxDiff:G4}, max|logit| {maxRef:G4}, relative {maxDiff / maxRef:G4}.");
        Assert.Equal(ArgMax(whole), ArgMax(split));
        Assert.True(maxDiff / maxRef < 2e-3, $"Segmented logits differ from one forward by {maxDiff / maxRef:G4} relative.");
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>The clip's frames encoded on the GPU, with the decoder, config and tokenizer they feed.</summary>
    private sealed class Fixture : IDisposable
    {
        private readonly List<Tensor> _merged = [];
        private readonly MossVlVisionTower _tower;

        private Fixture(IBackend backend, string backendName, MossVlConfig config, ILlmTokenizer tokenizer, MossVlVisionTower tower,
            MossVlTextDecoder decoder)
        {
            Backend = backend;
            BackendName = backendName;
            Config = config;
            Tokenizer = tokenizer;
            _tower = tower;
            Decoder = decoder;
        }

        public IBackend Backend { get; }
        public string BackendName { get; }
        public MossVlConfig Config { get; }
        public ILlmTokenizer Tokenizer { get; }
        public MossVlTextDecoder Decoder { get; }
        public List<VisionFrameFeatures> Features { get; } = [];

        public static Fixture Load()
        {
            string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
            MossVlConfig config;
            using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
                config = MossVlConfig.FromJson(doc.RootElement);
            ILlmTokenizer tokenizer;
            using (FileStream stream = File.OpenRead(Path.Combine(checkpoint, "tokenizer.json")))
                tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
            VideoFrameSamplingOptions sampling;
            using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "video_preprocessor_config.json"))))
                sampling = MossVlImageProcessor.VideoOptionsFromJson(doc.RootElement);

            string gguf = Environment.GetEnvironmentVariable(GgufVariable)!;
            string backendName = (Environment.GetEnvironmentVariable(BackendVariable) ?? "vulkan").ToLowerInvariant();
            IBackend backend = backendName == "cuda" ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
            MossVlVisionTower tower = MossVlVisionTower.OpenGguf(gguf, backend);
            MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(gguf, config, new MossVlLoadOptions { Backend = backend, LowVramQuant = true });
            Fixture fixture = new(backend, backendName, config, tokenizer, tower, decoder);

            string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossVl", "Video");
            RawRgbFrame[] frames = [.. Enumerable.Range(0, Frames).Select(i => new RawRgbFrame(File.ReadAllBytes(Path.Combine(dir, $"frame_{i:000}.rgb")), 256, 256))];
            VisionPatchGrid[] grids = MossVlImageProcessor.Default.PreprocessVideo(frames, sampling);
            try
            {
                for (int i = 0; i < grids.Length; i++)
                {
                    Tensor merged = tower.Encoder.Forward(backend, grids[i].PixelValues, grids[i].GridT, grids[i].GridH, grids[i].GridW).MergedTokens;
                    fixture._merged.Add(merged);
                    fixture.Features.Add(new VisionFrameFeatures(merged, grids[i].MergedHeight, grids[i].MergedWidth, TimestampSeconds: i));
                }
            }
            finally
            {
                foreach (VisionPatchGrid g in grids) g.Dispose();
            }
            return fixture;
        }

        public void Dispose()
        {
            foreach (Tensor t in _merged) t.Dispose();
            Decoder.Dispose();
            _tower.Dispose();
            Backend.Dispose();
        }
    }
}
