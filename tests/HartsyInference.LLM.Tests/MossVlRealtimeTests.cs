using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>A live session on the real MOSS-VL GGUF: the committed five-frame clip (a red disc moving left to right) pushed as a camera
/// would, with a question, through the generic loop, the MOSS decoder adapter and the GGUF tower on one GPU.
/// <c>HARTSY_MOSS_VL_INDEX</c> supplies the config and tokenizer; <c>HARTSY_MOSS_VL_BACKEND</c> picks <c>vulkan</c> (default) or <c>cuda</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MossVlRealtimeTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string GgufVariable = "HARTSY_MOSS_VL_GGUF";
    private const string BackendVariable = "HARTSY_MOSS_VL_BACKEND";

    [RealWeightsFact(CheckpointVariable, GgufVariable)]
    public async Task ALiveSessionAnswersAboutPushedFrames()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        MossVlConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            config = MossVlConfig.FromJson(doc.RootElement);
        ILlmTokenizer tokenizer;
        using (FileStream stream = File.OpenRead(Path.Combine(checkpoint, "tokenizer.json")))
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
        string gguf = Environment.GetEnvironmentVariable(GgufVariable)!;
        string backendName = (Environment.GetEnvironmentVariable(BackendVariable) ?? "vulkan").ToLowerInvariant();
        using IBackend backend = backendName == "cuda" ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
        using MossVlVisionTower tower = MossVlVisionTower.OpenGguf(gguf, backend);
        using MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(gguf, config, new MossVlLoadOptions { Backend = backend, LowVramQuant = true });

        TowerEncoder encoder = new(tower, backend);
        RealtimeVlmSession session = new(new MossVlStreamingDecoder(decoder, config, backend), encoder,
            new MossVlRealtimeProtocol(config, tokenizer), tokenizer);
        List<RealtimeVlmEvent> events = [];
        try
        {
            string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossVl", "Video");
            for (int i = 0; i < 3; i++)
                session.PushFrame(new RawRgbFrame(File.ReadAllBytes(Path.Combine(dir, $"frame_{i:000}.rgb")), 256, 256), i);
            session.PushPrompt("What color is the circle?");
            session.Start(systemPrompt: null, initialPrompt: "");

            using CancellationTokenSource timeout = new(TimeSpan.FromMinutes(5));
            await foreach (RealtimeVlmEvent e in session.ReadAsync(timeout.Token))
            {
                events.Add(e);
                Assert.NotEqual(RealtimeVlmEventKind.Faulted, e.Kind);
                // The answer to the prompt has started and ended in silence.
                if (events.Any(x => x.Kind == RealtimeVlmEventKind.Text) && e.Kind == RealtimeVlmEventKind.Silence) break;
            }
        }
        finally
        {
            RealtimeVlmStats stats = session.Stats;
            await session.DisposeAsync();
            Console.WriteLine($"MOSS-VL live ({backendName}): {string.Concat(events.Select(e => e.Kind == RealtimeVlmEventKind.Text ? e.Text : $"[{e.Kind}]"))}");
            Console.WriteLine($"Stats: {stats}");
            Assert.Equal(encoder.VisionTokens, stats.VisionTokens);
            Assert.Equal(3, stats.FramesProcessed);
        }
        Assert.Contains(events, e => e.Kind == RealtimeVlmEventKind.RoundStart);
        string reply = string.Concat(events.Where(e => e.Kind == RealtimeVlmEventKind.Text).Select(e => e.Text)).ToLowerInvariant();
        Assert.Contains("red", reply);
    }

    /// <summary>The tower on the session's backend; counts the vision tokens it hands over (merged tokens plus one separator per frame).</summary>
    private sealed class TowerEncoder(MossVlVisionTower tower, IBackend backend) : IStreamingFrameEncoder
    {
        public int VisionTokens { get; private set; }

        public IReadOnlyList<VisionFrameFeatures> Encode(IReadOnlyList<TimedRgbFrame> frames, CancellationToken cancel)
        {
            VisionPatchGrid[] grids = MossVlImageProcessor.Default.PreprocessBatch([.. frames.Select(f => f.Frame)]);
            try
            {
                VisionFrameFeatures[] features = new VisionFrameFeatures[grids.Length];
                for (int i = 0; i < grids.Length; i++)
                {
                    Tensor merged = tower.Encoder.Forward(backend, grids[i].PixelValues, grids[i].GridT, grids[i].GridH, grids[i].GridW).MergedTokens;
                    features[i] = new VisionFrameFeatures(merged, grids[i].MergedHeight, grids[i].MergedWidth, frames[i].TimestampSeconds);
                    VisionTokens += grids[i].MergedHeight * grids[i].MergedWidth + 1;
                }
                return features;
            }
            finally
            {
                foreach (VisionPatchGrid g in grids) g.Dispose();
            }
        }
    }
}
