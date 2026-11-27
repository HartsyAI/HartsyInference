using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>Pixels to reply on the real MOSS-VL GGUF: an 8-bit RGB image goes through <see cref="MossVlImageProcessor"/>, the vision
/// tower read from the GGUF, and the 48-layer decoder, all on one GPU backend. The image is a committed synthetic scene (a red disc
/// and a blue square on white), so the reply can be checked by its words. <c>HARTSY_MOSS_VL_INDEX</c> supplies the config and
/// tokenizer; <c>HARTSY_MOSS_VL_BACKEND</c> picks <c>vulkan</c> (default) or <c>cuda</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MossVlImageToReplyTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string GgufVariable = "HARTSY_MOSS_VL_GGUF";
    private const string BackendVariable = "HARTSY_MOSS_VL_BACKEND";

    [RealWeightsFact(CheckpointVariable, GgufVariable)]
    public void ProcessedPixelsProduceAReplyAboutTheScene()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        MossVlConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            config = MossVlConfig.FromJson(doc.RootElement);
        ILlmTokenizer tokenizer;
        using (FileStream stream = File.OpenRead(Path.Combine(checkpoint, "tokenizer.json")))
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
        MossVlImageProcessor processor = MossVlImageProcessor.FromFile(Path.Combine(checkpoint, "preprocessor_config.json"));

        byte[] rgb = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossVl", "Preprocess", "shapes_256x256.rgb"));
        using VisionPatchGrid grid = processor.Preprocess(new RawRgbFrame(rgb, 256, 256));

        string gguf = Environment.GetEnvironmentVariable(GgufVariable)!;
        string backendName = (Environment.GetEnvironmentVariable(BackendVariable) ?? "vulkan").ToLowerInvariant();
        using IBackend backend = backendName == "cuda" ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
        using MossVlVisionTower tower = MossVlVisionTower.OpenGguf(gguf, backend);
        using MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(gguf, config, new MossVlLoadOptions { Backend = backend, LowVramQuant = true });

        Qwen3VlVisionEncoder.VisionOutput vision = tower.Encoder.Forward(backend, grid.PixelValues, grid.GridT, grid.GridH, grid.GridW);
        try
        {
            MossVlGenerator generator = new(decoder, config, tokenizer, backend);
            const string Question = "What shapes and colors are in this image?";
            string reply = generator.Generate(Question, vision.MergedTokens, grid.MergedHeight, grid.MergedWidth, maxTokens: 48,
                sampling: SamplingOptions.GreedyPreset);

            Console.WriteLine($"MOSS-VL pixels to reply ({backendName}, greedy, grid {grid.GridH}x{grid.GridW}): \"{Question}\"");
            Console.WriteLine($"REPLY: {reply}");
            string lower = reply.ToLowerInvariant();
            Assert.Contains("red", lower);
            Assert.Contains("blue", lower);
        }
        finally
        {
            vision.MergedTokens.Dispose();
        }
    }
}
