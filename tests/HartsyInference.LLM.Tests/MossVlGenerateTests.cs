using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>End-to-end generation on the real MOSS-VL GGUF: the full 48-layer text stack on a GPU backend with the packed quantized
/// projections, the checkpoint's tokenizer, and the image features from the vision dump. Prints the reply so it can be read in the
/// test log. <c>HARTSY_MOSS_VL_BACKEND</c> picks the backend: <c>vulkan</c> (default) or <c>cuda</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MossVlGenerateTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string VisionVariable = "HARTSY_MOSS_VL_REF";
    private const string GgufVariable = "HARTSY_MOSS_VL_GGUF";
    private const string BackendVariable = "HARTSY_MOSS_VL_BACKEND";
    private const string GreedyVariable = "HARTSY_MOSS_VL_GREEDY";
    private const int MergedHeight = 8, MergedWidth = 12;
    // The dump's patch grid (1 x 16 x 24 patches), which merges to the 8 x 12 grid above.
    private const int PatchGridHeight = 16, PatchGridWidth = 24;
    private const string Question = "Describe the image.";

    [RealWeightsFact(CheckpointVariable, VisionVariable, GgufVariable)]
    public void GgufFullModelGeneratesAReply()
    {
        (MossVlConfig config, ILlmTokenizer tokenizer) = LoadConfigAndTokenizer();
        (long[] shape, float[] data) = NpyReader.ReadFloat32(Environment.GetEnvironmentVariable(VisionVariable)!);
        using Tensor merged = new(new TensorShape(shape[0], shape[1]), DType.F32);
        data.AsSpan().CopyTo(merged.AsSpan<float>());

        string backendName = (Environment.GetEnvironmentVariable(BackendVariable) ?? "vulkan").ToLowerInvariant();
        bool cuda = backendName == "cuda";
        // The backend is declared first so the decoder, which holds device tensors, is disposed before it.
        using IBackend backend = cuda ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
        using MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(Environment.GetEnvironmentVariable(GgufVariable)!, config,
            keepQuantized: true, lowVramQuant: cuda);
        MossVlGenerator generator = new(decoder, config, tokenizer, backend);

        // HARTSY_MOSS_VL_GREEDY=1 decodes by argmax, so two backends with the same math must print the same reply.
        bool greedy = Environment.GetEnvironmentVariable(GreedyVariable) == "1";
        string reply = generator.Generate(Question, merged, MergedHeight, MergedWidth, maxTokens: 48,
            sampling: greedy ? SamplingOptions.GreedyPreset : null);

        Console.WriteLine($"MOSS-VL GGUF generation ({backendName}, {(greedy ? "greedy" : "sampled")}, {config.Text.NumLayers} layers, packed quant weights): question=\"{Question}\"");
        Console.WriteLine($"REPLY: {reply}");
        Assert.False(string.IsNullOrWhiteSpace(reply));
    }

    /// <summary>The vision tower's output, not the dump, feeds the decoder: the patches run through the CPU tower (checked against the
    /// reference in <c>MossVlVisionParityTests</c>), its merged tokens go to the GPU decoder, and the greedy reply must describe the image.</summary>
    [RealWeightsFact(CheckpointVariable, VisionVariable, GgufVariable)]
    public void GgufVisionTowerFeedsGeneration()
    {
        (MossVlConfig config, ILlmTokenizer tokenizer) = LoadConfigAndTokenizer();
        string referenceDir = Path.GetDirectoryName(Environment.GetEnvironmentVariable(VisionVariable)!)!;
        (long[] patchShape, float[] patchData) = NpyReader.ReadFloat32(Path.Combine(referenceDir, "patches.npy"));
        using Tensor patches = new(new TensorShape(patchShape[0], patchShape[1]), DType.F32);
        patchData.AsSpan().CopyTo(patches.AsSpan<float>());

        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        using MossVlVisionTower tower = MossVlVisionTower.Open(checkpoint);
        using IBackend cpu = new CpuBackend();
        Qwen3VlVisionEncoder.VisionOutput vision = tower.Encoder.Forward(cpu, patches, 1, PatchGridHeight, PatchGridWidth);
        try
        {
            string backendName = (Environment.GetEnvironmentVariable(BackendVariable) ?? "vulkan").ToLowerInvariant();
            bool cuda = backendName == "cuda";
            using IBackend backend = cuda ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
            using MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(Environment.GetEnvironmentVariable(GgufVariable)!, config,
                keepQuantized: true, lowVramQuant: cuda);
            MossVlGenerator generator = new(decoder, config, tokenizer, backend);

            string reply = generator.Generate(Question, vision.MergedTokens, MergedHeight, MergedWidth, maxTokens: 48,
                sampling: SamplingOptions.GreedyPreset);

            Console.WriteLine($"MOSS-VL tower-to-reply generation ({backendName}, greedy, {config.Text.NumLayers} layers): question=\"{Question}\"");
            Console.WriteLine($"REPLY: {reply}");
            Assert.Contains("noise", reply);
        }
        finally
        {
            vision.MergedTokens.Dispose();
        }
    }

    /// <summary>Two images in one prompt on the real decoder: the cross-attention bias hides each image from the rows before its
    /// placeholder. There is no two-image reference, so this checks that the path runs and produces text.</summary>
    [RealWeightsFact(CheckpointVariable, VisionVariable, GgufVariable)]
    public void GgufTwoImagesGenerateAReply()
    {
        (MossVlConfig config, ILlmTokenizer tokenizer) = LoadConfigAndTokenizer();
        (long[] shape, float[] data) = NpyReader.ReadFloat32(Environment.GetEnvironmentVariable(VisionVariable)!);
        using Tensor merged = new(new TensorShape(shape[0], shape[1]), DType.F32);
        data.AsSpan().CopyTo(merged.AsSpan<float>());

        string backendName = (Environment.GetEnvironmentVariable(BackendVariable) ?? "vulkan").ToLowerInvariant();
        bool cuda = backendName == "cuda";
        using IBackend backend = cuda ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
        using MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(Environment.GetEnvironmentVariable(GgufVariable)!, config,
            keepQuantized: true, lowVramQuant: cuda);
        MossVlGenerator generator = new(decoder, config, tokenizer, backend);

        MossVlImageFeatures image = new(merged, MergedHeight, MergedWidth);
        string reply = generator.Generate("Are these two images the same?", [image, image], maxTokens: 32,
            sampling: SamplingOptions.GreedyPreset);

        Console.WriteLine($"MOSS-VL two-image generation ({backendName}, greedy, {config.Text.NumLayers} layers): the same image twice");
        Console.WriteLine($"REPLY: {reply}");
        Assert.False(string.IsNullOrWhiteSpace(reply));
    }

    private static (MossVlConfig Config, ILlmTokenizer Tokenizer) LoadConfigAndTokenizer()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        MossVlConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            config = MossVlConfig.FromJson(doc.RootElement);

        using FileStream tokenizerStream = File.OpenRead(Path.Combine(checkpoint, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream);
        return (config, tokenizer);
    }
}
