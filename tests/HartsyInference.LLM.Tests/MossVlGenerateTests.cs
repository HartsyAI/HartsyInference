using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Models.MossVl;
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
    private const int MergedHeight = 8, MergedWidth = 12;
    private const string Question = "Describe the image.";

    [RealWeightsFact(CheckpointVariable, VisionVariable, GgufVariable)]
    public void GgufFullModelGeneratesAReply()
    {
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        MossVlConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            config = MossVlConfig.FromJson(doc.RootElement);

        (long[] shape, float[] data) = NpyReader.ReadFloat32(Environment.GetEnvironmentVariable(VisionVariable)!);
        using Tensor merged = new(new TensorShape(shape[0], shape[1]), DType.F32);
        data.AsSpan().CopyTo(merged.AsSpan<float>());

        using FileStream tokenizerStream = File.OpenRead(Path.Combine(checkpoint, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream);

        string backendName = (Environment.GetEnvironmentVariable(BackendVariable) ?? "vulkan").ToLowerInvariant();
        bool cuda = backendName == "cuda";
        // The backend is declared first so the decoder, which holds device tensors, is disposed before it.
        using IBackend backend = cuda ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
        using MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(Environment.GetEnvironmentVariable(GgufVariable)!, config,
            keepQuantized: true, lowVramQuant: cuda);
        MossVlGenerator generator = new(decoder, config, tokenizer, backend);

        string reply = generator.Generate(Question, merged, MergedHeight, MergedWidth, maxTokens: 48);

        Console.WriteLine($"MOSS-VL GGUF generation ({backendName}, {config.Text.NumLayers} layers, packed quant weights): question=\"{Question}\"");
        Console.WriteLine($"REPLY: {reply}");
        Assert.False(string.IsNullOrWhiteSpace(reply));
    }
}
