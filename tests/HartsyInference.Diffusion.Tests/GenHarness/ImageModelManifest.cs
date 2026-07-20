using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests.GenHarness;

/// <summary>The single registry of image models the generation matrix drives. This replaces the per-model boilerplate
/// (load → tokenize → generate → validate → skip-when-absent) that was copy-pasted across ~30 <c>*GenerationTests</c>
/// files with one row per model plus a small delegate that reuses the model's existing pipeline wiring.
///
/// <para>To add a model: append one <see cref="ImageGenCase"/> whose <c>IsAvailable</c> probes its checkpoint +
/// tokenizer paths (from <see cref="TestPaths"/>) and whose <c>Generate</c> loads the pipeline and returns pixels.
/// Keep the delegate thin — call the model's own <c>PipelineFactory</c> loader / constructor; do not re-implement
/// conversion here.</para></summary>
public static class ImageModelManifest
{
    public static IReadOnlyList<ImageGenCase> All { get; } = new List<ImageGenCase>
    {
        new ImageGenCase
        {
            Name = "SDXL",
            IsAvailable = () =>
                File.Exists(TestPaths.Sdxl.SingleFile)
                && File.Exists(TestPaths.Tokenizers.ClipVocab)
                && File.Exists(TestPaths.Tokenizers.ClipMerges),
            Generate = GenerateSdxl,
        },
        // Template for the next model (uncomment + wire against its own loader):
        // new ImageGenCase
        // {
        //     Name = "SD15",
        //     IsAvailable = () => File.Exists(TestPaths.Sd15.SingleFile) && File.Exists(TestPaths.Tokenizers.ClipVocab),
        //     Generate = GenerateSd15,
        // },
    };

    private static GenImage GenerateSdxl(IBackend backend, ImageGenRequest req)
    {
        using SdxlPipeline pipeline = PipelineFactory.LoadSdxl(TestPaths.Sdxl.SingleFile, backend);
        using ClipTokenizer tokenizer = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);

        int[] promptL = tokenizer.Encode(req.Prompt);
        int[] negativeL = tokenizer.Encode("");
        int[] promptG = tokenizer.Encode(req.Prompt);
        int[] negativeG = tokenizer.Encode("");
        int eosG = ClipTokenizer.FindEosPosition(promptG);
        int negativeEosG = ClipTokenizer.FindEosPosition(negativeG);

        TextToImageRequest request = new()
        {
            Prompt = req.Prompt,
            NegativePrompt = "",
            Width = req.Width,
            Height = req.Height,
            Steps = req.Steps,
            CfgScale = req.CfgScale,
            Seed = req.Seed,
        };

        (byte[] rgb, int width, int height, int seed) = pipeline.GenerateFromTokens(
            promptL, negativeL, promptG, negativeG, eosG, negativeEosG, request);
        return new GenImage(rgb, width, height, seed);
    }
}
