using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.Registry;
using HartsyInference.Tokenizers;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>Text-to-image diffusion. Auto-detects the checkpoint architecture and builds the pipeline via
/// <see cref="PipelineFactory"/> (SDXL end-to-end today; other families surface a clear not-yet-wired error).</summary>
public sealed class ImageHandler : IModalityHandler
{
    /// <inheritdoc/>
    public Modality Modality => Modality.Image;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No SDXL checkpoint found. Pass a .safetensors checkpoint via --model-path, or `hartsy pull` it first.");
        }

        ModelArchitecture arch = PipelineFactory.DetectArchitecture(spec.LocalPath);
        progress.Stage($"Detected architecture: {arch}");
        progress.Stage("Loading pipeline (converting + wiring components) …");

        DiffusionPipelineBase pipeline = PipelineFactory.LoadAuto(spec.LocalPath, backend);
        if (pipeline is not SdxlPipeline sdxl)
        {
            pipeline.Dispose();
            throw new NotSupportedException(
                $"'{spec.Requested}' is {pipeline.GetType().Name}; the CLI's image path currently drives SDXL pipelines. " +
                "Other diffusion families are being wired.");
        }

        string id = spec.Catalog?.Id ?? Path.GetFileNameWithoutExtension(spec.LocalPath);
        return new ImageRunner(id, sdxl, new ClipTokenizer());
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        ImageRunner image = (ImageRunner)runner;

        string negative = parameters.Get("negative") ?? "";
        int width = parameters.GetInt("width", 1024);
        int height = parameters.GetInt("height", 1024);
        int steps = parameters.GetInt("steps", 20);
        int seedParam = parameters.GetInt("seed", -1);

        int[] tokensL = image.Tokenizer.Encode(prompt);
        int[] negL = image.Tokenizer.Encode(negative);
        int[] tokensG = image.Tokenizer.Encode(prompt);
        int[] negG = image.Tokenizer.Encode(negative);
        int eosG = ClipTokenizer.FindEosPosition(tokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negG);

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = parameters.GetFloat("cfg", 7.5f),
            Seed = seedParam < 0 ? null : seedParam,
        };

        progress.BeginSteps("denoise", steps);
        (byte[] rgb, int outWidth, int outHeight, int usedSeed) = image.Pipeline.GenerateFromTokens(
            tokensL, negL, tokensG, negG, eosG, negEosG, request,
            p => progress.Step(p.Step, $"{p.ElapsedMs:F0}ms"));
        progress.EndSteps();

        byte[] png = PngEncoder.Encode(rgb, outWidth, outHeight);
        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Image,
            FileBytes = png,
            Extension = "png",
            Text = $"{outWidth}x{outHeight} image (seed {usedSeed})",
            PreviewRgb = rgb,
            PreviewWidth = outWidth,
            PreviewHeight = outHeight,
        };
        artifact.Meta["model"] = image.ModelId;
        artifact.Meta["size"] = $"{outWidth}x{outHeight}";
        artifact.Meta["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["steps"] = steps.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }
}
