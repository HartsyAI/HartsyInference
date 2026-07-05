using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Interactive.ActionEncoders;
using HartsyInference.Interactive.Pipelines;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>Action-conditioned world-model rollout with Oasis (validation-pending). Loads the DiT (--model-path) and
/// ViT-VAE (aux "vae-path"), takes a first-frame image as the "prompt", drives a canned "move forward" action plan,
/// and writes the resulting frames as a BMP sequence.</summary>
public sealed class InteractiveHandler : IModalityHandler
{
    // Oasis VPT key index 11 is "forward" (see OasisActionEncoder / the reference rollout test).
    private const int ForwardKeyIndex = 11;

    /// <inheritdoc/>
    public Modality Modality => Modality.Interactive;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (spec.LocalPath is null)
            throw new FileNotFoundException("No Oasis DiT checkpoint found. Pass it via --model-path.");
        if (!spec.Aux.TryGetValue("vae-path", out string? vaePath))
            throw new ArgumentException("Oasis needs its ViT-VAE checkpoint via --vae-path.");

        progress.Stage("Loading Oasis DiT …");
        (Dictionary<string, Core.Tensors.Tensor> ditWeights, SafeTensorsLoader ditLoader) = OasisCheckpointConverter.LoadAndConvert(spec.LocalPath);
        OasisDit dit = new OasisDit(new OasisDitConfig());
        dit.LoadWeights(ditWeights);

        progress.Stage("Loading Oasis ViT-VAE …");
        (Dictionary<string, Core.Tensors.Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = OasisCheckpointConverter.LoadAndConvert(vaePath);
        OasisVitVae vae = new OasisVitVae();
        vae.LoadWeights(vaeWeights);

        OasisPipeline pipeline = new OasisPipeline(backend, dit, vae);
        string id = spec.Catalog?.Id ?? "oasis";
        return new InteractiveRunner(id, pipeline, new IDisposable[] { ditLoader, vaeLoader });
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        InteractiveRunner interactive = (InteractiveRunner)runner;

        string imagePath = prompt.Trim();
        if (!File.Exists(imagePath))
            throw new FileNotFoundException($"First-frame image not found: {imagePath}");

        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(imagePath);

        int totalFrames = Math.Max(2, parameters.GetInt("frames", 16));
        int steps = parameters.GetInt("steps", 10);
        int seedParam = parameters.GetInt("seed", -1);

        byte[] keys = new byte[OasisActionEncoder.KeyCount];
        keys[ForwardKeyIndex] = 1;
        float[][] actions = new float[totalFrames - 1][];
        for (int i = 0; i < actions.Length; i++)
        {
            actions[i] = new float[OasisActionEncoder.ActionDim];
            OasisActionEncoder.BuildRow(keys, cameraX: 0f, cameraY: 0f, actions[i]);
        }

        progress.BeginSteps("rollout", totalFrames);
        (byte[][] frames, int outWidth, int outHeight, int usedSeed) = interactive.Pipeline.Generate(
            rgb, width, height, actions, totalFrames, ddimSteps: steps, seed: seedParam < 0 ? null : seedParam,
            onProgress: p => progress.Step(p.Step, $"{p.Step}/{p.TotalSteps}"));
        progress.EndSteps();

        string baseDir = parameters.OutputDir ?? RepoPaths.OutputRoot();
        string dir = FrameWriter.WriteFrames(frames, outWidth, outHeight, baseDir, Path.GetFileNameWithoutExtension(imagePath));

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Video,
            Extension = "bmp",
            Text = $"{frames.Length} frames ({outWidth}x{outHeight}, action: forward) → {dir}",
        };
        artifact.Meta["model"] = interactive.ModelId;
        artifact.Meta["frames"] = frames.Length.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }
}
