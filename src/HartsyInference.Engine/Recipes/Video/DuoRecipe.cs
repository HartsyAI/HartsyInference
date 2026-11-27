using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Duo;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Duo v1 (inclusionAI) causal recipe: MiniMax-H3's Ref2VA video and audio DiT with the Duo LoRA merged at load, the H3 video and
/// audio VAE decoders, and a <see cref="DuoSession"/> per generation. No Intent model and no speech recognizer run: the Intent rows, the
/// reference latent and the noise come from a run folder (see <see cref="DuoRecipePipeline.RunFolderKey"/>), so no text encoder is loaded.
/// The distilled release needs a re-noise solver the session does not have, and is not matched. Not verified on released weights.</summary>
public sealed class DuoRecipe : IVideoRecipe
{
    /// <summary>The LoRA key prefix of a causal checkpoint; the distilled one uses <c>models.generator.</c>.</summary>
    internal const string CausalFamilyPrefix = "models.student.";

    private const string FamilyId = "duo-v1";
    private const long HugeDitBytes = 40L << 30;

    private static readonly string[] _familyIds = [FamilyId, "duo-v1-causal", "duo-v1-2-causal"];

    /// <inheritdoc/>
    public string Name => FamilyId;

    /// <inheritdoc/>
    /// <remarks>No encoder reads a prompt: the text condition is the Intent rows of the run folder.</remarks>
    public PromptProfile? Prompt { get; } = new() { Kind = "video", Negative = "none", Supports = [] };

    /// <summary>The released causal sampler (50 Euler steps, shift 1.0, no guidance) at the upstream demo's 576x768 portrait size and
    /// 24 fps. The frame count is eight 17-frame blocks; a run folder with another block count needs its own count.</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Revision = 1, Steps = DuoSchedule.CausalSteps, CfgScale = 1f, FlowShift = (float)DuoSchedule.CausalShift,
        Width = 576, Height = 768, Frames = 136, Fps = DuoCodecSpec.Fps,
    };

    /// <inheritdoc/>
    /// <remarks>Whole 17-frame blocks. An off-grid count is refused, since the run folder fixes the length.</remarks>
    public VideoConstraints Constraints { get; } = new VideoConstraints { Frames = new FrameGrid(17, 17, FrameNormalization.Reject) };

    /// <inheritdoc/>
    public VideoFeatures Supports => VideoFeatures.None;

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.None;

    /// <inheritdoc/>
    public bool RefusesUnhonouredPlacements => true;

    /// <inheritdoc/>
    public bool Matches(string familyId) => _familyIds.Contains(familyId, StringComparer.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public SamplingCapabilities.SamplingSupport SamplingSupportFor(string familyId, ResolvedModelVariant? variant) =>
        SamplingCapabilities.ForVideo(FamilyId);

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context) => Build(context, LoadCodecs);

    /// <summary><see cref="Construct"/> with the codec loader given, so tests can run the whole path with small decoders.</summary>
    internal DuoRecipePipeline Build(RecipeContext context, Func<DuoAssets, List<IDisposable>, DuoCodecs> loadCodecs)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(loadCodecs);
        // A text-encoder or VAE placement on another device is reported and run on the primary backend.
        UnhonouredBackends.Refuse(context, "Duo", warn: false);
        if (LoraResolver.Resolve(context.Loras).Count > 0)
        {
            throw new NotSupportedException("Duo does not take a user LoRA; the release's own LoRA is the only one merged.");
        }
        DuoAssets assets = DuoAssets.Resolve(context.CheckpointPath, context.Components);
        List<IDisposable> loaders = [];
        MiniMaxH3Transformer? transformer = null;
        try
        {
            CheckpointSource backbone = CheckpointSource.OpenShards(assets.BackboneShards);
            loaders.Add(backbone);
            transformer = MiniMaxH3Components.BuildTransformer(backbone, DType.F32, out MiniMaxH3Config config,
                out Dictionary<string, Tensor> weights);
            if (!config.HasAudio || config.UseAdalnCurves)
            {
                throw new NotSupportedException(
                    "Duo needs the full MiniMax-H3 Ref2VA backbone with audio and unpruned adaLN; this checkpoint is a different build.");
            }
            using (DuoLoraSource lora = OpenCausalLora(assets.LoraDirectory))
            {
                // Merged before LoadWeights, since device caches are keyed on the tensor a layer captured.
                DuoMergeOptions options = new() { Heads = config.NumAttentionHeads, HeadDim = config.AttentionHeadDim };
                loaders.AddRange(DuoRepacker.MergeInto(weights, lora, options, line => Logs.Info($"[DuoRecipe] {line}")));
            }
            transformer.LoadWeights(weights);

            long weightBytes = 0;
            foreach (Tensor tensor in transformer.EnumerateWeights())
            {
                weightBytes += tensor.DType.ComputeByteCount(tensor.ElementCount);
            }
            if (weightBytes >= HugeDitBytes)
            {
                // Cached casts of a build this size grow without bound.
                RecipeBackendFlags.DisableCacheWeightCasts(context, "DuoRecipe");
            }
            DuoCodecs codecs = loadCodecs(assets, loaders);
            long residentBytes = weightBytes + MiniMaxH3Recipe.MinimumActivationReserveBytes(config, context.Backend);
            return new DuoRecipePipeline(context.Backend, transformer, config, codecs, residentBytes, loaders);
        }
        catch
        {
            transformer?.Dispose();
            new CompositeDisposable([.. loaders]).Dispose();
            throw;
        }
    }

    private static DuoLoraSource OpenCausalLora(string directory)
    {
        try
        {
            return DuoLoraSource.FromDcp(directory, CausalFamilyPrefix);
        }
        catch (InvalidDataException ex)
        {
            throw new InvalidDataException(
                $"'{directory}' did not read as a causal Duo LoRA ('{CausalFamilyPrefix}' keys); the distilled release is not supported. "
                + ex.Message, ex);
        }
    }

    private static DuoCodecs LoadCodecs(DuoAssets assets, List<IDisposable> loaders)
    {
        (MiniMaxH3VideoVaeDecoder video, _) = MiniMaxH3Recipe.LoadVideoVae(assets.VideoVae, loaders);
        loaders.Add(video);
        (MiniMaxH3AudioVaeDecoder? audio, MiniMaxH3AudioVaeEncoder? audioEncoder) = MiniMaxH3Recipe.LoadAudioVae(assets.AudioVae, loaders);
        audioEncoder?.Dispose();
        ArgumentNullException.ThrowIfNull(audio);
        loaders.Add(audio);
        DuoCodecSpec spec = DuoCodecSpec.Get("duo_5_17");
        // Upstream decodes each audio block on its own, so the stream takes no context and no lookahead.
        return new DuoCodecs(backend => DuoVideoStreamDecoder.ForH3(video, backend),
            backend => DuoAudioStreamDecoder.ForH3(audio, backend, spec, lookaheadSteps: 0, contextSteps: 0),
            () => video.EnumerateWeights().Concat(audio.EnumerateWeights()));
    }
}
