using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Duo;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Duo v1 (inclusionAI) recipe: MiniMax-H3's Ref2VA video and audio DiT with the Duo LoRA merged at load, the H3 video and
/// audio VAE decoders, and a <see cref="DuoSession"/> per generation. One instance serves the causal releases (Euler steps) and one the
/// distilled release (<paramref name="distilled"/>: clean estimate and re-noise). No Intent model and no speech recognizer run: the Intent rows, the
/// reference latent and the noise come from a run folder (see <see cref="DuoRecipePipeline.RunFolderKey"/>), so no text encoder is loaded.
/// Not verified on released weights.</summary>
public sealed class DuoRecipe(bool distilled = false) : IVideoRecipe
{
    /// <summary>The LoRA key prefix of a causal checkpoint.</summary>
    internal const string CausalFamilyPrefix = "models.student.";

    /// <summary>The LoRA key prefix of the distilled checkpoint.</summary>
    internal const string DistilledFamilyPrefix = "models.generator.";

    private const string FamilyId = "duo-v1";
    private const string DistilledId = "duo-v1-distill";
    private const long HugeDitBytes = 40L << 30;

    private static readonly string[] _causalIds = [FamilyId, "duo-v1-causal", "duo-v1-2-causal"];
    private static readonly string[] _distilledIds = [DistilledId];

    /// <inheritdoc/>
    public string Name => distilled ? DistilledId : FamilyId;

    /// <inheritdoc/>
    /// <remarks>No encoder reads a prompt: the text condition is the Intent rows of the run folder.</remarks>
    public PromptProfile? Prompt { get; } = new() { Kind = "video", Negative = "none", Supports = [] };

    /// <summary>The release's sampler (causal: 50 Euler steps at shift 1.0; distilled: 4 re-noised steps at shift 5.0; no guidance) at the
    /// upstream demo's 576x768 portrait size and 24 fps. The frame count is eight 17-frame blocks; a run folder with another block count
    /// needs its own count.</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Revision = 1, Steps = distilled ? DuoSchedule.DistillSteps : DuoSchedule.CausalSteps, CfgScale = 1f,
        FlowShift = (float)(distilled ? DuoSchedule.DistillShift : DuoSchedule.CausalShift),
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
    public bool Matches(string familyId) =>
        (distilled ? _distilledIds : _causalIds).Contains(familyId, StringComparer.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public SamplingCapabilities.SamplingSupport SamplingSupportFor(string familyId, ResolvedModelVariant? variant) =>
        SamplingCapabilities.ForVideo(FamilyId);

    /// <summary>Why a catalogued Duo model must be refused before anything is downloaded: the release folder the run would read (the spec's
    /// path, else where the catalog puts its files) holds no <c>realization/backbone</c>, and no download provides it. Null when the folder
    /// holds one, or the spec is not a catalogued Duo model.</summary>
    public static string? BackboneRefusal(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        if (spec.Catalog is not { } entry || (!_causalIds.Contains(entry.Id, StringComparer.OrdinalIgnoreCase)
            && !_distilledIds.Contains(entry.Id, StringComparer.OrdinalIgnoreCase)))
        {
            return null;
        }
        string? path = spec.LocalPath ?? ModelDownloader.PrimaryLocalPath(entry);
        if (path is null)
        {
            return null;
        }
        string backbone = DuoAssets.BackboneFolder(path);
        return Directory.Exists(backbone)
            ? null
            : $"{entry.Id} reads the MiniMax-H3 Ref2VA backbone (about 66 GB) from '{backbone}', which is not there and no download "
                + "provides it. Put the release's shards there, or name a release folder that holds them with --model-path.";
    }

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
        DuoAssets assets = DuoAssets.Resolve(context.CheckpointPath, context.Components, distilled);
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
            CheckPlannedRequest(context, config);
            using (DuoLoraSource lora = OpenLora(assets.LoraDirectory))
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
            return new DuoRecipePipeline(context.Backend, transformer, config, codecs, residentBytes, loaders, distilled);
        }
        catch
        {
            transformer?.Dispose();
            new CompositeDisposable([.. loaders]).Dispose();
            throw;
        }
    }

    /// <summary>Runs the request and history checks of a service-bound plan before the merge, so a request that cannot run does not cost
    /// one. <see cref="DuoRecipePipeline.Generate"/> repeats them, since the pipeline is cached.</summary>
    private void CheckPlannedRequest(RecipeContext context, MiniMaxH3Config config)
    {
        if (context.VideoPlan?.RequestBinding is null)
        {
            return;
        }
        VideoPlan plan = VideoRequestExecutionBinding.RequirePlannedState(context.VideoPlan);
        VideoRequest planned = plan.EffectiveSettings.Apply(plan.RequestBinding!.Request);
        DuoRunInput input = DuoRunInput.Read(planned, plan.EffectiveSettings, config, distilled);
        input.RequireHistoryFits(config, context.Backend.GetDeviceMemoryUsage());
    }

    private DuoLoraSource OpenLora(string directory)
    {
        string prefix = distilled ? DistilledFamilyPrefix : CausalFamilyPrefix;
        try
        {
            return DuoLoraSource.FromDcp(directory, prefix);
        }
        catch (InvalidDataException ex)
        {
            throw new InvalidDataException(
                $"'{directory}' did not read as a {(distilled ? "distilled" : "causal")} Duo LoRA ('{prefix}' keys). " + ex.Message, ex);
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
