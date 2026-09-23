using System.Collections.Concurrent;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Registry;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>Everything about a model's memory that does not depend on the request or the device, read once from its
/// checkpoint header(s) and cached for the life of the process.</summary>
/// <remarks><para>Keyed by the spec's identity plus the checkpoint's size and modification time, so replacing a file on
/// disk is picked up on the next call while every other call is a dictionary hit. Values are <see cref="Lazy{T}"/> so
/// concurrent first requests for one model read its header once, not once each.</para>
/// <para>Unbounded by design: one entry per distinct checkpoint a process has been asked about, each a few dtype
/// buckets — the model library, not the request volume, sets the size.</para></remarks>
internal sealed class CheckpointMemoryProfile
{
    /// <summary>Generic denoiser working-memory floor for families that have not described their own.</summary>
    private const long GenericActivationBaseBytes = 1L << 30;

    /// <summary>Generic denoiser working memory per output pixel (x frames x batch). Coarse on purpose: it only has to
    /// rank a large video request above a still image; families that need precision describe themselves.</summary>
    private const long GenericActivationBytesPerPixel = 64;

    /// <summary>Generic decode working-memory floor.</summary>
    private const long GenericDecodeBaseBytes = 512L << 20;

    /// <summary>Generic decode working memory per output pixel: F32 feature maps at the VAE's widest stage.</summary>
    private const long GenericDecodeBytesPerPixel = 32;

    /// <summary>Text-encode working memory beside the encoder weights: one short sequence, dwarfed by the weights.</summary>
    private const long TextEncodeActivationBytes = 512L << 20;

    /// <summary>Blocks resident while streaming: the active block plus the default prefetch window.</summary>
    private const int StreamWindowBlocks = Core.MemoryManagement.BlockStreamingOptions.DefaultPrefetchAhead + 1;

    private static readonly ConcurrentDictionary<ProfileKey, Lazy<CheckpointMemoryProfile>> Cache = new();

    private CheckpointMemoryProfile(string familyId, int defaultFrames, MemoryCapabilities capabilities,
        CheckpointWeightInventory checkpoint, CheckpointWeightInventory? textEncoder, CheckpointWeightInventory? vae,
        RecipeMemoryModel? model)
    {
        FamilyId = familyId;
        DefaultFrames = defaultFrames;
        Capabilities = capabilities;
        Checkpoint = checkpoint;
        TextEncoder = textEncoder;
        Vae = vae;
        Model = model;
    }

    /// <summary>The family the spec resolved to.</summary>
    public string FamilyId { get; }

    /// <summary>The frame count a request that leaves it unset generates: the video recipe's default, 1 for images.</summary>
    public int DefaultFrames { get; }

    /// <summary>The memory behaviours the family's recipe wires.</summary>
    public MemoryCapabilities Capabilities { get; }

    private CheckpointWeightInventory Checkpoint { get; }

    private CheckpointWeightInventory? TextEncoder { get; }

    private CheckpointWeightInventory? Vae { get; }

    private RecipeMemoryModel? Model { get; }

    /// <summary>The cached profile for <paramref name="spec"/>, reading its header(s) on first use only.</summary>
    /// <exception cref="FileNotFoundException">The spec names no readable checkpoint.</exception>
    public static CheckpointMemoryProfile For(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        if (string.IsNullOrEmpty(spec.LocalPath))
        {
            throw new FileNotFoundException($"No local checkpoint for '{spec.Requested}' to estimate memory from.");
        }
        ProfileKey key = ProfileKey.For(spec);
        Lazy<CheckpointMemoryProfile> entry = Cache.GetOrAdd(key,
            _ => new Lazy<CheckpointMemoryProfile>(() => Build(spec), LazyThreadSafetyMode.ExecutionAndPublication));
        try
        {
            return entry.Value;
        }
        catch
        {
            // A failed read (file mid-copy, permissions) must not be cached forever: drop it so the next call retries.
            Cache.TryRemove(new KeyValuePair<ProfileKey, Lazy<CheckpointMemoryProfile>>(key, entry));
            throw;
        }
    }

    /// <summary>The estimate for one geometry, with weights sized for a device described by
    /// <paramref name="supportsResidentQuant"/>.</summary>
    public MemoryEstimate Estimate(MemoryEstimateRequest request, Func<DType, bool> supportsResidentQuant)
    {
        ArgumentNullException.ThrowIfNull(supportsResidentQuant);
        request = request with { Frames = request.Frames ?? DefaultFrames };
        int batch = Math.Max(1, request.Batch);
        bool streamable = Capabilities.HasFlag(MemoryCapabilities.BlockStreaming);
        long denoiserActivation = Model is not null
            ? Model.DenoiserActivationBytes(request) * batch
            : GenericActivationBaseBytes + request.Workload * GenericActivationBytesPerPixel;
        long decodeActivation = Model?.VaeActivationBytes is { } decode
            ? decode(request) * batch
            : GenericDecodeBaseBytes + request.Workload * GenericDecodeBytesPerPixel;

        List<MemoryPhase> phases = [];
        long textEncoderBytes = ComponentBytes(MemoryComponent.TextEncoder, TextEncoder, supportsResidentQuant);
        if (textEncoderBytes > 0)
        {
            phases.Add(new MemoryPhase(MemoryComponent.TextEncoder, textEncoderBytes, TextEncodeActivationBytes, 0, false));
        }
        long denoiserBytes = Checkpoint.ResidentBytes(MemoryComponent.Denoiser, supportsResidentQuant);
        phases.Add(new MemoryPhase(MemoryComponent.Denoiser, denoiserBytes, denoiserActivation,
            Checkpoint.DenoiserStreamFloorBytes(supportsResidentQuant, StreamWindowBlocks), streamable));
        long vaeBytes = ComponentBytes(MemoryComponent.Vae, Vae, supportsResidentQuant);
        phases.Add(new MemoryPhase(MemoryComponent.Vae, vaeBytes, decodeActivation, 0, false));

        return new MemoryEstimate
        {
            FamilyId = FamilyId,
            Phases = phases,
            Accuracy = Model is null ? MemoryEstimateAccuracy.HeaderOnly : MemoryEstimateAccuracy.Recipe,
        };
    }

    /// <summary>A component's bytes: bundled in the checkpoint when it carries one, else the recipe's side model.</summary>
    private long ComponentBytes(MemoryComponent component, CheckpointWeightInventory? side, Func<DType, bool> quant) =>
        Checkpoint.Has(component) ? Checkpoint.ResidentBytes(component, quant) : side?.ResidentBytes(component, quant) ?? 0;

    private static CheckpointMemoryProfile Build(ModelSpec spec)
    {
        bool video = spec.Modality == Modality.Video;
        string familyId = video ? InferenceEngine.ResolveVideoFamilyId(spec) : InferenceEngine.ResolveFamilyId(spec);
        ModelLayout layout = ModelLayoutResolver.Resolve(spec.LocalPath!);
        CheckpointHeader representative = CheckpointHeader.Read(layout.RepresentativeFile);
        CheckpointWeightInventory checkpoint = CheckpointWeightInventory.FromFiles(layout.SafeTensorsFiles.Select(file =>
            (DescriptorsOf(file, layout, representative), ComponentOf(file, layout))));

        MemoryCapabilities capabilities;
        RecipeMemoryModel? model;
        int defaultFrames = 1;
        if (video)
        {
            IVideoRecipe? recipe = VideoRecipeRegistry.Resolve(familyId);
            capabilities = recipe?.MemorySupports ?? MemoryCapabilities.None;
            model = recipe?.DescribeMemory(representative);
            defaultFrames = InferenceEngine.VideoDefaultsFor(spec).Frames ?? VideoDefaults.Standard.Frames ?? 1;
        }
        else
        {
            IArchitectureRecipe? recipe = RecipeRegistry.Resolve(familyId);
            capabilities = recipe?.MemorySupports ?? MemoryCapabilities.None;
            model = recipe?.DescribeMemory(representative);
        }
        return new CheckpointMemoryProfile(familyId, defaultFrames, capabilities, checkpoint,
            SideInventory(model?.TextEncoder, MemoryComponent.TextEncoder),
            SideInventory(model?.Vae, MemoryComponent.Vae), model);
    }

    /// <summary>One file's descriptors, reusing the representative header already read.</summary>
    private static IEnumerable<ModelAssets.SafeTensors.SafeTensorDescriptor> DescriptorsOf(string file, ModelLayout layout,
        CheckpointHeader representative) =>
        (file == layout.RepresentativeFile ? representative : CheckpointHeader.Read(file)).Descriptors.Values;

    /// <summary>The component a file's unprefixed keys belong to: its diffusers component folder, else the denoiser.</summary>
    private static MemoryComponent ComponentOf(string file, ModelLayout layout)
    {
        if (layout.Kind != ModelLayoutKind.Diffusers)
        {
            return MemoryComponent.Denoiser;
        }
        string folder = Path.GetFileName(Path.GetDirectoryName(file)) ?? "";
        return folder.Equals("vae", StringComparison.OrdinalIgnoreCase) ? MemoryComponent.Vae
            : folder.StartsWith("text_encoder", StringComparison.OrdinalIgnoreCase) ? MemoryComponent.TextEncoder
            : MemoryComponent.Denoiser;
    }

    /// <summary>The inventory of a side model already on disk. One that is not downloaded yet contributes nothing: the
    /// estimate stays honest about what it could read rather than guessing a size.</summary>
    private static CheckpointWeightInventory? SideInventory(ModelAsset? asset, MemoryComponent component)
    {
        if (asset is null)
        {
            return null;
        }
        string path = ModelDownloader.TargetPath(asset);
        return File.Exists(path)
            ? CheckpointWeightInventory.FromDescriptors(CheckpointHeader.Read(path).Descriptors.Values, component)
            : null;
    }

    /// <summary>A spec's identity plus its checkpoint's stamp, so an edited or replaced file re-profiles.</summary>
    private readonly record struct ProfileKey(Modality Modality, string Requested, string? CatalogId, string FullPath,
        long Length, DateTime LastWriteUtc)
    {
        public static ProfileKey For(ModelSpec spec)
        {
            string path = Path.GetFullPath(spec.LocalPath!);
            FileInfo file = new(path);
            (long length, DateTime written) = file.Exists
                ? (file.Length, file.LastWriteTimeUtc)
                : (0L, Directory.GetLastWriteTimeUtc(path));
            return new ProfileKey(spec.Modality, spec.Requested ?? "", spec.Catalog?.Id, path, length, written);
        }
    }
}
