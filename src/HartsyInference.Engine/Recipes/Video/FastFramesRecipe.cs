using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.FastFrames;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Apple FastFrames (S 1.3B and L 14B): a frame-wise causal Wan2.1 text-to-video generator with a per-layer frame
/// KV cache, a KV refinement network and a causal pixel decoder, conditioned by Wan's umT5-XXL. Text-to-video only: no init
/// image, no end frame, no LoRA, no negative prompt and no guidance. The released sampler is fixed (four steps for the first
/// latent frame, then two plus a refine pass per frame), so Steps and CFG are not tunables here.</summary>
/// <remarks>Weights come from three files: the generator (the Causal Forcing++ <c>framewise-2step.pt</c> for S, the Apple
/// <c>dit.safetensors</c> for L), <c>kv_refiner.safetensors</c> and <c>pixel_decoder.safetensors</c>. The last two sit beside the
/// generator in the same folder, which is how the catalog places them. S and L are told apart by the generator's width.</remarks>
public sealed class FastFramesRecipe : IVideoRecipe
{
    /// <summary>The only frame size both released models generate: 60x104 latent cells, 16 px per cell.</summary>
    internal const int OutputWidth = 832;

    /// <summary>The only frame size both released models generate.</summary>
    internal const int OutputHeight = 480;

    /// <summary>Longest clip the released models run: 21 latent frames, each covering four pixel frames.</summary>
    internal const int MaxFrames = 81;

    /// <summary>Playback rate upstream uses for every clip.</summary>
    internal const int OutputFps = 16;

    /// <summary>umT5-XXL's hidden width, the prompt embedding width the generator's text cross-attention takes.</summary>
    internal const int TextDim = 4096;

    /// <summary>Steps the first latent frame runs (upstream's first-frame timesteps); reported as the step count.</summary>
    internal const int FirstFrameSteps = 4;

    /// <summary>The guidance scale at which classifier-free guidance changes nothing: FastFrames samples without guidance.</summary>
    internal const float NoGuidance = 1f;

    /// <summary>File names of the two companions, looked up beside the generator.</summary>
    internal const string RefinerFile = "kv_refiner.safetensors";

    /// <summary>File name of the pixel decoder, looked up beside the generator.</summary>
    internal const string DecoderFile = "pixel_decoder.safetensors";

    private const string FamilyId = "fastframes";

    /// <inheritdoc/>
    public string Name => FamilyId;

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, FamilyId, StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "fastframes-s", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "fastframes-l", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    /// <remarks>Declared from the recipe's own encoder construction: one umT5-XXL pass over the prompt, zero-padded to 512 rows
    /// (<c>embed_text</c> upstream). Upstream has no negative prompt and no weighting, so neither is declared.</remarks>
    public PromptProfile? Prompt { get; } = new()
    {
        Kind = "video", Encoders = ["t5"], Format = "text", Negative = "none", Supports = [],
    };

    /// <inheritdoc/>
    /// <remarks>Upstream's <c>pipeline.py</c> call: 832x480, 81 frames at 16 fps, and no guidance. The registry requires a positive scale, and 1 is the value at which guidance has no effect. Steps is
    /// the first-frame step count, recorded for the plan; the later frames' schedule is fixed by the released generator.</remarks>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Revision = 1, Steps = FirstFrameSteps, CfgScale = NoGuidance, Width = OutputWidth, Height = OutputHeight, Frames = MaxFrames, Fps = OutputFps,
    };

    /// <inheritdoc/>
    /// <remarks>Frame counts on the 4n+1 grid up to 81. The grid is the causal VAE's grid upstream: the first pixel frame is
    /// dropped by <c>align_frames</c>, so every latent frame carries four pixel frames.</remarks>
    public VideoConstraints Constraints { get; } = new VideoConstraints
    {
        Frames = FrameGrid.StepPlusOne(4, FrameNormalization.SnapNearest, MaxFrames),
    };

    /// <inheritdoc/>
    public SamplingCapabilities.SamplingSupport SamplingSupportFor(string familyId, ResolvedModelVariant? variant) =>
        SamplingCapabilities.ForVideo(FamilyId);

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        string directory = Path.GetDirectoryName(Path.GetFullPath(context.CheckpointPath))
            ?? throw new ArgumentException($"FastFrames generator path '{context.CheckpointPath}' has no folder.", nameof(context));
        string refinerPath = Path.Combine(directory, RefinerFile);
        string decoderPath = Path.Combine(directory, DecoderFile);
        // Checked before the text encoder is resolved, so a folder missing a companion fails in a second, not after a download.
        RequireCompanion(refinerPath);
        RequireCompanion(decoderPath);

        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, context.Cancel).GetAwaiter().GetResult();
        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            Dictionary<string, Tensor> generator = ReadGenerator(context.CheckpointPath, loaders);
            WanVideoCheckpointConverter.ConvertedWeights converted = WanVideoCheckpointConverter.Convert(generator);
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"FastFrames generator '{context.CheckpointPath}' has no recognized transformer weights after conversion.");
            }
            WanVideoConfig wan = WanConfigDetector.Detect(converted.Transformer);
            FastFramesConfig config = ConfigFor(wan.InnerDim);

            WanVideoTransformer transformer = new WanVideoTransformer(wan);
            loaders.Add(transformer);
            transformer.LoadWeights(converted.Transformer);

            CheckpointSource refinerSource = CheckpointSource.Open(refinerPath);
            loaders.Add(refinerSource);
            FastFramesKvRefiner refiner = new FastFramesKvRefiner(FastFramesKvRefiner.DetectConfig(refinerSource.Weights, wan.NumHeads));
            loaders.Add(refiner);
            refiner.LoadWeights(refinerSource.Weights);

            CheckpointSource decoderSource = CheckpointSource.Open(decoderPath);
            loaders.Add(decoderSource);
            FastFramesPixelDecoder decoder = new FastFramesPixelDecoder(FastFramesPixelDecoder.DetectConfig(decoderSource.Weights));
            loaders.Add(decoder);
            decoder.LoadWeights(decoderSource.Weights);

            (T5TextEncoder umt5, T5Tokenizer tokenizer) = VideoRecipeUtils.LoadUmt5(umt5Path, loaders);

            FastFramesPipeline pipeline = new FastFramesPipeline(context.Backend, transformer, refiner, decoder, config);
            Logs.Info($"[FastFramesRecipe] {config.Name} ready ({wan.NumLayers} layers, inner {wan.InnerDim}).");
            return new FastFramesRecipePipeline(context.Backend, context.TextEncoderBackendOrDefault, pipeline, config, umt5, tokenizer, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[FastFramesRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>The released configuration for a generator of this width: S is 1536 wide, L is 5120.</summary>
    /// <exception cref="NotSupportedException">The generator is neither released width.</exception>
    internal static FastFramesConfig ConfigFor(int innerDim) => innerDim switch
    {
        1536 => FastFramesConfig.S,
        5120 => FastFramesConfig.L,
        _ => throw new NotSupportedException(
            $"A FastFrames generator is 1536 or 5120 wide; this one is {innerDim}. It is not a released FastFrames checkpoint."),
    };

    private static void RequireCompanion(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"FastFrames needs '{Path.GetFileName(path)}' beside the generator. Both downloads come from the same catalog entry.", path);
        }
    }

    /// <summary>Reads the generator as the Wan converter's input: original Wan key names, no wrapper prefix. A PyTorch pickle
    /// (the Causal Forcing++ S file) keeps its tensors under <c>generator_ema</c> with an FSDP wrapper prefix; those are read one
    /// at a time, cast to F16 and unwrapped. A safetensors file (L) is passed through as the file stores it.</summary>
    private static Dictionary<string, Tensor> ReadGenerator(string path, List<IDisposable> loaders)
    {
        if (!path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) && !path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase))
        {
            CheckpointSource source = CheckpointSource.Open(path);
            loaders.Add(source);
            return new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
        }

        using PytorchPickleLoader pickle = new PytorchPickleLoader();
        pickle.OpenLazy(path);
        const string wrapper = "model._fsdp_wrapped_module.";
        List<PytorchTensorDescriptor> ordered = pickle.Descriptors.Values
            .OrderBy(d => d.StorageKey, StringComparer.Ordinal)
            .ThenBy(d => d.StorageByteOffset)
            .ToList();
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(ordered.Count, StringComparer.Ordinal);
        try
        {
            foreach (PytorchTensorDescriptor descriptor in ordered)
            {
                if (!descriptor.Name.StartsWith(wrapper, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException(
                        $"FastFrames generator '{path}' has a tensor '{descriptor.Name}' outside the expected '{wrapper}' wrapper.");
                }
                weights[descriptor.Name[wrapper.Length..]] = pickle.ReadTensor(descriptor.Name, DType.F16);
            }
        }
        catch
        {
            foreach (Tensor tensor in weights.Values)
            {
                tensor.Dispose();
            }
            throw;
        }
        return weights;
    }
}
