using HartsyInference.Core.Backends.Cpu;
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

    /// <summary>File name of the KV refiner, looked up beside the generator.</summary>
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
    /// <remarks>The generator's blocks are placed per clip by <see cref="HartsyInference.Core.Memory.Vram.Streaming.BlockStreamingScope"/>
    /// in the rollout, with the frame cache reserved first. On a card that holds the whole generator the placement is resident-only.</remarks>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming;

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
            // F32 on CPU, where the kernels do not cast a half-precision bias; F16 on a GPU backend.
            bool castForCpu = context.Backend is CpuBackend;
            DType pickleDType = castForCpu ? DType.F32 : DType.F16;
            Dictionary<string, Tensor> transformerWeights = LoadTransformer(context.CheckpointPath, pickleDType, castForCpu, loaders);
            WanVideoConfig wan = WanConfigDetector.Detect(transformerWeights);
            FastFramesConfig config = ConfigFor(wan.InnerDim);

            WanVideoTransformer transformer = new WanVideoTransformer(wan);
            loaders.Add(transformer);
            transformer.LoadWeights(transformerWeights);

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

    /// <summary>The transformer weights the recipe loads from a generator file: read, converted to the Wan key names, and checked.
    /// Used by the recipe and by the tests, so the tests exercise the same load path as a generation.</summary>
    internal static Dictionary<string, Tensor> LoadTransformer(string path, DType pickleDType, bool castForCpu, List<IDisposable> loaders)
    {
        Dictionary<string, Tensor> generator = ReadGenerator(path, pickleDType, castForCpu, loaders);
        WanVideoCheckpointConverter.ConvertedWeights converted = WanVideoCheckpointConverter.Convert(generator);
        if (converted.Transformer.Count == 0)
        {
            throw new InvalidOperationException($"FastFrames generator '{path}' has no recognized transformer weights after conversion.");
        }
        RefuseUnsupported(converted.Transformer);
        return converted.Transformer;
    }

    /// <summary>Refuses a tensor the FastFrames path cannot read as plain floats, by name. A quantized weight carries scales (ComfyUI
    /// fp8, int8 or NVFP4) the path cannot apply. Any dtype other than F32, F16 or BF16 is refused too, which covers unscaled fp8.
    /// The released Apple checkpoints are float and unquantized, so this refuses only a checkpoint someone else made.</summary>
    internal static void RefuseUnsupported(IReadOnlyDictionary<string, Tensor> weights)
    {
        foreach (KeyValuePair<string, Tensor> entry in weights)
        {
            Tensor tensor = entry.Value;
            if (tensor.QuantInfo is { } quant)
            {
                throw new NotSupportedException(
                    $"FastFrames does not load quantized weights: '{entry.Key}' is {quant.Format}. Use the released bf16 or fp16 generator.");
            }
            if (tensor.DType != DType.F32 && tensor.DType != DType.F16 && tensor.DType != DType.BF16)
            {
                throw new NotSupportedException(
                    $"FastFrames loads float weights only: '{entry.Key}' is {tensor.DType.Name}. Use the released bf16 or fp16 generator.");
            }
        }
    }

    /// <summary>Reads the generator as the Wan converter's input: original Wan key names, no wrapper prefix. A PyTorch pickle
    /// (the Causal Forcing++ S file) is a training checkpoint whose <c>generator_ema</c> dict holds the generator; only that dict
    /// is read, and the FSDP wrapper prefix is stripped. The <c>generator</c> and <c>critic</c> dicts are never touched. A
    /// safetensors file (L) is passed through as the file stores it, unless <paramref name="castForCpu"/> is set: the CPU kernels
    /// read F32, so each tensor is then cast once to <paramref name="dtype"/> and the copies are freed with the loaders. A CPU
    /// run of L would cast about 28.6 GB of BF16 to about 57 GB of F32 in host RAM, so CPU is not a supported path for L.</summary>
    internal static Dictionary<string, Tensor> ReadGenerator(string path, DType dtype, bool castForCpu, List<IDisposable> loaders)
    {
        if (!path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) && !path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase))
        {
            CheckpointSource source = CheckpointSource.Open(path);
            loaders.Add(source);
            Dictionary<string, Tensor> passthrough = new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
            if (castForCpu)
            {
                CastCopies(passthrough, dtype, loaders);
            }
            return passthrough;
        }

        // The owner is registered before anything else touches the tensors, so every later failure frees them.
        (Dictionary<string, Tensor> weights, IDisposable owner) = PytorchEnvelopeReader.Read(
            path, "generator_ema", "model._fsdp_wrapped_module.", dtype);
        loaders.Add(owner);
        return weights;
    }

    /// <summary>Replaces each tensor not already in <paramref name="dtype"/> with an owned copy in that dtype, and registers the
    /// copies so that they are freed with the loaders.</summary>
    private static void CastCopies(Dictionary<string, Tensor> weights, DType dtype, List<IDisposable> loaders)
    {
        List<Tensor> copies = new List<Tensor>();
        try
        {
            foreach (string name in weights.Keys.ToList())
            {
                Tensor tensor = weights[name];
                if (tensor.DType == dtype)
                {
                    continue;
                }
                Tensor copy = tensor.CastTo(dtype);
                copies.Add(copy);
                weights[name] = copy;
            }
        }
        catch
        {
            foreach (Tensor copy in copies)
            {
                copy.Dispose();
            }
            throw;
        }
        loaders.Add(new OwnedTensors(copies));
    }

    /// <summary>Frees the tensors it was given, once.</summary>
    private sealed class OwnedTensors(List<Tensor> tensors) : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            foreach (Tensor tensor in tensors)
            {
                tensor.Dispose();
            }
        }
    }
}
