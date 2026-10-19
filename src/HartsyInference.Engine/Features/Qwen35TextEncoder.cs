using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Features;

/// <summary>A dense Qwen3.5 trunk (0.8B–27B) as a diffusion text encoder, returning the final normalized hidden state of
/// every prompt token. Loads a Hugging Face component folder (<c>config.json</c> plus shards, as OpenMOSS ships
/// <c>text_encoder/</c>), a single safetensors file (ComfyUI's repackaged encoders, config derived from the weight
/// shapes; fp8-scaled files fold their scales on open), or a <c>qwen35</c> GGUF at any llama.cpp quant, which also
/// carries its tokenizer. Runs headless: the output head is never loaded and the token embedding stays on the host, so
/// only the trunk's weights reach the device. The gated-DeltaNet layers prefill through the backend's device kernels.</summary>
public sealed class Qwen35TextEncoder : IMultimodalFinalHiddenStateEncoder, IDisposable
{
    /// <summary>Tokens per prefill call; the conv and recurrent state carry between calls, so this only bounds the
    /// per-call activations.</summary>
    public const int PrefillChunk = 2048;

    private readonly Qwen35Model _model;
    private readonly IDisposable _owner;
    private Qwen35VisionTower? _vision;
    private int _disposed;

    private Qwen35TextEncoder(Qwen35Model model, IDisposable owner, ILlmTokenizer? tokenizer)
    {
        _model = model;
        _owner = owner;
        EmbeddedTokenizer = tokenizer;
    }

    /// <inheritdoc/>
    public int HiddenSize => _model.DModel;

    /// <summary>The tokenizer the checkpoint carries (a GGUF's), or null when it comes separately.</summary>
    public ILlmTokenizer? EmbeddedTokenizer { get; }

    /// <summary>The weights the checkpoint was opened from (null for a GGUF), for finding a vision tower beside the trunk.</summary>
    public IReadOnlyDictionary<string, Tensor>? SourceWeights => (_owner as CheckpointSource)?.Weights;

    /// <inheritdoc/>
    public bool HasVision => _vision is not null;

    /// <summary>Id of the image placeholder token in prompts handed to <see cref="Encode(IBackend, IReadOnlyList{int}, IReadOnlyList{Tensor})"/>;
    /// set from the tokenizer before image input.</summary>
    public int ImageTokenId { get; set; } = -1;

    /// <summary>Takes ownership of <paramref name="vision"/> as this encoder's tower.</summary>
    public void AttachVision(Qwen35VisionTower vision)
    {
        ArgumentNullException.ThrowIfNull(vision);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (vision.Config.OutHiddenSize != _model.DModel)
            throw new InvalidDataException($"The vision tower emits {vision.Config.OutHiddenSize}-wide tokens; the trunk is {_model.DModel} wide.");
        _vision?.Dispose();
        _vision = vision;
    }

    /// <summary>Loads the encoder at <paramref name="path"/>, sized for prompts of up to <paramref name="maxTokens"/>.</summary>
    /// <param name="family">Model family, for log lines and errors.</param>
    public static Qwen35TextEncoder Load(string path, int maxTokens, string family)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (maxTokens < 1)
            throw new ArgumentOutOfRangeException(nameof(maxTokens), maxTokens, "The encoder needs room for at least one token.");
        if (path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            SsmLanguageModel gguf = SsmLanguageModel.Load(path, "qwen35");
            if (gguf.Model is not Qwen35Model ggufModel)
            {
                gguf.Dispose();
                throw new InvalidDataException($"{family}: '{Path.GetFileName(path)}' is not a Qwen3.5 GGUF.");
            }
            Logs.Info($"[{family}] Qwen3.5 text encoder (GGUF): {Path.GetFileName(path)}, hidden {ggufModel.DModel}.");
            return new Qwen35TextEncoder(ggufModel, gguf, gguf.Tokenizer);
        }

        bool folder = Directory.Exists(path);
        IReadOnlyList<string> files = folder ? HfCheckpointDirectory.WeightFiles(path) : [path];
        CheckpointSource source = CheckpointSource.OpenShards(files);
        try
        {
            IReadOnlyDictionary<string, Tensor> weights = source.Weights;
            string prefix = Qwen35HfConfig.LanguageModelPrefix(weights);
            Tensor embed = weights[$"{prefix}.embed_tokens.weight"];
            if (embed.DType != DType.F32 && embed.DType != DType.BF16 && embed.DType != DType.F16)
            {
                throw new NotSupportedException(
                    $"{family}: the Qwen3.5 token embedding in '{Path.GetFileName(path)}' is {embed.DType}; a safetensors encoder needs it in F32, BF16 or F16 (use a GGUF for a quantized table).");
            }
            string? configPath = folder ? Path.Combine(path, HfCheckpointDirectory.ConfigFileName) : null;
            Qwen35HfConfig config = configPath is not null && File.Exists(configPath)
                ? Qwen35HfConfig.FromJson(File.ReadAllText(configPath))
                : Qwen35HfConfig.FromWeights(weights, prefix);
            if (config.IsMoe)
                throw new NotSupportedException($"{family}: a Qwen3.5-MoE trunk is not a supported text encoder.");
            // Headless: no output head, so the embedding stays a host-side lookup table.
            Qwen35Model model = Qwen35Model.FromHuggingFace(weights, config with { TieWordEmbeddings = false },
                maxTokens + 8, prefix, lmHeadKey: null);
            Logs.Info($"[{family}] Qwen3.5 text encoder: {(folder ? path : Path.GetFileName(path))}, hidden {config.HiddenSize}, "
                + $"{config.NumLayers} layers ({config.NumLayers / Math.Max(1, CountFull(config))}:1 linear:full), {embed.DType} weights.");
            return new Qwen35TextEncoder(model, source, null);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public unsafe Tensor Encode(IBackend backend, IReadOnlyList<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tokenIds);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (tokenIds.Count == 0)
            throw new ArgumentException("A prompt needs at least one token.", nameof(tokenIds));
        int hidden = _model.DModel;
        Tensor output = new(new TensorShape(1, tokenIds.Count, hidden), DType.F32);
        try
        {
            _model.ResetState();
            float* dst = (float*)output.DataPointer;
            for (int start = 0; start < tokenIds.Count; start += PrefillChunk)
            {
                int count = Math.Min(PrefillChunk, tokenIds.Count - start);
                int[] chunk = new int[count];
                for (int i = 0; i < count; i++) chunk[i] = tokenIds[start + i];
                using Tensor embeds = new(new TensorShape(1, count, hidden), DType.F32);
                _model.EmbedLookup(embeds, chunk);
                using Tensor states = _model.ForwardEmbedsHidden(backend, embeds, count);
                long bytes = (long)count * hidden * sizeof(float);
                Buffer.MemoryCopy((void*)states.DataPointer, dst + (long)start * hidden, bytes, bytes);
            }
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
        finally
        {
            _model.ResetState();
        }
    }

    /// <inheritdoc/>
    public (int Width, int Height) VisionInputSize(int width, int height)
    {
        Qwen35VisionTower vision = RequireVision();
        (int h, int w) = vision.Processor.SmartResize(height, width);
        return (w, h);
    }

    /// <inheritdoc/>
    public int CountImageTokens(int width, int height)
    {
        Qwen35VisionTower vision = RequireVision();
        (int h, int w) = vision.Processor.SmartResize(height, width);
        int unit = vision.Config.PatchSize * vision.Config.SpatialMergeSize;
        return h / unit * (w / unit);
    }

    /// <inheritdoc/>
    public unsafe Tensor Encode(IBackend backend, IReadOnlyList<int> tokenIds, IReadOnlyList<Tensor> images)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(images);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (images.Count == 0)
            return Encode(backend, tokenIds);
        Qwen35VisionTower vision = RequireVision();
        if (ImageTokenId < 0)
            throw new InvalidOperationException("The image placeholder token id must be set before image input.");
        int[] ids = [.. tokenIds];
        MRopeImageGrid[] grids = new MRopeImageGrid[images.Count];
        List<Tensor> features = new(images.Count);
        int hidden = _model.DModel;
        Tensor? embeds = null;
        try
        {
            backend.PreloadWeights(vision.Encoder.EnumerateWeights());
            int merge = vision.Config.SpatialMergeSize;
            for (int i = 0; i < images.Count; i++)
            {
                Tensor image = images[i];
                if (image.Shape.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3 || image.DType != DType.F32)
                    throw new ArgumentException($"Image {i} must be F32 [1, 3, H, W], got {image.Shape} {image.DType}.", nameof(images));
                Tensor rgb = image.Reshape(new TensorShape(3, image.Shape[2], image.Shape[3]));
                (Tensor pixels, int gridT, int gridH, int gridW) = vision.Processor.Preprocess(rgb);
                Qwen3VlVisionEncoder.VisionOutput output;
                using (pixels)
                {
                    output = vision.Encoder.Forward(backend, pixels, gridT, gridH, gridW);
                }
                foreach (Tensor tap in output.DeepstackFeatures) tap?.Dispose();
                features.Add(output.MergedTokens);
                grids[i] = new MRopeImageGrid(gridT, gridH / merge, gridW / merge);
            }
            backend.Sync();
            foreach (Tensor f in features) _ = f.DataPointer;
            backend.FreeWeights(vision.Encoder.EnumerateWeights());

            embeds = new Tensor(new TensorShape(1, ids.Length, hidden), DType.F32);
            _model.EmbedLookup(embeds, ids);
            VisionTokenSplice.Apply(embeds, ids, ImageTokenId, features);
            int[] positions = MRopePositions.Build(ids, ImageTokenId, grids, out _);
            return EncodeEmbeds(backend, embeds, positions, ids.Length);
        }
        finally
        {
            foreach (Tensor f in features) f.Dispose();
            embeds?.Dispose();
        }
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateVisionWeights() => _vision?.Encoder.EnumerateWeights() ?? [];

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateWeights() => _model.EnumerateWeights();

    /// <summary>Runs spliced embeddings through the trunk in prefill chunks, rotating the full-attention layers by the
    /// multimodal positions of each chunk's rows.</summary>
    private unsafe Tensor EncodeEmbeds(IBackend backend, Tensor embeds, int[] positions, int length)
    {
        int hidden = _model.DModel;
        int rotaryDim = _model.RotaryDim;
        byte[] pairAxes = MRopePairAxes.Build([.. _model.MropeSection], rotaryDim / 2,
            _model.MropeInterleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);
        int maxPosition = 0;
        foreach (int p in positions) maxPosition = Math.Max(maxPosition, p);
        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(rotaryDim, _model.RopeTheta, RopeScaling.None, maxPosition + 1);
        Tensor output = new(new TensorShape(1, length, hidden), DType.F32);
        try
        {
            _model.ResetState();
            float* src = (float*)embeds.DataPointer;
            float* dst = (float*)output.DataPointer;
            for (int start = 0; start < length; start += PrefillChunk)
            {
                int count = Math.Min(PrefillChunk, length - start);
                long bytes = (long)count * hidden * sizeof(float);
                using Tensor chunk = new(new TensorShape(1, count, hidden), DType.F32);
                Buffer.MemoryCopy(src + (long)start * hidden, (void*)chunk.DataPointer, bytes, bytes);
                (Tensor cos, Tensor sin) = MRopeTable.BuildTensors(positions, length, start, count, invFreq, scaling, pairAxes,
                    _model.AttnHeadDim, rotaryDim);
                using (cos)
                using (sin)
                using (Tensor states = _model.ForwardEmbedsHidden(backend, chunk, count, cos, sin))
                {
                    Buffer.MemoryCopy((void*)states.DataPointer, dst + (long)start * hidden, bytes, bytes);
                }
            }
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
        finally
        {
            _model.ResetState();
        }
    }

    private Qwen35VisionTower RequireVision() =>
        _vision ?? throw new InvalidOperationException("This Qwen3.5 encoder has no vision tower; image input needs one.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _vision?.Dispose();
        _model.Dispose();
        _owner.Dispose();
    }

    private static int CountFull(Qwen35HfConfig config)
    {
        int full = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (!config.IsLinearLayer(i)) full++;
        }
        return full;
    }
}
