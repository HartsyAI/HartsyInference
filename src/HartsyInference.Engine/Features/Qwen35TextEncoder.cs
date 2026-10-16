using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
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
public sealed class Qwen35TextEncoder : IFinalHiddenStateEncoder, IDisposable
{
    /// <summary>Tokens per prefill call; the conv and recurrent state carry between calls, so this only bounds the
    /// per-call activations.</summary>
    public const int PrefillChunk = 2048;

    private readonly Qwen35Model _model;
    private readonly IDisposable _owner;
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
    public IEnumerable<Tensor> EnumerateWeights() => _model.EnumerateWeights();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
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
