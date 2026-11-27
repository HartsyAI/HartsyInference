using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Everything the Text path needs to run a Hy4 Flash directory: the generation model, the tokenizer from the checkpoint's <c>tokenizer.json</c>, and the
/// chat template from its <c>chat_template.jinja</c> (a file without one gets a completion template).</summary>
/// <remarks>The host reference loads every weight as F32 in host memory, so only checkpoints up to <see cref="MaxHostWeightBytes"/> load; the released 220B
/// checkpoint is refused by name. Owns the generation model and the checkpoint it reads.</remarks>
public sealed class Hy4FlashTextModel : IHostReferenceTextModel
{
    /// <summary>Largest checkpoint the host reference loads (its safetensors bytes). The tiny test checkpoints are well under it.</summary>
    public const long MaxHostWeightBytes = 1L << 30;

    private readonly Hy4FlashCheckpoint _checkpoint;

    private Hy4FlashTextModel(Hy4FlashGenerationModel generation, Hy4FlashModel model, Hy4FlashVisionTower vision, ILlmTokenizer tokenizer,
        IChatTemplate template, Hy4FlashCheckpoint checkpoint)
    {
        Generation = generation;
        Model = model;
        Vision = vision;
        Tokenizer = tokenizer;
        Template = template;
        _checkpoint = checkpoint;
    }

    /// <summary>The engine-facing model.</summary>
    public Hy4FlashGenerationModel Generation { get; }

    /// <summary>The loaded model, owned by <see cref="Generation"/>; the image route runs it directly, so it is not disposed here.</summary>
    public Hy4FlashModel Model { get; }

    /// <summary>The vision tower the checkpoint's image weights built; the image route encodes each image through it.</summary>
    public Hy4FlashVisionTower Vision { get; }

    /// <inheritdoc />
    public ILlmTokenizer Tokenizer { get; }

    /// <inheritdoc />
    public IChatTemplate Template { get; }

    IGenerationModel IHostReferenceTextModel.Generation => Generation;

    /// <summary>Refuses a directory that could not load, before a caller unloads what it replaces: the config must parse, the weights must fit <paramref name="maxWeightBytes"/>, and the
    /// <c>tokenizer.json</c> must exist.</summary>
    /// <exception cref="HartsyInferenceException">The config is not runnable, the weights are larger than <paramref name="maxWeightBytes"/>, or <c>tokenizer.json</c> is missing.</exception>
    public static void RequireLoadable(string directory, long maxWeightBytes = MaxHostWeightBytes)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        Hy4FlashConfig.Load(Path.Combine(directory, "config.json"));
        long bytes = WeightBytes(directory);
        if (bytes > maxWeightBytes)
            throw new HartsyInferenceException($"'{directory}' holds {bytes / (1 << 20)} MiB of weights. The Hy4 Flash host reference loads every weight as F32 in host "
                + $"memory and runs only checkpoints up to {maxWeightBytes / (1 << 20)} MiB; the released checkpoints need the device path, which is not built.");
        if (!File.Exists(Path.Combine(directory, "tokenizer.json")))
            throw new HartsyInferenceException($"'{directory}' has no tokenizer.json, which the Hy4 Flash Text path needs to build its prompt.");
    }

    // A literal the tokenizer does not know is not checked (a tokenizer without image tokens is a text-only one); one it knows must have the config's id.
    private static void CheckImageId(ILlmTokenizer tokenizer, string literal, int configId)
    {
        if (tokenizer.SpecialId(literal) is not int id || id == configId) return;
        throw new HartsyInferenceException($"tokenizer.json puts {literal} at {id}, but config.json says {configId}.");
    }

    /// <summary>Loads <paramref name="directory"/>; the result owns everything it opened.</summary>
    /// <exception cref="HartsyInferenceException">The checkpoint is larger than <see cref="MaxHostWeightBytes"/>, the config is not runnable, the tokenizer is missing or
    /// unusable, or its special ids disagree with <c>config.json</c>.</exception>
    public static Hy4FlashTextModel Load(IBackend backend, string directory, int maxSequenceTokens) =>
        Load(backend, directory, maxSequenceTokens, MaxHostWeightBytes);

    /// <summary>As <see cref="Load(IBackend, string, int)"/> with an explicit limit on the weight bytes the host reference accepts.</summary>
    internal static Hy4FlashTextModel Load(IBackend backend, string directory, int maxSequenceTokens, long maxWeightBytes)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentException.ThrowIfNullOrEmpty(directory);
        RequireLoadable(directory, maxWeightBytes);
        Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(directory, "config.json"));
        string tokenizerPath = Path.Combine(directory, "tokenizer.json"), chatPath = Path.Combine(directory, "chat_template.jinja");
        (string? bosLiteral, string? eosLiteral) = HfChatAssets.SpecialTokenLiterals(directory);
        List<int> stopIds = HfStopIds.Read(directory, config.EosTokenId);
        ILlmTokenizer tokenizer;
        IChatTemplate template;
        try
        {
            tokenizer = HfChatAssets.LoadTokenizer(directory, stopIds, bosLiteral, eosLiteral);
            template = File.Exists(chatPath) ? new Hy4FlashChatTemplate(File.ReadAllText(chatPath)) : HfChatAssets.LoadCompletionTemplate(directory);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or JsonException or FormatException or IOException)
        {
            throw new HartsyInferenceException($"'{tokenizerPath}' or '{chatPath}' could not be loaded: {ex.Message}", ex);
        }
        // a mismatch would make the model read the wrong embeddings, so refuse rather than guess
        if (config.BosTokenId is { } bos && tokenizer.BosId != bos)
            throw new HartsyInferenceException($"tokenizer.json puts begin-of-sentence at {tokenizer.BosId?.ToString() ?? "no id"}, but config.json says {bos}.");
        if (config.EosTokenId is { } eos && tokenizer.EosId != eos)
            throw new HartsyInferenceException($"tokenizer.json puts end-of-sentence at {tokenizer.EosId?.ToString() ?? "no id"}, but config.json says {eos}.");
        // The model reads image runs by these ids, so a tokenizer that disagrees would place the features on the wrong tokens.
        CheckImageId(tokenizer, Hy4FlashChatTemplate.ImagePadLiteral, config.ImageTokenId);
        CheckImageId(tokenizer, Hy4FlashChatTemplate.ImageStartLiteral, config.ImageStartTokenId);
        CheckImageId(tokenizer, Hy4FlashChatTemplate.ImageEndLiteral, config.ImageEndTokenId);

        Hy4FlashCheckpoint checkpoint = Hy4FlashCheckpoint.Open(directory);
        Hy4FlashModel? model = null;
        Hy4FlashVisionTower? vision = null;
        try
        {
            model = new Hy4FlashModel(backend, config, checkpoint, maxSequenceTokens);
            vision = new Hy4FlashVisionTower(backend, config.Vision, checkpoint);
            Hy4FlashGenerationModel generation = new(model, backend, maxSequenceTokens);
            // the generation model owns the model from its constructor on, so it is not released here
            Hy4FlashModel owned = model;
            model = null;
            return new Hy4FlashTextModel(generation, owned, vision, tokenizer, template, checkpoint);
        }
        catch
        {
            // whatever was built and not handed over is released here
            model?.Dispose();
            vision?.Dispose();
            checkpoint.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Vision.Dispose();
        Generation.Dispose();
        _checkpoint.Dispose();
    }

    // The bytes of every .safetensors file the directory holds (the shard index is not read; the sizes are the files' own).
    private static long WeightBytes(string directory) =>
        Directory.GetFiles(directory, "*.safetensors").Sum(static f => new FileInfo(f).Length);
}
