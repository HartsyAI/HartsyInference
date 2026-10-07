using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Everything the Text path needs to run a V4.1 directory: the generation model, the tokenizer from the checkpoint's <c>tokenizer.json</c>, and the chat template.</summary>
/// <remarks>Owns the generation model, which owns the checkpoint. This is the host reference path, not a serving path: dense weights are F32 on the host.</remarks>
public sealed class DeepSeekV41TextModel : IDisposable
{
    /// <summary>The begin-of-sentence literal in the official <c>tokenizer.json</c>.</summary>
    public const string BosLiteral = "<｜begin▁of▁sentence｜>";

    /// <summary>The end-of-sentence literal in the official <c>tokenizer.json</c>.</summary>
    public const string EosLiteral = "<｜end▁of▁sentence｜>";

    /// <summary>The engine-facing model.</summary>
    public DeepSeekV41GenerationModel Generation { get; }

    /// <summary>The checkpoint's byte-level BPE tokenizer, with begin and end of sentence set.</summary>
    public ILlmTokenizer Tokenizer { get; }

    /// <summary>The V4.1 conversation encoder behind the template interface.</summary>
    public ChatTemplateEncoderAdapter Template { get; }

    private DeepSeekV41TextModel(DeepSeekV41GenerationModel generation, ILlmTokenizer tokenizer, ChatTemplateEncoderAdapter template)
    {
        Generation = generation;
        Tokenizer = tokenizer;
        Template = template;
    }

    /// <summary>Loads <paramref name="directory"/>; the result owns everything it opened.</summary>
    /// <exception cref="HartsyInferenceException">The directory has no <c>tokenizer.json</c>, or it is not a byte-level BPE file.</exception>
    public static DeepSeekV41TextModel Load(IBackend backend, string directory, DeepSeekV41LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(backend);
        string tokenizerPath = Path.Combine(directory, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
            throw new HartsyInferenceException($"'{directory}' has no tokenizer.json, which the DeepSeek-V4.1 Text path needs to build its prompt.");

        // Tokenizer first: it is cheap, so a bad file fails before the multi-GiB weight load.
        GgufTokenizer tokenizer;
        try
        {
            using FileStream stream = File.OpenRead(tokenizerPath);
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream, bosToken: BosLiteral, eosToken: EosLiteral);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or System.Text.Json.JsonException)
        {
            throw new HartsyInferenceException($"'{tokenizerPath}' is not a usable byte-level BPE tokenizer: {ex.Message}", ex);
        }

        DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, directory, options);
        try
        {
            DeepSeekV41Config cfg = loaded.Checkpoint.Config;
            // a mismatch would make the model read the wrong embeddings, so refuse rather than guess
            if (cfg.BosTokenId is { } bos && tokenizer.BosId != bos)
                throw new HartsyInferenceException($"tokenizer.json puts begin-of-sentence at {tokenizer.BosId?.ToString() ?? "no id"}, but config.json says {bos}.");
            if (cfg.EosTokenId is { } eos && tokenizer.EosId != eos)
                throw new HartsyInferenceException($"tokenizer.json puts end-of-sentence at {tokenizer.EosId?.ToString() ?? "no id"}, but config.json says {eos}.");
            return new DeepSeekV41TextModel(new DeepSeekV41GenerationModel(loaded, backend), tokenizer, new ChatTemplateEncoderAdapter(new DeepSeekV41Encoder()));
        }
        catch
        {
            loaded.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public void Dispose() => Generation.Dispose();
}
