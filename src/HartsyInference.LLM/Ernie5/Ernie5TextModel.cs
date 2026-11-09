using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Everything the Text path needs to run an ERNIE-5.5 directory on the host: the generation model, the tokenizer from the checkpoint's
/// <c>tokenizer.json</c>, and the chat template from its <c>chat_template.jinja</c>.</summary>
/// <remarks>Owns the generation model, which owns the checkpoint. The host reference path only: weights are read on the CPU.</remarks>
public sealed class Ernie5TextModel : IHostReferenceTextModel
{
    private Ernie5TextModel(Ernie5GenerationModel generation, ILlmTokenizer tokenizer, IChatTemplate template)
    {
        Generation = generation;
        Tokenizer = tokenizer;
        Template = template;
    }

    /// <summary>The engine-facing model.</summary>
    public Ernie5GenerationModel Generation { get; }

    /// <inheritdoc />
    public ILlmTokenizer Tokenizer { get; }

    /// <inheritdoc />
    public IChatTemplate Template { get; }

    IGenerationModel IHostReferenceTextModel.Generation => Generation;

    /// <summary>Loads <paramref name="directory"/>; the result owns everything it opened.</summary>
    /// <exception cref="HartsyInferenceException">The directory lacks a tokenizer or chat template, the tokenizer is not byte-level BPE, or its
    /// special ids disagree with <c>config.json</c>.</exception>
    public static Ernie5TextModel Load(IBackend backend, string directory, Ernie5LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(options);
        string tokenizerPath = Path.Combine(directory, "tokenizer.json"), templatePath = Path.Combine(directory, "chat_template.jinja");
        if (!File.Exists(tokenizerPath))
            throw new HartsyInferenceException($"'{directory}' has no tokenizer.json, which the ERNIE-5.5 Text path needs to build its prompt.");
        if (!File.Exists(templatePath))
            throw new HartsyInferenceException($"'{directory}' has no chat_template.jinja, which the ERNIE-5.5 Text path needs.");

        // Tokenizer first: it is cheap, so a bad file fails before the weight load.
        (string? bosLiteral, string? eosLiteral) = HfChatAssets.SpecialTokenLiterals(directory);
        GgufTokenizer tokenizer;
        try
        {
            using FileStream stream = File.OpenRead(tokenizerPath);
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream, bosToken: bosLiteral, eosToken: eosLiteral);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or System.Text.Json.JsonException)
        {
            throw new HartsyInferenceException($"'{tokenizerPath}' is not a usable byte-level BPE tokenizer: {ex.Message}", ex);
        }

        Ernie5Model model = Ernie5Model.Load(directory, backend, options);
        try
        {
            Ernie5Config cfg = model.Config;
            // a mismatch would make the model read the wrong embeddings, so refuse rather than guess
            if (cfg.BosTokenId is { } bos && tokenizer.BosId != bos)
                throw new HartsyInferenceException($"tokenizer.json puts begin-of-sentence at {tokenizer.BosId?.ToString() ?? "no id"}, but config.json says {bos}.");
            if (cfg.EosTokenId is { } eos && tokenizer.EosId != eos)
                throw new HartsyInferenceException($"tokenizer.json puts end-of-sentence at {tokenizer.EosId?.ToString() ?? "no id"}, but config.json says {eos}.");
            Ernie5GenerationModel generation = new(model, backend, options.MaxSequenceTokens);
            return new Ernie5TextModel(generation, tokenizer, new JinjaChatTemplate(File.ReadAllText(templatePath)));
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public void Dispose() => Generation.Dispose();
}
