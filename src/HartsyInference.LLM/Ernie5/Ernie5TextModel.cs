using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Everything the Text path needs to run an ERNIE-5.5 directory: the generation model, the tokenizer from the checkpoint's
/// <c>tokenizer.json</c>, and the chat template from its <c>chat_template.jinja</c>. A base checkpoint without that file (the published
/// ERNIE-5.5-Flash-Base) gets a plain-text completion template instead.</summary>
/// <remarks>Owns the generation model, which owns the checkpoint. The model runs on the backend it is loaded on.</remarks>
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
    /// <exception cref="HartsyInferenceException">The directory lacks a tokenizer, the tokenizer is not byte-level BPE or its post-processor cannot be
    /// applied, or its special ids disagree with <c>config.json</c>.</exception>
    public static Ernie5TextModel Load(IBackend backend, string directory, Ernie5LoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(options);
        string tokenizerPath = Path.Combine(directory, "tokenizer.json"), templatePath = Path.Combine(directory, "chat_template.jinja");
        if (!File.Exists(tokenizerPath))
            throw new HartsyInferenceException($"'{directory}' has no tokenizer.json, which the ERNIE-5.5 Text path needs to build its prompt.");
        // The base checkpoint ships no chat template; Baidu serves it through /v1/completions with a raw prompt, so the prompt is the plain text
        // wrapped as tokenizer.json's post-processor wraps it, not a guessed chat format.
        bool chat = File.Exists(templatePath);

        // Config and tokenizer first: both are cheap, so a bad file fails before the weight load.
        Ernie5Config config = Ernie5Config.Load(Path.Combine(directory, "config.json"));
        (string? bosLiteral, string? eosLiteral) = HfChatAssets.SpecialTokenLiterals(directory);
        ILlmTokenizer tokenizer;
        IChatTemplate template;
        List<int> stopIds = HfStopIds.Read(directory, config.EosTokenId);
        try
        {
            if (chat)
            {
                (tokenizer, template) = HfChatAssets.Load(directory, stopIds, bosLiteral, eosLiteral);
            }
            else
            {
                tokenizer = HfChatAssets.LoadTokenizer(directory, stopIds, bosLiteral, eosLiteral);
                template = HfChatAssets.LoadCompletionTemplate(directory);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or JsonException or FormatException)
        {
            // Both files come through one call, so the message names both rather than guessing which one failed.
            string files = chat ? $"'{tokenizerPath}' or '{templatePath}'" : $"'{tokenizerPath}'";
            throw new HartsyInferenceException($"{files} could not be loaded: {ex.Message}", ex);
        }

        // TODO(ernie5): the Text path never sets Ernie5LoadOptions.Mtp (HfTextDirectoryLoader.Ernie5OptionsFor leaves it false), so MTP layer 43 does not
        // run. Finish it by loading with Mtp on and driving MtpForward as a draft step in generation.
        Ernie5Model model = Ernie5Model.Load(directory, backend, options);
        try
        {
            // a mismatch would make the model read the wrong embeddings, so refuse rather than guess
            if (config.BosTokenId is { } bos && tokenizer.BosId != bos)
                throw new HartsyInferenceException($"tokenizer.json puts begin-of-sentence at {tokenizer.BosId?.ToString() ?? "no id"}, but config.json says {bos}.");
            if (config.EosTokenId is { } eos && tokenizer.EosId != eos)
                throw new HartsyInferenceException($"tokenizer.json puts end-of-sentence at {tokenizer.EosId?.ToString() ?? "no id"}, but config.json says {eos}.");
            Ernie5GenerationModel generation = new(model, backend, options.MaxSequenceTokens);
            return new Ernie5TextModel(generation, tokenizer, template);
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
