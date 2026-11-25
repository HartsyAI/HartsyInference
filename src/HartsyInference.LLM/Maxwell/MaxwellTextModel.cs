using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Maxwell;

/// <summary>Everything the Text path needs to run a Maxwell directory: the generation model, the tokenizer from <c>tokenizer.json</c>, and the
/// chat template from <c>chat_template.jinja</c>, which the release requires (thinking is on by default in it).</summary>
/// <remarks>Owns the generation model, which owns the checkpoint.</remarks>
public sealed class MaxwellTextModel : IHostReferenceTextModel
{
    private MaxwellTextModel(MaxwellGenerationModel generation, ILlmTokenizer tokenizer, IChatTemplate template)
    {
        Generation = generation;
        Tokenizer = tokenizer;
        Template = template;
    }

    /// <summary>The engine-facing model.</summary>
    public MaxwellGenerationModel Generation { get; }

    /// <summary>The loaded checkpoint model under <see cref="Generation"/>.</summary>
    public MaxwellModel Model => Generation.Model;

    /// <inheritdoc />
    public ILlmTokenizer Tokenizer { get; }

    /// <inheritdoc />
    public IChatTemplate Template { get; }

    IGenerationModel IHostReferenceTextModel.Generation => Generation;

    /// <summary>Loads <paramref name="directory"/>; the result owns everything it opened.</summary>
    /// <exception cref="HartsyInferenceException">The directory lacks a tokenizer or chat template, the tokenizer cannot be applied, or its
    /// special ids disagree with <c>config.json</c>.</exception>
    public static MaxwellTextModel Load(IBackend backend, string directory, MaxwellLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(options);
        string tokenizerPath = Path.Combine(directory, "tokenizer.json"), templatePath = Path.Combine(directory, "chat_template.jinja");
        if (!File.Exists(tokenizerPath))
            throw new HartsyInferenceException($"'{directory}' has no tokenizer.json, which the Maxwell Text path needs to build its prompt.");
        if (!File.Exists(templatePath))
            throw new HartsyInferenceException($"'{directory}' has no chat_template.jinja, which the Maxwell Text path needs.");

        // Config and tokenizer first: both are cheap, so a bad file fails before the weight load.
        MaxwellConfig config = MaxwellConfig.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        (string? bosLiteral, string? eosLiteral) = HfChatAssets.SpecialTokenLiterals(directory);
        List<int> stopIds = HfStopIds.Read(directory, config.EosTokenId);
        ILlmTokenizer tokenizer;
        IChatTemplate template;
        try
        {
            (tokenizer, template) = HfChatAssets.Load(directory, stopIds, bosLiteral, eosLiteral);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or System.Text.Json.JsonException or FormatException)
        {
            throw new HartsyInferenceException($"'{tokenizerPath}' or '{templatePath}' could not be loaded: {ex.Message}", ex);
        }

        // A mismatch would make the model read the wrong embeddings, so refuse rather than guess; checked before the weight load.
        if (config.BosTokenId is { } bos && tokenizer.BosId != bos)
            throw new HartsyInferenceException($"tokenizer.json puts begin-of-sentence at {tokenizer.BosId?.ToString() ?? "no id"}, but config.json says {bos}.");
        if (config.EosTokenId is { } eos && tokenizer.EosId != eos)
            throw new HartsyInferenceException($"tokenizer.json puts end-of-sentence at {tokenizer.EosId?.ToString() ?? "no id"}, but config.json says {eos}.");

        MaxwellModel model = MaxwellModel.Load(directory, backend, options);
        try
        {
            MaxwellGenerationModel generation = new(model, backend, options.MaxSequenceTokens);
            return new MaxwellTextModel(generation, tokenizer, template);
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
