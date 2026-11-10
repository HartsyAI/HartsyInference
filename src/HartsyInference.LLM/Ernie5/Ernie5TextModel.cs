using System.Text.Json;
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

        // Config and tokenizer first: both are cheap, so a bad file fails before the weight load.
        Ernie5Config config = Ernie5Config.Load(Path.Combine(directory, "config.json"));
        (string? bosLiteral, string? eosLiteral) = HfChatAssets.SpecialTokenLiterals(directory);
        ILlmTokenizer tokenizer;
        IChatTemplate template;
        List<int> stopIds = StopIds(directory, config);
        try
        {
            (tokenizer, template) = HfChatAssets.Load(directory, stopIds, bosLiteral, eosLiteral);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or JsonException or FormatException)
        {
            // Both files come through one call, so the message names both rather than guessing which one failed.
            throw new HartsyInferenceException($"'{tokenizerPath}' or '{templatePath}' could not be loaded: {ex.Message}", ex);
        }

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

    /// <summary>The end-of-sequence ids from <c>config.json</c> and <c>generation_config.json</c> (an int or a list), which end generation
    /// alongside the end token.</summary>
    private static List<int> StopIds(string directory, Ernie5Config config)
    {
        List<int> ids = [];
        if (config.EosTokenId is { } eos) ids.Add(eos);
        string path = Path.Combine(directory, "generation_config.json");
        if (!File.Exists(path)) return ids;
        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            if (!document.RootElement.TryGetProperty("eos_token_id", out JsonElement value)) return ids;
            if (value.ValueKind == JsonValueKind.Number) ids.Add(StopId(value, path));
            else if (value.ValueKind == JsonValueKind.Array)
                foreach (JsonElement item in value.EnumerateArray())
                    if (item.ValueKind == JsonValueKind.Number) ids.Add(StopId(item, path));
        }
        catch (JsonException ex)
        {
            throw new HartsyInferenceException($"'{path}' is not valid JSON: {ex.Message}", ex);
        }
        return [.. ids.Distinct()];
    }

    private static int StopId(JsonElement value, string path)
    {
        if (value.TryGetInt32(out int id)) return id;
        throw new HartsyInferenceException($"'{path}' names eos_token_id {value.GetRawText()}, which is not a 32-bit token id.");
    }

    /// <inheritdoc />
    public void Dispose() => Generation.Dispose();
}
