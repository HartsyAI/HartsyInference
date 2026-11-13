using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Kimi.Vision;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Kimi;

/// <summary>Everything the Text path needs to run a Kimi K3.1 directory: the generation model, the tokenizer from the checkpoint's
/// <c>tokenizer.json</c> or, when that is absent, its <c>tiktoken.model</c>, and the chat template from its <c>chat_template.jinja</c> or, when that
/// is absent on a <c>tiktoken.model</c> checkpoint, the built-in XTML template (<see cref="KimiK31ChatTemplate"/>).</summary>
/// <remarks>Takes images through <see cref="Vision"/> when the checkpoint's vision tower loads; otherwise image input is refused by name. Owns the
/// generation model, which owns the checkpoint, and the vision input.</remarks>
public sealed class KimiK31TextModel : IHostReferenceTextModel
{
    private KimiK31TextModel(KimiK31GenerationModel generation, ILlmTokenizer tokenizer, IChatTemplate template, KimiVisionInput vision)
    {
        Generation = generation;
        Tokenizer = tokenizer;
        Template = template;
        Vision = vision;
    }

    /// <summary>The engine-facing model.</summary>
    public KimiK31GenerationModel Generation { get; }

    /// <summary>The checkpoint's image inputs: the vision encoder, or the reason images are refused.</summary>
    public KimiVisionInput Vision { get; }

    /// <inheritdoc />
    public ILlmTokenizer Tokenizer { get; }

    /// <inheritdoc />
    public IChatTemplate Template { get; }

    IGenerationModel IHostReferenceTextModel.Generation => Generation;

    /// <summary>Loads <paramref name="directory"/> onto <paramref name="backend"/>; the result owns everything it opened.</summary>
    /// <param name="maxSequenceTokens">Longest sequence (prompt plus generation) the decoder accepts; it sizes the positional tables.</param>
    /// <exception cref="HartsyInferenceException">The directory has neither tokenizer file, lacks a chat template where it has no <c>tiktoken.model</c>, the tokenizer is not usable, or its special ids disagree with <c>config.json</c>.</exception>
    /// <exception cref="InvalidDataException">The config or a tensor does not match the Kimi K3.1 decoder.</exception>
    public static KimiK31TextModel Load(IBackend backend, string directory, int maxSequenceTokens) =>
        Load(backend, directory, maxSequenceTokens, KimiK31CheckpointWeights.Open);

    /// <summary>Loads like <see cref="Load(IBackend, string, int)"/>, reading the checkpoint's tensors from the stores <paramref name="openWeights"/> opens.</summary>
    internal static KimiK31TextModel Load(IBackend backend, string directory, int maxSequenceTokens, Func<string, IKimiK31WeightStore> openWeights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentNullException.ThrowIfNull(openWeights);
        string tokenizerPath = Path.Combine(directory, "tokenizer.json"), templatePath = Path.Combine(directory, "chat_template.jinja");
        string tiktokenPath = Path.Combine(directory, KimiK31Tokenizer.TiktokenFile);
        // The Moonshot repo ships tiktoken.model and no tokenizer.json; tokenizer.json wins when both are present.
        bool useTiktoken = !File.Exists(tokenizerPath);
        if (useTiktoken && !File.Exists(tiktokenPath))
            throw new HartsyInferenceException($"'{directory}' has no tokenizer.json or tiktoken.model, which the Kimi K3.1 Text path needs to build its prompt.");
        bool hasTemplate = File.Exists(templatePath);
        if (!useTiktoken && !hasTemplate)
            throw new HartsyInferenceException($"'{directory}' has no chat_template.jinja, which the Kimi K3.1 Text path needs.");

        // Tokenizer and template first: both are cheap, so a bad file fails before the weight load.
        (int? configBos, int? configEos) = ReadSpecialIds(Path.Combine(directory, "config.json"));
        List<int> stopIds = StopIds(directory, configEos);
        (string? bosLiteral, string? eosLiteral) = HfChatAssets.SpecialTokenLiterals(directory);
        ILlmTokenizer tokenizer;
        IChatTemplate template;
        try
        {
            if (useTiktoken)
            {
                tokenizer = KimiK31Tokenizer.Load(directory, stopIds, configBos, configEos);
                template = hasTemplate ? new JinjaChatTemplate(File.ReadAllText(templatePath)) : new KimiK31ChatTemplate();
            }
            else
            {
                (tokenizer, template) = HfChatAssets.Load(directory, stopIds, bosLiteral, eosLiteral);
            }
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or JsonException or FormatException or InvalidDataException or FileNotFoundException)
        {
            throw new HartsyInferenceException($"'{(useTiktoken ? tiktokenPath : tokenizerPath)}' or '{templatePath}' could not be loaded: {ex.Message}", ex);
        }

        KimiK31Checkpoint checkpoint = KimiK31Checkpoint.Load(directory, backend, maxSequenceTokens, openWeights);
        KimiVisionInput? vision = null;
        try
        {
            // a mismatch would make the model read the wrong embeddings, so refuse rather than guess
            string tokenizerFile = useTiktoken ? KimiK31Tokenizer.TiktokenFile : "tokenizer.json";
            if (configBos is { } bos && tokenizer.BosId != bos)
                throw new HartsyInferenceException(
                    $"{tokenizerFile} puts begin-of-sentence at {tokenizer.BosId?.ToString() ?? "no id"}, but config.json says {bos}.");
            if (configEos is { } eos && tokenizer.EosId != eos)
                throw new HartsyInferenceException(
                    $"{tokenizerFile} puts end-of-sentence at {tokenizer.EosId?.ToString() ?? "no id"}, but config.json says {eos}.");
            // Reads no vision tensor: the tower loads on the first image request, so a text load never depends on it.
            vision = KimiVisionInput.Load(directory, tokenizer, checkpoint.Config.Hidden, template is KimiK31ChatTemplate, openWeights);
            return new KimiK31TextModel(new KimiK31GenerationModel(checkpoint, backend, maxSequenceTokens), tokenizer, template, vision);
        }
        catch
        {
            vision?.Dispose();
            checkpoint.Dispose();
            throw;
        }
    }

    /// <summary>The top-level <c>bos_token_id</c> and <c>eos_token_id</c> of <c>config.json</c>, each null when absent.</summary>
    private static (int? Bos, int? Eos) ReadSpecialIds(string configPath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(configPath));
        JsonElement root = document.RootElement;
        return (IntOrNull(root, "bos_token_id"), IntOrNull(root, "eos_token_id"));
    }

    private static int? IntOrNull(JsonElement root, string name) =>
        root.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int id) ? id : null;

    /// <summary>The end-of-sequence ids: <c>config.json</c>'s, and <c>generation_config.json</c>'s (an int or a list), each once.</summary>
    private static List<int> StopIds(string directory, int? configEos)
    {
        List<int> ids = [];
        if (configEos is { } eos) ids.Add(eos);
        string path = Path.Combine(directory, "generation_config.json");
        if (!File.Exists(path)) return ids;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        if (!document.RootElement.TryGetProperty("eos_token_id", out JsonElement value)) return ids;
        if (value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int single)) ids.Add(single);
        else if (value.ValueKind == JsonValueKind.Array)
            foreach (JsonElement item in value.EnumerateArray())
                if (item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out int id)) ids.Add(id);
        return [.. ids.Distinct()];
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Vision.Dispose();
        Generation.Dispose();
    }
}
