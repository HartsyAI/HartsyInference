using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>Everything the Text path needs to run a glm5_next directory (Sage S2.1 Flash): the generation model, the tokenizer from the checkpoint's
/// <c>tokenizer.json</c> and the chat template from its <c>chat_template.jinja</c>.</summary>
/// <remarks>Owns the generation model and the checkpoint reader it reads from. Runs on the one backend it is loaded on. The template renders with the
/// engine's Jinja engine, which escapes control-token spellings in message content so they encode as ordinary text, as the repository's
/// <c>tokenization_glm5.py</c> requires. <c>reasoning_effort</c> is not passed to the template, so it renders its default, <c>max</c>.</remarks>
public sealed class Glm5NextTextModel : IHostReferenceTextModel
{
    /// <summary>The <c>model_type</c> of a glm5_next checkpoint.</summary>
    public const string ModelType = "glm5_next";

    private readonly BlockFp8CheckpointWeights _weights;

    private Glm5NextTextModel(Glm5NextGenerationModel generation, ILlmTokenizer tokenizer, IChatTemplate template, BlockFp8CheckpointWeights weights)
    {
        Generation = generation;
        Tokenizer = tokenizer;
        Template = template;
        _weights = weights;
    }

    /// <summary>The engine-facing model.</summary>
    public Glm5NextGenerationModel Generation { get; }

    /// <inheritdoc />
    public ILlmTokenizer Tokenizer { get; }

    /// <inheritdoc />
    public IChatTemplate Template { get; }

    IGenerationModel IHostReferenceTextModel.Generation => Generation;

    /// <summary>Loads <paramref name="directory"/>; the result owns everything it opened.</summary>
    /// <param name="maxSequenceTokens">The longest sequence (prompt plus reply) the model accepts; it sizes every sequence's DSA caches.</param>
    /// <exception cref="HartsyInferenceException">The directory lacks a tokenizer or template, a file cannot be loaded, or the config is not a variant the engine runs.</exception>
    public static Glm5NextTextModel Load(IBackend backend, string directory, int maxSequenceTokens)
    {
        ArgumentNullException.ThrowIfNull(backend);
        string tokenizerPath = Path.Combine(directory, "tokenizer.json"), templatePath = Path.Combine(directory, "chat_template.jinja");
        if (!File.Exists(tokenizerPath)) throw new HartsyInferenceException($"'{directory}' has no tokenizer.json, which the Sage S2.1 Flash Text path needs.");
        if (!File.Exists(templatePath)) throw new HartsyInferenceException($"'{directory}' has no chat_template.jinja, which the Sage S2.1 Flash Text path needs.");

        Glm5NextConfig config = ReadConfig(Path.Combine(directory, "config.json"));
        (string? bos, string? eos) = HfChatAssets.SpecialTokenLiterals(directory);
        ILlmTokenizer tokenizer;
        IChatTemplate template;
        try
        {
            (tokenizer, template) = HfChatAssets.Load(directory, config.EosTokenIds, bos, eos, hfWhitespace: true);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or JsonException or FormatException)
        {
            throw new HartsyInferenceException($"'{tokenizerPath}' or '{templatePath}' could not be loaded: {ex.Message}", ex);
        }

        BlockFp8CheckpointWeights weights = BlockFp8CheckpointWeights.Open(directory, ModelType);
        Glm5NextModel? model = null;
        try
        {
            model = new Glm5NextModel(backend, config, weights);
            return new Glm5NextTextModel(new Glm5NextGenerationModel(model, backend, maxSequenceTokens), tokenizer, template, weights);
        }
        catch
        {
            model?.Dispose();
            weights.Dispose();
            throw;
        }
    }

    /// <summary>Reads and validates <c>config.json</c> without opening a weight.</summary>
    /// <exception cref="HartsyInferenceException">The config is not a glm5_next decoder the engine can build.</exception>
    public static Glm5NextConfig ReadConfig(string configPath)
    {
        try
        {
            return Glm5NextHfConfig.Parse(File.ReadAllText(configPath));
        }
        catch (Exception ex) when (ex is InvalidDataException or JsonException or IOException)
        {
            throw new HartsyInferenceException($"'{configPath}' is not a glm5_next config the engine can run: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Generation.Dispose();
        _weights.Dispose();
    }
}
