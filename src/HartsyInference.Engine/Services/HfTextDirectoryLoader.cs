using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Models.Lfm3;
using HartsyInference.LLM.Models.Spark;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Services;

/// <summary>Opens a Hugging Face safetensors directory for the Text path; DeepSeek-V4.1 (host reference model), stateful families (LFM3, Spark-X) and MiniMax-M3 are recognised.</summary>
internal static class HfTextDirectoryLoader
{
    /// <summary>Longest sequence (prompt plus generation) a loaded V4.1 model accepts. It sizes the rope tables and every sequence state; the checkpoint's own limit is far higher, but a host reference run does not need it.</summary>
    internal const int MaxSequenceTokens = 16384;

    /// <summary>Longest sequence (prompt plus generation) a loaded stateful-family model accepts; it sizes the KV cache of its attention layers.</summary>
    internal const int StatefulMaxSequenceTokens = 8192;

    /// <summary>The options the Text path loads with; the RAM guard and the memory profile size against the same ones.</summary>
    internal static DeepSeekV41LoadOptions LoadOptions { get; } = new(MaxSequenceTokens);

    /// <summary>Loads the directory onto <paramref name="backend"/> (the CPU backend: the reference model keeps its weights on the host).</summary>
    /// <exception cref="HartsyInferenceException">The directory's model_type is not supported, it has no usable tokenizer.json, or the checkpoint is invalid.</exception>
    internal static DeepSeekV41TextModel Load(HfCheckpointInfo info, IBackend backend)
    {
        RequireSupported(info);
        return DeepSeekV41TextModel.Load(backend, info.Root, LoadOptions);
    }

    /// <summary>Whether <paramref name="info"/> is a family that carries its own state (LFM3, Spark-X) rather than DeepSeek-V4.1.</summary>
    internal static bool IsStatefulFamily(HfCheckpointInfo info) => SsmLanguageModel.IsHuggingFaceFamily(info.ModelType);

    /// <summary>Loads a stateful-family directory with its tokenizer and chat template.</summary>
    internal static SsmLanguageModel LoadStateful(HfCheckpointInfo info) =>
        SsmLanguageModel.LoadHuggingFace(info.Root, info.ModelType, StatefulMaxSequenceTokens);

    /// <summary>Loads the MiniMax-M3 directory's text decoder, tokenizer and chat template as the model object the transformer slot serves. Matrices are 4-bit blocks on a backend that reads them,
    /// else F32 (the CPU backend widens quantized weights anyway); <paramref name="memoryBudgetBytes"/> of 0 means no limit.</summary>
    internal static GgufLanguageModel LoadMiniMaxM3(HfCheckpointInfo info, IBackend backend, long memoryBudgetBytes)
    {
        MiniMaxM3HfLoader loader = MiniMaxM3HfLoader.Load(info.Root, backend.Capabilities.SupportsQuantized ? DType.Q4_0 : DType.F32, memoryBudgetBytes);
        try
        {
            GenericTransformer transformer = new(loader.Config);
            transformer.LoadWeights(loader.Weights, "model");
            (ILlmTokenizer tokenizer, IChatTemplate template) = HfChatAssets.Load(info.Root, [], MiniMaxM3Bos, MiniMaxM3Eos);
            return GgufLanguageModel.FromWeights(loader.Config, transformer, tokenizer, template, loader);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private const string MiniMaxM3Bos = "]~b]";
    private const string MiniMaxM3Eos = "[e~[";

    /// <summary>Refuses a directory the Text path cannot run for a reason visible without opening a weight (an unsupported model_type, no tokenizer.json), so the caller can do it before it unloads anything.</summary>
    /// <exception cref="HartsyInferenceException">The model_type is not supported or the tokenizer is missing.</exception>
    internal static void RequireSupported(HfCheckpointInfo info)
    {
        bool stateful = IsStatefulFamily(info);
        bool miniMax = string.Equals(info.ModelType, MiniMaxM3HfConfigReader.ModelType, StringComparison.Ordinal);
        if (!stateful && !miniMax && !string.Equals(info.ModelType, DeepSeekV41Config.ModelType, StringComparison.Ordinal))
        {
            throw new HartsyInferenceException(
                $"'{info.Root}' is a Hugging Face checkpoint with model_type '{info.ModelType}', which the Text path cannot load; "
                + "pass a .gguf file instead.");
        }
        if (miniMax && !File.Exists(Path.Combine(info.Root, "chat_template.jinja")))
            throw new HartsyInferenceException($"'{info.Root}' has no chat_template.jinja, which the MiniMax-M3 Text path needs.");
        if (stateful)
        {
            bool single = string.Equals(info.ModelType, Lfm3Config.ModelType, StringComparison.Ordinal);
            bool hasWeights = single ? File.Exists(Path.Combine(info.Root, "model.safetensors")) : Directory.EnumerateFiles(info.Root, "*.safetensors").Any();
            if (!hasWeights)
                throw new HartsyInferenceException($"'{info.Root}' has no {(single ? "model.safetensors" : "*.safetensors file")}, which the {info.ModelType} Text path needs.");
            if (!File.Exists(Path.Combine(info.Root, "chat_template.jinja")))
                throw new HartsyInferenceException($"'{info.Root}' has no chat_template.jinja, which the {info.ModelType} Text path needs.");
        }
        if (!File.Exists(Path.Combine(info.Root, "tokenizer.json")))
            throw new HartsyInferenceException($"'{info.Root}' has no tokenizer.json, which the Text path needs to build its prompt.");
    }
}
