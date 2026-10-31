using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Models.Lfm3;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Services;

/// <summary>Opens a Hugging Face safetensors directory for the Text path; DeepSeek-V4.1 (host reference model) and LiquidAI LFM3 are recognised.</summary>
internal static class HfTextDirectoryLoader
{
    /// <summary>Longest sequence (prompt plus generation) a loaded V4.1 model accepts. It sizes the rope tables and every sequence state; the checkpoint's own limit is far higher, but a host reference run does not need it.</summary>
    internal const int MaxSequenceTokens = 16384;

    /// <summary>Longest sequence (prompt plus generation) a loaded LFM3 model accepts; it sizes the KV cache of its attention layers.</summary>
    internal const int Lfm3MaxSequenceTokens = 8192;

    /// <summary>The options the Text path loads with; the RAM guard and the memory profile size against the same ones.</summary>
    internal static DeepSeekV41LoadOptions LoadOptions { get; } = new(MaxSequenceTokens);

    /// <summary>Loads the directory onto <paramref name="backend"/> (the CPU backend: the reference model keeps its weights on the host).</summary>
    /// <exception cref="HartsyInferenceException">The directory's model_type is not supported, it has no usable tokenizer.json, or the checkpoint is invalid.</exception>
    internal static DeepSeekV41TextModel Load(HfCheckpointInfo info, IBackend backend)
    {
        RequireSupported(info);
        return DeepSeekV41TextModel.Load(backend, info.Root, LoadOptions);
    }

    /// <summary>Loads the LFM3 directory with its tokenizer and chat template.</summary>
    internal static SsmLanguageModel LoadLfm3(HfCheckpointInfo info) => SsmLanguageModel.LoadLfm3(info.Root, Lfm3MaxSequenceTokens);

    /// <summary>Refuses a directory the Text path cannot run for a reason visible without opening a weight (an unsupported model_type, no tokenizer.json), so the caller can do it before it unloads anything.</summary>
    /// <exception cref="HartsyInferenceException">The model_type is not supported or the tokenizer is missing.</exception>
    internal static void RequireSupported(HfCheckpointInfo info)
    {
        bool lfm3 = string.Equals(info.ModelType, Lfm3Config.ModelType, StringComparison.Ordinal);
        if (!lfm3 && !string.Equals(info.ModelType, DeepSeekV41Config.ModelType, StringComparison.Ordinal))
        {
            throw new HartsyInferenceException(
                $"'{info.Root}' is a Hugging Face checkpoint with model_type '{info.ModelType}', which the Text path cannot load; "
                + "pass a .gguf file instead.");
        }
        if (lfm3)
        {
            foreach (string file in (string[])["model.safetensors", "chat_template.jinja"])
            {
                if (!File.Exists(Path.Combine(info.Root, file)))
                    throw new HartsyInferenceException($"'{info.Root}' has no {file}, which the LFM3 Text path needs.");
            }
        }
        if (!File.Exists(Path.Combine(info.Root, "tokenizer.json")))
            throw new HartsyInferenceException($"'{info.Root}' has no tokenizer.json, which the Text path needs to build its prompt.");
    }
}
