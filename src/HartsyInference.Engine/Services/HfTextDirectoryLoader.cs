using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Services;

/// <summary>Opens a Hugging Face safetensors directory for the Text path; only DeepSeek-V4.1 is recognised, and it runs on the host reference model.</summary>
internal static class HfTextDirectoryLoader
{
    /// <summary>Longest sequence (prompt plus generation) a loaded V4.1 model accepts. It sizes the rope tables and every sequence state; the checkpoint's own limit is far higher, but a host reference run does not need it.</summary>
    internal const int MaxSequenceTokens = 16384;

    /// <summary>Loads the directory onto <paramref name="backend"/> (the CPU backend: the reference model keeps its weights on the host).</summary>
    /// <exception cref="HartsyInferenceException">The directory's model_type is not supported, it has no usable tokenizer.json, or the checkpoint is invalid.</exception>
    internal static DeepSeekV41TextModel Load(HfCheckpointInfo info, IBackend backend)
    {
        RequireSupported(info);
        return DeepSeekV41TextModel.Load(backend, info.Root, new DeepSeekV41LoadOptions(MaxSequenceTokens));
    }

    /// <summary>Refuses a directory whose model_type the Text path cannot run, before anything is opened or unloaded.</summary>
    /// <exception cref="HartsyInferenceException">The model_type is not supported.</exception>
    internal static void RequireSupported(HfCheckpointInfo info)
    {
        if (!string.Equals(info.ModelType, DeepSeekV41Config.ModelType, StringComparison.Ordinal))
        {
            throw new HartsyInferenceException(
                $"'{info.Root}' is a Hugging Face checkpoint with model_type '{info.ModelType}', which the Text path cannot load; "
                + "pass a .gguf file instead.");
        }
    }
}
