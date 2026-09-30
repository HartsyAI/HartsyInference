using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Services;

/// <summary>Opens a Hugging Face safetensors directory for the Text path; only DeepSeek-V4.1 is recognised, and its model class is not wired yet.</summary>
internal static class HfTextDirectoryLoader
{
    private const double BytesPerGiB = 1024.0 * 1024.0 * 1024.0;

    /// <summary>Validates the checkpoint from headers, then refuses because no model class consumes it yet.</summary>
    /// <exception cref="NotSupportedException">The checkpoint is a valid DeepSeek-V4.1 directory but the model class is not wired.</exception>
    /// <exception cref="HartsyInferenceException">The directory's model_type is not supported, or the checkpoint is invalid.</exception>
    internal static void Load(HfCheckpointInfo info)
    {
        if (!string.Equals(info.ModelType, DeepSeekV41Config.ModelType, StringComparison.Ordinal))
        {
            throw new HartsyInferenceException(
                $"'{info.Root}' is a Hugging Face checkpoint with model_type '{info.ModelType}', which the Text path cannot load; "
                + "pass a .gguf file instead.");
        }
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(info.Root);
        double gib = checkpoint.Weights.TotalBytes / BytesPerGiB;
        string shape = string.Create(CultureInfo.InvariantCulture,
            $"{checkpoint.Flavor}, {checkpoint.Shards.Shards.Count} shards, {checkpoint.Weights.TotalCount} tensors, {gib:F1} GiB, draft {checkpoint.Draft.Status}");
        throw new NotSupportedException($"DeepSeek-V4.1 checkpoint '{info.Root}' opened and validated ({shape}), but its model class is not wired into TextService yet.");
    }
}
