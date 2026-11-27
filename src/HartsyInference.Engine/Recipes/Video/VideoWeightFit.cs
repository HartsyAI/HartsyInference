using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Engine.Placement;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The device-fit refusal every video recipe shares: a backend with a streaming cache moves the DiT's blocks through the device and is never
/// refused; one without keeps every block on the device, so weights larger than its usable memory cannot run there.</summary>
/// <remarks>Only the weights are counted, so a refusal is a run that could not have loaded. Activations and LoRA operands come on top and are not judged.
/// The weight size is computed lazily, so a streaming backend or one without device memory reads no file.</remarks>
public static class VideoWeightFit
{
    /// <summary>Why <paramref name="backend"/> cannot hold the weights <paramref name="model"/> needs, or null.</summary>
    /// <param name="weights">The weight bytes and an optional note on how they are held (for example the load dtype).</param>
    public static string? Refusal(IBackend backend, string model, Func<(long Bytes, string Detail)> weights)
    {
        if (backend.StreamingCache is not null) return null;
        long total = backend.GetVramInfo().TotalBytes;
        if (total <= 0) return null;
        (long bytes, string detail) = weights();
        return Decide(backend.Capabilities.Name, total, bytes, model, detail);
    }

    /// <summary>The refusal for <paramref name="weightBytes"/> on a non-streaming device of <paramref name="totalBytes"/>, or null when it fits the usable
    /// memory (the total less the per-device reserve). Pure: the backend's probe is the caller's.</summary>
    public static string? Decide(string backendName, long totalBytes, long weightBytes, string model, string detail)
    {
        long usable = Math.Max(0, totalBytes - PlacementPlanner.PerDeviceReserveBytes);
        if (weightBytes <= usable) return null;
        return $"{model} cannot run on {backendName}: this backend has no weight streaming, so every DiT block stays on the device, "
            + $"and the weights take {ByteFormat.GbF1(weightBytes)}{detail}. The device has {ByteFormat.GbF1(usable)} usable of {ByteFormat.GbF1(totalBytes)}. "
            + "The CUDA backend streams the blocks.";
    }
}
