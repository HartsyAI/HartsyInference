using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Features;

/// <summary>Staging helpers that sever a checkpoint's mmap so the loader can be released before the pipeline runs.</summary>
public static class WeightStaging
{
    /// <summary>Copies every weight into an owned F32 tensor and releases any source that owned its memory; the result no longer borrows the safetensors mmap, so the loader can be disposed immediately.</summary>
    public static Dictionary<string, Tensor> ToOwnedF32(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            Tensor f32 = kvp.Value.CastTo(DType.F32);
            if (kvp.Value.OwnsMemory)
            {
                kvp.Value.Dispose();
            }
            result[kvp.Key] = f32;
        }
        return result;
    }
}
