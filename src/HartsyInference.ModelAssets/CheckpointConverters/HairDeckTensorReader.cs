using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Reads named F32 tensors of a HairDeck checkpoint, failing on a missing key, a dtype or a shape mismatch.</summary>
internal static class HairDeckTensorReader
{
    public static Tensor Take(IReadOnlyDictionary<string, Tensor> all, string key, params long[] shape)
    {
        Tensor t = Find(all, key);
        if (!HasShape(t, shape))
            throw new InvalidDataException($"HairDeck tensor '{key}' has shape {t.Shape}, expected [{string.Join(", ", shape)}].");
        return t;
    }

    public static Tensor TakeRank(IReadOnlyDictionary<string, Tensor> all, string key, int rank)
    {
        Tensor t = Find(all, key);
        if (t.Shape.Rank != rank)
            throw new InvalidDataException($"HairDeck tensor '{key}' has rank {t.Shape.Rank}, expected {rank}.");
        return t;
    }

    public static bool Exists(IReadOnlyDictionary<string, Tensor> all, string key) => all.ContainsKey(key);

    private static Tensor Find(IReadOnlyDictionary<string, Tensor> all, string key)
    {
        ArgumentNullException.ThrowIfNull(all);
        if (!all.TryGetValue(key, out Tensor? t))
            throw new InvalidDataException($"HairDeck checkpoint is missing tensor '{key}'.");
        if (t.DType != DType.F32)
            throw new InvalidDataException($"HairDeck tensor '{key}' must be F32, got {t.DType}.");
        return t;
    }

    private static bool HasShape(Tensor t, long[] shape)
    {
        if (shape.Length == 0) return true;
        if (t.Shape.Rank != shape.Length) return false;
        for (int i = 0; i < shape.Length; i++)
            if (t.Shape[i] != shape[i]) return false;
        return true;
    }
}
