using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Validates a checkpoint against a config's tensor specs and resolves each tensor to F32 (or, for the tensors a caller selects, to a narrower type).</summary>
internal static class ShapeVaeWeights
{
    /// <summary>Returns the spec'd tensors keyed by checkpoint key. Allocating casts go to <paramref name="owned"/>; F32 is borrowed.</summary>
    internal static Dictionary<string, Tensor> Resolve(
        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs,
        IReadOnlyDictionary<string, Tensor> weights,
        List<Tensor> owned,
        string owner = "Shape VAE",
        DType? narrowDType = null,
        Func<string, Tensor, bool>? narrow = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Dictionary<string, Tensor> resolved = new(specs.Count);
        foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
        {
            if (!weights.TryGetValue(spec.CheckpointKey, out Tensor? t))
                throw new HartsyInferenceException($"{owner} checkpoint is missing '{spec.CheckpointKey}' ({spec.ParameterName}).");
            if (!HasShape(t, spec.Shape))
                throw new HartsyInferenceException(
                    $"{owner} tensor '{spec.CheckpointKey}' is {ShapeText(t.Shape)}, expected {ShapeText(spec.Shape)}.");
            // The tensors a caller opts into narrowing go straight to the narrow type: from BF16 or F32 in one pass, never through an F32 copy.
            Tensor f = narrow is not null && narrowDType is { } target && target != DType.F32 && narrow(spec.CheckpointKey, t)
                ? (t.DType == target ? t : t.CastTo(target))
                : TensorCasts.EnsureF32(t);
            if (!ReferenceEquals(f, t)) owned.Add(f);
            resolved[spec.CheckpointKey] = f;
        }
        return resolved;
    }

    private static bool HasShape(Tensor t, long[] expected)
    {
        if (t.Shape.Rank != expected.Length) return false;
        for (int d = 0; d < expected.Length; d++)
            if (t.Shape[d] != expected[d]) return false;
        return true;
    }

    private static string ShapeText(TensorShape shape)
    {
        long[] dims = new long[shape.Rank];
        for (int d = 0; d < dims.Length; d++) dims[d] = shape[d];
        return ShapeText(dims);
    }

    private static string ShapeText(long[] dims) => "[" + string.Join(", ", dims) + "]";
}
