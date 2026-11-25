namespace HartsyInference.Core.Tensors;

/// <summary>Reads a model's tensors out of a checkpoint dictionary by name and shape. A wrong or truncated checkpoint fails here, naming the
/// family and the key, instead of in the first op that reads it.</summary>
/// <remarks>Every method borrows from the dictionary except where a widened copy is needed; that copy joins the caller's owned list, so exactly
/// the tensors a module allocated are the ones it disposes.</remarks>
public static class CheckpointTensors
{
    /// <summary>The tensor under <paramref name="key"/>, borrowed. A shape, when given, must match exactly.</summary>
    /// <exception cref="KeyNotFoundException">The checkpoint has no such tensor.</exception>
    /// <exception cref="InvalidDataException">The tensor has another shape.</exception>
    public static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string family, string key, params long[] shape)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(shape);
        if (!weights.TryGetValue(key, out Tensor? tensor))
            throw new KeyNotFoundException($"{family} checkpoint has no tensor '{key}'.");
        return Check(tensor, family, key, shape);
    }

    private static Tensor Check(Tensor tensor, string family, string key, long[] shape)
    {
        if (shape.Length > 0 && !HasShape(tensor, shape))
            throw new InvalidDataException($"{family} tensor '{key}' has shape {tensor.Shape}, expected [{string.Join(", ", shape)}].");
        return tensor;
    }

    /// <summary><see cref="Require"/> as F32. A 16-bit tensor is widened into a copy that joins <paramref name="owned"/>; an F32 one is borrowed.</summary>
    /// <exception cref="NotSupportedException">The tensor carries a quantization recipe: a cast would drop its scales and load wrong values.</exception>
    public static Tensor RequireF32(IReadOnlyDictionary<string, Tensor> weights, string family, string key, ICollection<Tensor> owned, params long[] shape) =>
        Widen(Require(weights, family, key, shape), family, key, owned);

    /// <summary><see cref="RequireF32"/> for a tensor the checkpoint may leave out: null when it is absent.</summary>
    public static Tensor? OptionalF32(IReadOnlyDictionary<string, Tensor> weights, string family, string key, ICollection<Tensor> owned, params long[] shape)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(shape);
        return weights.TryGetValue(key, out Tensor? tensor) ? Widen(Check(tensor, family, key, shape), family, key, owned) : null;
    }

    private static Tensor Widen(Tensor tensor, string family, string key, ICollection<Tensor> owned)
    {
        ArgumentNullException.ThrowIfNull(owned);
        if (tensor.DType == DType.F32)
            return tensor;
        if (tensor.QuantInfo is { } quant)
            throw new NotSupportedException($"{family} tensor '{key}' is {tensor.DType} in {quant.Format}; a plain cast would drop its scales.");
        Tensor widened = tensor.CastTo(DType.F32);
        owned.Add(widened);
        return widened;
    }

    private static bool HasShape(Tensor tensor, long[] shape)
    {
        if (tensor.Shape.Rank != shape.Length)
            return false;
        for (int axis = 0; axis < shape.Length; axis++)
        {
            if (tensor.Shape[axis] != shape[axis])
                return false;
        }
        return true;
    }
}
