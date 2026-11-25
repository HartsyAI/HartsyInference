using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>F32 reads of a model's small vectors and matrices from an <see cref="IModelWeightSource"/>.</summary>
internal static class WeightReads
{
    /// <summary>A weight as an F32 tensor the caller owns; a block-scaled FP8 weight is decoded through its grid and any other dtype is cast.</summary>
    public static Tensor ReadF32(IModelWeightSource source, string name)
    {
        Tensor t = source.Read(name);
        if (t.DType == DType.F32) return t;
        using (t) return ProjectionOps.EnsureF32(t);
    }

    /// <summary>A weight copied to a host F32 array.</summary>
    public static float[] HostF32(IModelWeightSource source, string name)
    {
        using Tensor t = ReadF32(source, name);
        return t.AsReadOnlySpan<float>().ToArray();
    }
}
