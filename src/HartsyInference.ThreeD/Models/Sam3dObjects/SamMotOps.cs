using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Layout helpers of the structure flow, on F32 token-major <c>[1, tokens, width]</c> tensors.</summary>
internal static class SamMotOps
{
    /// <summary>The <paramref name="width"/> columns from <paramref name="offset"/> of <paramref name="packed"/> (a fused projection split).</summary>
    public static Tensor Slice(IBackend backend, Tensor packed, int offset, int width)
    {
        Tensor result = new(new TensorShape(1, packed.Shape[1], width), DType.F32);
        backend.SliceLastDim(result, packed, offset);
        return result;
    }

    /// <summary>Token-major <c>[1, n, heads·dim]</c> to head-major <c>[1, heads, n, dim]</c>; <paramref name="flat"/> is disposed.</summary>
    public static Tensor ToHeadMajor(IBackend backend, Tensor flat, int heads, int headDim)
    {
        int tokens = (int)flat.Shape[1];
        Tensor result = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        backend.Permute0213(result, flat, tokens, heads, headDim);
        flat.Dispose();
        return result;
    }

    /// <summary>Head-major <c>[1, heads, n, dim]</c> back to token-major <c>[1, n, heads·dim]</c>; <paramref name="headMajor"/> is disposed.</summary>
    public static Tensor ToTokenMajor(IBackend backend, Tensor headMajor)
    {
        int heads = (int)headMajor.Shape[1], tokens = (int)headMajor.Shape[2], headDim = (int)headMajor.Shape[3];
        Tensor result = new(new TensorShape(1, tokens, heads * headDim), DType.F32);
        backend.Permute0213(result, headMajor, heads, tokens, headDim);
        headMajor.Dispose();
        return result;
    }

    /// <summary>The parts joined on the token axis, in order, on the backend; the parts are not touched.</summary>
    public static Tensor Concat(IBackend backend, IReadOnlyList<Tensor> parts)
    {
        long tokens = 0;
        foreach (Tensor part in parts)
            tokens += part.Shape[1];
        Tensor result = new(new TensorShape(1, tokens, parts[0].Shape[2]), DType.F32);
        backend.Concat(result, parts.ToArray(), 1);
        return result;
    }

    /// <summary>A copy of <paramref name="count"/> tokens from <paramref name="start"/>, on the backend.</summary>
    public static Tensor Tokens(IBackend backend, Tensor source, int start, int count)
    {
        Tensor result = new(new TensorShape(1, count, source.Shape[2]), DType.F32);
        backend.SliceRows(result, source, start);
        return result;
    }
}
