using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Rotates only the leading <paramref name="rows"/> of each head of a joint head-major Q and K (the image rows
/// ahead of the text) with <paramref name="inner"/>, by copying them out, rotating them as a standalone
/// <c>[B, H, rows, D]</c> pair and copying them back: the rotation is per vector, so this equals rotating the image
/// stream before the concatenation. Host math; batched fallbacks only.</summary>
internal sealed unsafe class LeadingRowsRotation(int rows, IHeadMajorQkRotation inner) : IHeadMajorQkRotation
{
    public void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens)
    {
        if (q.DType != DType.F32 || k.DType != DType.F32 || kvHeads != heads)
            throw new NotSupportedException("The leading-rows rotation takes F32 Q and K with equal head counts.");
        int headDim = (int)q.Shape[q.Shape.Rank - 1];
        TensorShape shape = new(batch, heads, rows, headDim);
        using Tensor qRows = new(shape, DType.F32);
        using Tensor kRows = new(shape, DType.F32);
        Copy(qRows, q, batch, heads, tokens, headDim, toJoint: false);
        Copy(kRows, k, batch, heads, tokens, headDim, toJoint: false);
        inner.RotateHeadMajor(backend, qRows, kRows, batch, heads, kvHeads, rows);
        Copy(qRows, q, batch, heads, tokens, headDim, toJoint: true);
        Copy(kRows, k, batch, heads, tokens, headDim, toJoint: true);
    }

    private void Copy(Tensor part, Tensor joint, int batch, int heads, int tokens, int headDim, bool toJoint)
    {
        float* partPtr = (float*)part.DataPointer;
        float* jointPtr = (float*)joint.DataPointer;
        long bytes = (long)rows * headDim * sizeof(float);
        for (long bh = 0; bh < (long)batch * heads; bh++)
        {
            float* p = partPtr + bh * rows * headDim, j = jointPtr + bh * tokens * headDim;
            if (toJoint) Buffer.MemoryCopy(p, j, bytes, bytes);
            else Buffer.MemoryCopy(j, p, bytes, bytes);
        }
    }
}
