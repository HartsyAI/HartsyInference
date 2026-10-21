using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>HunyuanImageRope over the image rows (2-axis for an image, 3-axis for video): on the device before the head
/// permute for one batch item, on the host over head-major Q/K for a batch (wrapped in <see cref="LeadingRowsRotation"/>
/// when text rows follow).</summary>
internal sealed class HunyuanImageRotation(HunyuanImageRope rope, int packedT, int packedH, int packedW)
    : IQkRotation, IHeadMajorQkRotation
{
    public void RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads)
    {
        Span<int> dims = packedT > 1 ? stackalloc int[3] { packedT, packedH, packedW } : stackalloc int[2] { packedH, packedW };
        rope.ApplyGpu(backend, q, k, heads, dims);
    }

    public void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens)
    {
        if (packedT > 1)
        {
            Span<int> dims = stackalloc int[3] { packedT, packedH, packedW };
            rope.ApplyJoint(q, k, batch, heads, dims);
        }
        else
        {
            rope.ApplyImage(q, k, batch, heads, packedH, packedW);
        }
    }
}
