using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>UCPE token transform on a <c>[B, N, H, D]</c> tensor: each token's 4×4 matrix mixes every group of four
/// channels in the first <c>D/2</c> dims of each head, then RoPE rotates the last <c>D/2</c> dims. The matrix is applied
/// as <c>y = M·x</c> per group, as upstream <c>_apply_ray_projmat</c> does.</summary>
internal static class SanaWmCamUcpe
{
    /// <summary>Returns a new tensor shaped like <paramref name="input"/>; the input is left untouched.</summary>
    /// <param name="matrices">F32 <c>[B·N, 4, 4]</c> row-major per-token matrices.</param>
    /// <param name="cos">F32 <c>[N, D/2]</c> interleaved duplicated cosines.</param>
    /// <param name="sin">F32 <c>[N, D/2]</c> interleaved duplicated sines; negate for the inverse rotation.</param>
    internal static Tensor Apply(IBackend backend, Tensor input, Tensor matrices, Tensor cos, Tensor sin,
        int batch, int tokens, int heads, int headDim)
    {
        int half = headDim / 2;
        int groups = half / 4;
        long channels = (long)heads * headDim;
        Tensor output = new(input.Shape, DType.F32, backend.Device);
        bool built = false;
        try
        {
            backend.CopyTo(output, input);
            ReadOnlySpan<long> dims = [batch, tokens, 4, groups, heads];
            ReadOnlySpan<long> strides = [tokens * channels, channels, 1, 4, headDim];
            StridedView lanes = new(dims, strides, 0);
            using (Tensor gathered = new(new TensorShape(batch * tokens, 4, groups * heads), DType.F32, backend.Device))
            using (Tensor mixed = new(gathered.Shape, DType.F32, backend.Device))
            {
                backend.CopyStrided(gathered, input, StridedCopySpec.Gather(lanes));
                backend.BatchedMatMul(mixed, matrices, gathered);
                backend.CopyStrided(output, mixed, StridedCopySpec.Scatter(lanes));
            }

            Tensor rotated = output.Reshape(new TensorShape(batch, tokens, heads, headDim));
            backend.Rotary(RotaryCall.TokenMajor(rotated, cos, sin, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair, half, half));
            built = true;
        }
        finally
        {
            if (!built)
                output.Dispose();
        }
        return output;
    }
}
