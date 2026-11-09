using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>UCPE token transform on one batch item's token-major <c>[N, H·D]</c> tensor: each token's 4×4 matrix mixes every group
/// of four channels in the first <c>D/2</c> dims of each head, then RoPE rotates the last <c>D/2</c> dims. The matrix is applied as
/// <c>y = M·x</c> per group, as upstream <c>_apply_ray_projmat</c> does.</summary>
internal static class SanaWmCamUcpe
{
    /// <summary>Returns a new tensor shaped like <paramref name="input"/>; the input is left untouched.</summary>
    /// <param name="matrices">F32 <c>[N, 4, 4]</c> row-major per-token matrices.</param>
    /// <param name="cos">F32 <c>[N, D/2]</c> interleaved duplicated cosines.</param>
    /// <param name="sin">F32 <c>[N, D/2]</c> interleaved duplicated sines; negated for the inverse rotation.</param>
    internal static Tensor Apply(IBackend backend, Tensor input, Tensor matrices, Tensor cos, Tensor sin, int tokens, int heads,
        int headDim)
    {
        int half = headDim / 2;
        int groups = half / 4;
        long channels = (long)heads * headDim;
        using SanaWmScratch scratch = new();
        Tensor output = scratch.Track(SanaWmTensors.Copy(backend, input, input.Shape));
        ReadOnlySpan<long> dims = [tokens, 4, groups, heads];
        ReadOnlySpan<long> strides = [channels, 1, 4, headDim];
        StridedView lanes = new(dims, strides, 0);
        Tensor gathered = scratch.New(tokens, 4, (long)groups * heads);
        backend.CopyStrided(gathered, input, StridedCopySpec.Gather(lanes));
        Tensor mixed = scratch.New(tokens, 4, (long)groups * heads);
        backend.BatchedMatMul(mixed, matrices, gathered);
        scratch.Release(gathered);
        backend.CopyStrided(output, mixed, StridedCopySpec.Scatter(lanes));
        scratch.Release(mixed);

        TensorShape tokenMajor = new(1, tokens, heads, headDim);
        RotarySpec rotation = RotaryViews.Create(RotaryViews.TokenMajor(tokenMajor), RotaryViews.SharedTable(cos.Shape, 1, heads),
            RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair, half, half);
        backend.Rotary(new RotaryCall(output, cos, sin, rotation));
        return scratch.Detach(output);
    }
}
