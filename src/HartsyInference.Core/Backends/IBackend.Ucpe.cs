using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// UCPE (camera-relative positional encoding) token transform; the host reference is the default.
public partial interface IBackend
{
    /// <summary>The UCPE token transform of token-major <c>[N, heads·headDim]</c> vectors: in each head, every group of four
    /// channels in the first <c>headDim/2</c> is mixed by the token's 4×4 matrix (<c>y = M·x</c>), and the last <c>headDim/2</c>
    /// are rotated as interleaved pairs (<c>(x0, x1) → (x0·c − x1·s, x1·c + x0·s)</c>) with a table that stores each pair's
    /// angle at its even index.</summary>
    /// <param name="output">F32 <c>[N, heads·headDim]</c>, written whole.</param>
    /// <param name="input">F32 <c>[N, heads·headDim]</c>.</param>
    /// <param name="matrices">F32 <c>[N, 4, 4]</c> row-major per-token matrices.</param>
    /// <param name="cos">F32 <c>[N, headDim/2]</c>.</param>
    /// <param name="sin">F32 <c>[N, headDim/2]</c>; negate it for the inverse rotation.</param>
    void UcpeTransform(Tensor output, Tensor input, Tensor matrices, Tensor cos, Tensor sin, int tokens, int heads, int headDim) =>
        UcpeTransformReference.Run(output, input, matrices, cos, sin, tokens, heads, headDim);
}
