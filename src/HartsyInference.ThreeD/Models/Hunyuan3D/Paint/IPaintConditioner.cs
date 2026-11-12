using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Turns the reference image into the DINO context tokens of Paint 2.1's DINO attention.</summary>
public interface IPaintConditioner
{
    /// <summary>Returns the projected tokens <c>[1, tokens, dim]</c> for an RGB image; the caller owns them.</summary>
    Tensor Tokens(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height);

    /// <summary>Returns the tokens of the unconditional branch: upstream zeroes the DINO <i>hidden states</i> and projects them like any others, so the
    /// result is the projection of zeros (a bias and a layer norm), not zeros. The caller owns them.</summary>
    Tensor UnconditionalTokens(IBackend backend);
}
