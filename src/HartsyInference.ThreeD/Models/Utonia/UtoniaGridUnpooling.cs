using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Decoder unpooling: returns the finer parent with its features replaced by <c>skip(parent) + up(coarse)[inverse]</c>,
/// where both projections are Linear, LayerNorm and GELU.
/// </summary>
internal sealed class UtoniaGridUnpooling
{
    private readonly IBackend _backend;
    private readonly UtoniaLinear _up;
    private readonly UtoniaLayerNorm _upNorm;
    private readonly UtoniaLinear _skip;
    private readonly UtoniaLayerNorm _skipNorm;

    public UtoniaGridUnpooling(IBackend backend, UtoniaLinear up, UtoniaLayerNorm upNorm, UtoniaLinear skip, UtoniaLayerNorm skipNorm)
    {
        _backend = backend;
        _up = up;
        _upNorm = upNorm;
        _skip = skip;
        _skipNorm = skipNorm;
    }

    /// <summary>Restores the finer level behind <paramref name="coarse"/>. The returned point is the parent object.</summary>
    public UtoniaPoint Forward(UtoniaPoint coarse)
    {
        UtoniaPoint parent = coarse.PoolingParent ?? throw new HartsyInferenceException("Utonia unpooling needs a pooling parent.");
        int[] inverse = coarse.PoolingInverse ?? throw new HartsyInferenceException("Utonia unpooling needs a pooling inverse.");
        Tensor skip = Project(_skip, _skipNorm, parent.Feat);
        Tensor upCoarse = Project(_up, _upNorm, coarse.Feat);
        Tensor gathered = UtoniaTensors.Gather(_backend, upCoarse, inverse);
        upCoarse.Dispose();
        Tensor sum = UtoniaTensors.Add(_backend, skip, gathered);
        skip.Dispose();
        gathered.Dispose();
        parent.ReplaceFeat(sum);
        return parent;
    }

    private Tensor Project(UtoniaLinear linear, UtoniaLayerNorm norm, Tensor input)
    {
        Tensor linearOut = linear.Forward(input);
        Tensor normed = norm.Forward(linearOut);
        linearOut.Dispose();
        Tensor activated = new(normed.Shape, DType.F32);
        _backend.GeluErf(activated, normed);
        normed.Dispose();
        return activated;
    }
}
