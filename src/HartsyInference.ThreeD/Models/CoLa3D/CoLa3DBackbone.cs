using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using static HartsyInference.ThreeD.Models.CoLa3D.CoLa3DOps;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>CoLa3D input fusion (latent and Utonia features, each projected and normalized, then concatenated and
/// projected), the stack of blocks, and the final point and query norms.</summary>
internal sealed class CoLa3DBackbone : IDisposable
{
    private readonly Dense _latentProj, _utoniaProj, _fusion;
    private readonly Affine _latentNorm, _utoniaNorm, _pFinalNorm, _qFinalNorm;
    private readonly CoLa3DBlock[] _blocks;
    private readonly int _rotaryDim;
    private int _disposed;

    internal CoLa3DBackbone(IReadOnlyDictionary<string, Tensor> w, CoLa3DConfig config)
    {
        _latentProj = LoadDense(w, "backbone.latentProj");
        _utoniaProj = LoadDense(w, "backbone.utoniaProj");
        _latentNorm = LoadAffine(w, "backbone.latentNorm");
        _utoniaNorm = LoadAffine(w, "backbone.utoniaNorm");
        _fusion = LoadDense(w, "backbone.fusion");
        _pFinalNorm = LoadAffine(w, "backbone.pFinalNorm");
        _qFinalNorm = LoadAffine(w, "backbone.qFinalNorm");
        _rotaryDim = CoLa3DRope.RotaryDim(config.HeadDim);
        _blocks = new CoLa3DBlock[config.NumLayers];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new CoLa3DBlock(w, i, config);
    }

    /// <summary>Point features <c>[1, N, C]</c> from latent <c>[1, N, L]</c> and Utonia <c>[1, N, U]</c>; the inputs are borrowed.</summary>
    internal Tensor FuseInput(IBackend backend, Tensor latent, Tensor utonia)
    {
        int n = (int)latent.Shape[1];
        Tensor lat = Linear(backend, latent, _latentProj, _latentProj.Weight.Shape[0]);
        Tensor latNormed = Norm(backend, lat, _latentNorm, BlockNormEps);
        lat.Dispose();
        Tensor ut = Linear(backend, utonia, _utoniaProj, _utoniaProj.Weight.Shape[0]);
        Tensor utNormed = Norm(backend, ut, _utoniaNorm, BlockNormEps);
        ut.Dispose();
        long width = _latentProj.Weight.Shape[0];
        Tensor cat = Alloc(1, n, 2 * width);
        backend.Concat(cat, [latNormed, utNormed], 2);
        latNormed.Dispose();
        utNormed.Dispose();
        Tensor fused = Linear(backend, cat, _fusion, width);
        cat.Dispose();
        return fused;
    }

    /// <summary>Runs the blocks on one sample with at least one query, then the final norms. Consumes <paramref name="p"/> and <paramref name="q"/>.</summary>
    internal (Tensor P, Tensor Q) Run(IBackend backend, Tensor p, Tensor q, CoLa3DRopeTable pRope, CoLa3DRopeTable qRope)
    {
        foreach (CoLa3DBlock block in _blocks)
            (p, q) = block.Forward(backend, p, q, pRope, qRope, _rotaryDim);

        Tensor pOut = Norm(backend, p, _pFinalNorm, BlockNormEps);
        p.Dispose();
        Tensor qOut = Norm(backend, q, _qFinalNorm, BlockNormEps);
        q.Dispose();
        return (pOut, qOut);
    }

    /// <summary>The final point norm alone, for a sample with no queries.</summary>
    internal Tensor FinalPoints(IBackend backend, Tensor p)
    {
        Tensor pOut = Norm(backend, p, _pFinalNorm, BlockNormEps);
        p.Dispose();
        return pOut;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (CoLa3DBlock block in _blocks)
            block.Dispose();
    }
}
