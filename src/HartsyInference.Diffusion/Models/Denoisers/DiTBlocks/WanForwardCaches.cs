using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Per-generation caches for a Wan-family transformer's step-invariant conditioning work: RoPE cos/sin
/// tables (grid-keyed) and the projected text context (encoder-identity-keyed). Both were recomputed on every CFG
/// forward before — 2×steps host table builds + uploads and 2×steps text-projection GEMMs per generation. The text
/// projection is host-materialized so it survives the pipelines' per-step <c>FreeActivations</c> (the device copy
/// re-uploads on demand; host data stays authoritative). Cached tensors are owned by this class — callers must NOT
/// dispose them per forward; entries are disposed on cap-eviction only (never from a transformer's
/// <c>Dispose</c> — the backing CudaContext may already be torn down there).</summary>
public sealed class WanForwardCaches
{
    private const int Cap = 4;

    private readonly Dictionary<(int T, int H, int W, int Extra), (Tensor Cos, Tensor Sin)> _rope = new();
    private readonly Dictionary<Tensor, Tensor> _textProj = new(ReferenceEqualityComparer.Instance);

    /// <summary>Returns the cached cos/sin pair for <paramref name="key"/>, building it via <paramref name="build"/>
    /// on first use. <c>Extra</c> disambiguates composite tables (e.g. S2V's video⊕reference concat).</summary>
    public (Tensor Cos, Tensor Sin) RopeCosSin((int T, int H, int W, int Extra) key, Func<(Tensor Cos, Tensor Sin)> build)
    {
        if (_rope.TryGetValue(key, out (Tensor Cos, Tensor Sin) hit))
        {
            return hit;
        }
        if (_rope.Count >= Cap)
        {
            foreach ((Tensor cos, Tensor sin) in _rope.Values) { cos.Dispose(); sin.Dispose(); }
            _rope.Clear();
        }
        (Tensor Cos, Tensor Sin) built = build();
        _rope[key] = built;
        return built;
    }

    /// <summary>Returns the (host-materialized) PixArt text projection for <paramref name="encoder"/>, computing it
    /// once per encoder tensor identity.</summary>
    public Tensor TextProj(IBackend backend, Tensor encoder, int dim, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2)
    {
        if (_textProj.TryGetValue(encoder, out Tensor? hit))
        {
            return hit;
        }
        if (_textProj.Count >= Cap)
        {
            foreach (Tensor t in _textProj.Values) t.Dispose();
            _textProj.Clear();
        }
        Tensor proj = WanDitOps.TextEmbed(backend, encoder, dim, w1, b1, w2, b2);
        unsafe { _ = (nint)proj.DataPointer; }   // host-materialize (see class doc)
        _textProj[encoder] = proj;
        return proj;
    }
}
