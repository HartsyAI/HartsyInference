using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-Video timestep-conditioned VAE decode embedder — diffusers <c>PixArtAlphaCombinedTimestepSizeEmbeddings</c>
/// (size-emb disabled): <c>get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0)</c> →
/// <c>linear_1</c> → SiLU → <c>linear_2</c> → <c>[outDim]</c>. Each mid/up block carries one (<c>outDim = 4·C</c>) feeding all
/// its resnets; the decoder norm_out carries one (<c>outDim = 2·C</c>). Weights live under <c>{prefix}.timestep_embedder.linear_1/2.*</c>.
///
/// <para>VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — sinusoidal flavor (flip_sin_to_cos, shift 0) and
/// the SiLU-between-linears activation.</para></summary>
public sealed unsafe class LtxVaeTimeEmbedder
{
    private readonly int _outDim;
    private readonly Tensor _l1W, _l2W;
    private readonly Tensor? _l1B, _l2B;

    private LtxVaeTimeEmbedder(int outDim, Tensor l1W, Tensor? l1B, Tensor l2W, Tensor? l2B)
    {
        _outDim = outDim;
        _l1W = l1W;
        _l1B = l1B;
        _l2W = l2W;
        _l2B = l2B;
    }

    public static LtxVaeTimeEmbedder Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int outDim)
    {
        Tensor l1W = TensorCasts.LoadF32(w, $"{prefix}.timestep_embedder.linear_1.weight");
        Tensor l2W = TensorCasts.LoadF32(w, $"{prefix}.timestep_embedder.linear_2.weight");
        w.TryGetValue($"{prefix}.timestep_embedder.linear_1.bias", out Tensor? l1B);
        w.TryGetValue($"{prefix}.timestep_embedder.linear_2.bias", out Tensor? l2B);
        return new LtxVaeTimeEmbedder(outDim, l1W, l1B is null ? null : TensorCasts.EnsureF32(l1B), l2W, l2B is null ? null : TensorCasts.EnsureF32(l2B));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _l1W;
        if (_l1B is not null) yield return _l1B;
        yield return _l2W;
        if (_l2B is not null) yield return _l2B;
    }

    /// <summary>Embeds a single (already 1000×-scaled) timestep into <c>[outDim]</c> (B=1).</summary>
    public Tensor Embed(IBackend backend, float timestep)
    {
        Tensor sin = new Tensor(new TensorShape(1, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, timestep, 1, 256, 10000f);
        int hidden = (int)_l1W.Shape[0];
        Tensor e1 = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.Linear(e1, sin, _l1W, _l1B); sin.Dispose();
        backend.Silu(e1, e1);
        Tensor out2d = new Tensor(new TensorShape(1, _outDim), DType.F32);
        backend.Linear(out2d, e1, _l2W, _l2B); e1.Dispose();
        // Flatten to [outDim] for the per-block modulation consumers.
        Tensor outT = new Tensor(new TensorShape(_outDim), DType.F32);
        Buffer.MemoryCopy((float*)out2d.DataPointer, (float*)outT.DataPointer, (long)_outDim * 4, (long)_outDim * 4);
        out2d.Dispose();
        return outT;
    }
}
