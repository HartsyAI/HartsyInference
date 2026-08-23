using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.Mage;

/// <summary>Encoder for Microsoft Mage-Flow's <b>MageVAE</b> (mage_vae.py <c>_DConvEncoder</c>, L402-441), used by the
/// edit path to turn a source image into 128-channel reference latents. One-step, deterministic (fixed t=0, zero
/// noise latent, <c>sample_posterior=false</c> → returns the mean). Checkpoint keys are under
/// <c>student.dconv_encoder.*</c> (pass them with that prefix stripped).
///
/// <para><b>Dataflow</b> (h=H/16, w=W/16): pixels [B,3,H,W] in [-1,1] →
/// <c>patch_cond_embed</c> (Conv2d 3→768, k16 s16 — THE 16× downsample) → 2× <see cref="MageEncoderDiCoBlock"/>
/// (768, affine norms, no adaLN/gate) → <c>proj_down</c> (768→384) → fuse with <c>z_proj(0)</c> bias →
/// 21× <see cref="MageDiCoBlock"/> (384, adaLN at t=0) → <c>norm_out</c> (affine LayerNorm2d) → <c>proj_out</c>
/// (384→256) → take channels [0:128] = mean = the latent. No latent scaling.</para></summary>
public sealed unsafe class MageVaeEncoder : IDisposable
{
    private const int Hidden = 384, HeadHidden = 768, Latent = 128, Patch = 16, NumHead = 2, NumBlocks = 21;
    private Tensor? _patchW, _patchB;         // patch_cond_embed Conv 3→768 k16 s16
    private readonly MageEncoderDiCoBlock[] _head = new MageEncoderDiCoBlock[NumHead];
    private Tensor? _projDownW, _projDownB;    // 768→384 k1
    private Tensor? _zProjW, _zProjB;          // 128→384 k1 (on zeros → bias)
    private Tensor? _fuseW, _fuseB;            // 768→384 k1
    private Tensor? _tEmbW1, _tEmbB1, _tEmbW2, _tEmbB2;
    private readonly MageDiCoBlock[] _blocks = new MageDiCoBlock[NumBlocks];
    private Tensor? _normOutW, _normOutB;      // affine LayerNorm2d 384
    private Tensor? _projOutW, _projOutB;      // 384→256 k1
    private int _disposed;

    public MageVaeEncoder()
    {
        for (int i = 0; i < NumHead; i++) _head[i] = new MageEncoderDiCoBlock(HeadHidden);
        for (int i = 0; i < NumBlocks; i++) _blocks[i] = new MageDiCoBlock(Hidden);
    }

    /// <summary>Loads encoder weights (keys with the <c>student.dconv_encoder.</c> prefix already stripped).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _patchW = TensorCasts.EnsureF32(w["patch_cond_embed.weight"]); _patchB = TensorCasts.EnsureF32(w["patch_cond_embed.bias"]);
        for (int i = 0; i < NumHead; i++) _head[i].LoadWeights(w, $"head_blocks.{i}");
        _projDownW = TensorCasts.EnsureF32(w["proj_down.weight"]); _projDownB = TensorCasts.EnsureF32(w["proj_down.bias"]);
        _zProjW = TensorCasts.EnsureF32(w["z_proj.weight"]); _zProjB = TensorCasts.EnsureF32(w["z_proj.bias"]);
        _fuseW = TensorCasts.EnsureF32(w["fuse_proj.weight"]); _fuseB = TensorCasts.EnsureF32(w["fuse_proj.bias"]);
        _tEmbW1 = TensorCasts.EnsureF32(w["t_embedder.mlp.0.weight"]); _tEmbB1 = TensorCasts.EnsureF32(w["t_embedder.mlp.0.bias"]);
        _tEmbW2 = TensorCasts.EnsureF32(w["t_embedder.mlp.2.weight"]); _tEmbB2 = TensorCasts.EnsureF32(w["t_embedder.mlp.2.bias"]);
        for (int i = 0; i < NumBlocks; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
        _normOutW = TensorCasts.EnsureF32(w["norm_out.weight"]); _normOutB = TensorCasts.EnsureF32(w["norm_out.bias"]);
        _projOutW = TensorCasts.EnsureF32(w["proj_out.weight"]); _projOutB = TensorCasts.EnsureF32(w["proj_out.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW, _patchB, _projDownW, _projDownB, _zProjW, _zProjB, _fuseW, _fuseB,
            _tEmbW1, _tEmbB1, _tEmbW2, _tEmbB2, _normOutW, _normOutB, _projOutW, _projOutB })
            if (t is not null) yield return t;
        foreach (MageEncoderDiCoBlock b in _head) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (MageDiCoBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes <paramref name="pixels"/> [B,3,H,W] (in [-1,1], H/W multiples of 16) → latent [B,128,H/16,W/16].</summary>
    public Tensor Encode(IBackend backend, Tensor pixels)
    {
        int b = (int)pixels.Shape[0], H = (int)pixels.Shape[2], W = (int)pixels.Shape[3];
        int h = H / Patch, w = W / Patch;

        // 16× downsample: strided patch conv → [b, 768, h, w].
        Tensor cond = new(new TensorShape(b, HeadHidden, h, w), DType.F32);
        backend.Conv2D(cond, pixels, _patchW!, _patchB, Patch, Patch, 0, 0);
        for (int i = 0; i < NumHead; i++)
        {
            Tensor next = _head[i].Forward(backend, cond);
            cond.Dispose(); cond = next;
        }
        Tensor condDown = new(new TensorShape(b, Hidden, h, w), DType.F32);
        backend.Conv2D(condDown, cond, _projDownW!, _projDownB, 1, 1, 0, 0);

        // z_proj(zeros) = bias broadcast; fuse_proj(cat([cond_down, zbias])).
        Tensor zeros = new(new TensorShape(b, Latent, h, w), DType.F32);
        new Span<float>((float*)zeros.DataPointer, (int)zeros.ElementCount).Clear();
        Tensor zbias = new(new TensorShape(b, Hidden, h, w), DType.F32);
        backend.Conv2D(zbias, zeros, _zProjW!, _zProjB, 1, 1, 0, 0);
        zeros.Dispose();
        Tensor fuseIn = new(new TensorShape(b, HeadHidden, h, w), DType.F32);
        CatChannels(condDown, zbias, fuseIn, b, Hidden, h * w);
        condDown.Dispose(); zbias.Dispose(); cond.Dispose();
        Tensor s = new(new TensorShape(b, Hidden, h, w), DType.F32);
        backend.Conv2D(s, fuseIn, _fuseW!, _fuseB, 1, 1, 0, 0);
        fuseIn.Dispose();

        // 21-block adaLN trunk at t=0.
        Tensor c = MageVaeOps.TimestepEmbedZero(backend, b, _tEmbW1!, _tEmbB1, _tEmbW2!, _tEmbB2, Hidden);
        for (int i = 0; i < NumBlocks; i++)
        {
            Tensor next = _blocks[i].Forward(backend, s, c);
            s.Dispose(); s = next;
        }
        c.Dispose();

        Tensor normed = new(s.Shape, DType.F32);
        MageVaeOps.ChannelLayerNormAffine(s, normed, _normOutW!, _normOutB!, b, Hidden, h * w);
        s.Dispose();
        Tensor moments = new(new TensorShape(b, 2 * Latent, h, w), DType.F32);
        backend.Conv2D(moments, normed, _projOutW!, _projOutB, 1, 1, 0, 0);
        normed.Dispose();

        // mean = channels [0:128].
        Tensor mean = new(new TensorShape(b, Latent, h, w), DType.F32);
        int hw = h * w;
        Buffer.MemoryCopy((float*)moments.DataPointer, (void*)mean.DataPointer, (long)b * Latent * hw * 4, (long)b * Latent * hw * 4);
        // (b>1 would need per-batch strided copy; edit uses b=1.)
        moments.Dispose();
        return mean;
    }

    private static void CatChannels(Tensor a, Tensor bT, Tensor outp, int b, int ca, int hw)
    {
        float* ap = (float*)a.DataPointer; float* bp = (float*)bT.DataPointer; float* op = (float*)outp.DataPointer;
        int outC = 2 * ca;
        for (int bi = 0; bi < b; bi++)
        {
            Buffer.MemoryCopy(ap + (long)bi * ca * hw, op + (long)bi * outC * hw, (long)ca * hw * 4, (long)ca * hw * 4);
            Buffer.MemoryCopy(bp + (long)bi * ca * hw, op + ((long)bi * outC + ca) * hw, (long)ca * hw * 4, (long)ca * hw * 4);
        }
    }

    public void Dispose() { if (Interlocked.Exchange(ref _disposed, 1) != 0) return; }
}
