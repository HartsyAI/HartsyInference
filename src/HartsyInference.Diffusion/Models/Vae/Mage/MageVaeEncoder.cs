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
        _patchW = F32(w["patch_cond_embed.weight"]); _patchB = F32(w["patch_cond_embed.bias"]);
        for (int i = 0; i < NumHead; i++) _head[i].LoadWeights(w, $"head_blocks.{i}");
        _projDownW = F32(w["proj_down.weight"]); _projDownB = F32(w["proj_down.bias"]);
        _zProjW = F32(w["z_proj.weight"]); _zProjB = F32(w["z_proj.bias"]);
        _fuseW = F32(w["fuse_proj.weight"]); _fuseB = F32(w["fuse_proj.bias"]);
        _tEmbW1 = F32(w["t_embedder.mlp.0.weight"]); _tEmbB1 = F32(w["t_embedder.mlp.0.bias"]);
        _tEmbW2 = F32(w["t_embedder.mlp.2.weight"]); _tEmbB2 = F32(w["t_embedder.mlp.2.bias"]);
        for (int i = 0; i < NumBlocks; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
        _normOutW = F32(w["norm_out.weight"]); _normOutB = F32(w["norm_out.bias"]);
        _projOutW = F32(w["proj_out.weight"]); _projOutB = F32(w["proj_out.bias"]);
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

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
        Tensor c = TimestepEmbedZero(backend, b);
        for (int i = 0; i < NumBlocks; i++)
        {
            Tensor next = _blocks[i].Forward(backend, s, c);
            s.Dispose(); s = next;
        }
        c.Dispose();

        Tensor normed = new(s.Shape, DType.F32);
        ChannelLayerNormAffine(s, normed, _normOutW!, _normOutB!, b, Hidden, h * w);
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

    private Tensor TimestepEmbedZero(IBackend backend, int b)
    {
        Tensor sinEmb = new(new TensorShape(b, 256), DType.F32);
        float* p = (float*)sinEmb.DataPointer;
        for (int i = 0; i < b; i++) for (int j = 0; j < 256; j++) p[i * 256 + j] = j >= 128 ? 1f : 0f;
        Tensor t1 = new(new TensorShape(b, Hidden), DType.F32);
        backend.Linear(t1, sinEmb, _tEmbW1!, _tEmbB1); sinEmb.Dispose();
        Tensor act = new(t1.Shape, DType.F32);
        backend.Silu(act, t1); t1.Dispose();
        Tensor c = new(new TensorShape(b, Hidden), DType.F32);
        backend.Linear(c, act, _tEmbW2!, _tEmbB2); act.Dispose();
        return c;
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

    private static void ChannelLayerNormAffine(Tensor input, Tensor output, Tensor weight, Tensor bias, int b, int c, int hw)
    {
        float* src = (float*)input.DataPointer; float* dst = (float*)output.DataPointer;
        float* gw = (float*)weight.DataPointer; float* gb = (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int p = 0; p < hw; p++)
            {
                double mean = 0;
                for (int ch = 0; ch < c; ch++) mean += src[((long)(bi * c + ch) * hw) + p];
                mean /= c;
                double var = 0;
                for (int ch = 0; ch < c; ch++) { double d = src[((long)(bi * c + ch) * hw) + p] - mean; var += d * d; }
                float inv = (float)(1.0 / Math.Sqrt(var / c + 1e-6));
                for (int ch = 0; ch < c; ch++)
                {
                    long o = ((long)(bi * c + ch) * hw) + p;
                    dst[o] = (float)((src[o] - mean) * inv) * gw[ch] + gb[ch];
                }
            }
    }

    public void Dispose() { if (Interlocked.Exchange(ref _disposed, 1) != 0) return; }
}
