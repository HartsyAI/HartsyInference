using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Shared Wan-family DiT operations — Conv3d-as-linear patchify/unpatchify, the condition embedder
/// (per-frame-group timestep MLP + 6·dim projection), the PixArt text projection, and the final AdaLN layer. Hoisted
/// from <see cref="WanVideoTransformer"/> so Matrix-Game 3.0 (a Wan2.2 finetune with a memory-augmented sequence)
/// reuses them verbatim instead of duplicating.</summary>
public static unsafe class WanDitOps
{
    /// <summary>Non-overlapping Conv3d patchify as a linear: latent <c>[1, inC, T, H, W]</c> → tokens <c>[S, dim]</c>
    /// with patch-vector order <c>(ic, kt, kh, kw)</c> matching the Conv3d weight layout.</summary>
    public static Tensor Patchify(IBackend backend, Tensor latent, int inC, int innerDim,
        (int T, int H, int W) patch, Tensor patchW2d, Tensor? patchB)
    {
        (int pt, int ph, int pw) = patch;
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int s = gt * gh * gw;
        int patchVec = inC * pt * ph * pw;
        Tensor patches = new Tensor(new TensorShape(s, patchVec), DType.F32);
        float* lp = (float*)latent.DataPointer;
        float* pp = (float*)patches.DataPointer;
        for (int ti = 0; ti < gt; ti++)
            for (int ho = 0; ho < gh; ho++)
                for (int wo = 0; wo < gw; wo++)
                {
                    long token = ((long)ti * gh + ho) * gw + wo;
                    long pBase = token * patchVec;
                    int d = 0;
                    for (int ic = 0; ic < inC; ic++)
                        for (int kt = 0; kt < pt; kt++)
                            for (int kh = 0; kh < ph; kh++)
                                for (int kw = 0; kw < pw; kw++)
                                {
                                    long src = (((long)ic * t + (ti * pt + kt)) * hh + (ho * ph + kh)) * ww + (wo * pw + kw);
                                    pp[pBase + d++] = lp[src];
                                }
                }
        Tensor outT = new Tensor(new TensorShape(s, innerDim), DType.F32);
        backend.Linear(outT, patches, patchW2d, patchB);
        patches.Dispose();
        return outT;
    }

    /// <summary>Inverse of the patchify projection layout: rows <c>[S, oc·pt·ph·pw]</c> → latent <c>[1, oc, T, H, W]</c>.</summary>
    public static Tensor Unpatchify(Tensor proj, int outC, int gt, int gh, int gw, (int T, int H, int W) patch)
    {
        (int pt, int ph, int pw) = patch;
        int patchVec = outC * pt * ph * pw;
        int t = gt * pt, h = gh * ph, w = gw * pw;
        Tensor outT = new Tensor(new TensorShape([1L, outC, t, h, w]), DType.F32);
        float* sp = (float*)proj.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int ti = 0; ti < gt; ti++)
            for (int ho = 0; ho < gh; ho++)
                for (int wo = 0; wo < gw; wo++)
                {
                    long token = ((long)ti * gh + ho) * gw + wo;
                    long pBase = token * patchVec;
                    for (int kt = 0; kt < pt; kt++)
                        for (int kh = 0; kh < ph; kh++)
                            for (int kw = 0; kw < pw; kw++)
                            {
                                int blk = ((kt * ph + kh) * pw + kw) * outC;
                                int f = ti * pt + kt, hy = ho * ph + kh, wx = wo * pw + kw;
                                for (int c = 0; c < outC; c++)
                                {
                                    long dst = (((long)c * t + f) * h + hy) * w + wx;
                                    op[dst] = sp[pBase + blk + c];
                                }
                            }
                }
        return outT;
    }

    /// <summary>Condition embedder over timestep groups: returns (temb <c>[G, dim]</c> for the final AdaLN,
    /// timestep_proj <c>[G, 6, dim]</c> for block modulation) — G=1 scalar path, G=frames for per-frame timesteps
    /// (Wan TI2V I2V, Matrix-Game memory/past/current).</summary>
    public static (Tensor Temb, Tensor TimestepProj) ConditionTimeGroups(IBackend backend, float[] timesteps,
        int freqDim, int dim, Tensor emb1W, Tensor? emb1B, Tensor emb2W, Tensor? emb2B, Tensor projW, Tensor? projB)
    {
        int g = timesteps.Length;
        Tensor temb = new Tensor(new TensorShape(g, dim), DType.F32);
        Tensor proj = new Tensor(new TensorShape([(long)g, 6, dim]), DType.F32);
        for (int gi = 0; gi < g; gi++)
        {
            Tensor sinEmb = new Tensor(new TensorShape(1, freqDim), DType.F32);
            DiTUtils.SinusoidalTimestepEmbedding(sinEmb, timesteps[gi], 1, freqDim, 10000f);
            Tensor e1 = new Tensor(new TensorShape(1, dim), DType.F32); backend.Linear(e1, sinEmb, emb1W, emb1B); sinEmb.Dispose();
            Tensor e1a = new Tensor(e1.Shape, DType.F32); backend.Silu(e1a, e1); e1.Dispose();
            Tensor temb2d = new Tensor(new TensorShape(1, dim), DType.F32); backend.Linear(temb2d, e1a, emb2W, emb2B); e1a.Dispose();
            Tensor sil = new Tensor(new TensorShape(1, dim), DType.F32); backend.Silu(sil, temb2d);
            Tensor projFlat = new Tensor(new TensorShape(1, 6 * dim), DType.F32); backend.Linear(projFlat, sil, projW, projB); sil.Dispose();
            if (g == 1)
            {
                // Device copies keep temb/proj GPU-resident. The old host Buffer.MemoryCopy drained both Linear
                // outputs D2H per forward AND left the results host-side, so every per-block consumer re-uploaded
                // them — one such 2 KB pageable H2D can stall the whole queued stream (the Kandinsky temb bug).
                backend.CopyInto(temb, temb2d);
                backend.CopyInto(proj, projFlat);
            }
            else
            {
                Buffer.MemoryCopy((float*)temb2d.DataPointer, (float*)temb.DataPointer + (long)gi * dim, (long)dim * 4, (long)dim * 4);
                Buffer.MemoryCopy((float*)projFlat.DataPointer, (float*)proj.DataPointer + (long)gi * 6 * dim, (long)6 * dim * 4, (long)6 * dim * 4);
            }
            temb2d.Dispose();
            projFlat.Dispose();
        }
        return (temb, proj);
    }

    /// <summary>PixArtAlphaTextProjection: <c>linear_2(gelu_tanh(linear_1(x)))</c>, text features <c>[L, textDim] → [L, dim]</c>.</summary>
    public static Tensor TextEmbed(IBackend backend, Tensor encoder, int dim, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2)
    {
        // Row count = product of all leading dims (handles both rank-2 [L, hidden] and rank-3 [1, L, hidden] —
        // CfgHelper.SliceBatchElement yields the latter). Using Shape[0] alone read L=1 for the rank-3 case, which
        // undersized the output buffer while the GEMM still produced L rows → buffer overflow → NaN text context.
        int lastDim = (int)encoder.Shape[encoder.Shape.Rank - 1];
        int l = (int)(encoder.Shape.ElementCount / lastDim);
        Tensor h1 = new Tensor(new TensorShape(l, dim), DType.F32); backend.Linear(h1, encoder, w1, b1);
        Tensor act = new Tensor(h1.Shape, DType.F32); backend.Gelu(act, h1); h1.Dispose();
        Tensor outT = new Tensor(new TensorShape(l, dim), DType.F32); backend.Linear(outT, act, w2, b2); act.Dispose();
        return outT;
    }

    /// <summary>Final AdaLN + projection: <c>proj(norm(x)·(1+scale)+shift)</c> with per-group shift/scale from
    /// <c>scale_shift_table[2,dim] + temb[G,dim]</c>. Returns rows <c>[S, outVec]</c>.</summary>
    public static Tensor FinalLayer(IBackend backend, Tensor hidden, Tensor temb, Tensor finalScaleShift,
        Tensor projOutW, Tensor? projOutB, int s, int dim, float eps, int tokensPerGroup)
    {
        int g = (int)temb.Shape[0];
        int outVec = (int)projOutW.Shape[0];
        if (g == 1)
        {
            // Device-resident single-group path (all non-per-frame-timestep Wan variants): the host version below
            // drains the FULL final hidden state D2H, modulates on the CPU, and re-uploads it for proj_out —
            // one full-hidden round-trip per forward.
            Tensor ssShift = new Tensor(new TensorShape(dim), DType.F32);
            backend.SliceRows(ssShift, finalScaleShift, 0);
            Tensor ssScale = new Tensor(new TensorShape(dim), DType.F32);
            backend.SliceRows(ssScale, finalScaleShift, 1);
            Tensor shift1 = new Tensor(new TensorShape(dim), DType.F32);
            backend.Add(shift1, ssShift, temb);
            ssShift.Dispose();
            Tensor scaleSum = new Tensor(new TensorShape(dim), DType.F32);
            backend.Add(scaleSum, ssScale, temb);
            ssScale.Dispose();
            Tensor scale1p = new Tensor(new TensorShape(dim), DType.F32);
            backend.AddScalar(scale1p, scaleSum, 1f);
            scaleSum.Dispose();
            Tensor normedDev = new Tensor(new TensorShape(s, dim), DType.F32);
            backend.LayerNormNoAffine(normedDev, hidden, eps);
            Tensor modded = new Tensor(new TensorShape(s, dim), DType.F32);
            backend.AffineBroadcastLastDim(modded, normedDev, scale1p, shift1);
            normedDev.Dispose(); scale1p.Dispose(); shift1.Dispose();
            Tensor projDev = new Tensor(new TensorShape(s, outVec), DType.F32);
            backend.Linear(projDev, modded, projOutW, projOutB);
            modded.Dispose();
            return projDev;
        }

        float* ss = (float*)finalScaleShift.DataPointer;
        float* em = (float*)temb.DataPointer;
        Tensor shift = new Tensor(new TensorShape(g, dim), DType.F32);
        Tensor scale = new Tensor(new TensorShape(g, dim), DType.F32);
        float* shp = (float*)shift.DataPointer; float* scp = (float*)scale.DataPointer;
        for (int gi = 0; gi < g; gi++)
            for (int d = 0; d < dim; d++)
            {
                shp[gi * dim + d] = ss[d] + em[gi * dim + d];
                scp[gi * dim + d] = ss[dim + d] + em[gi * dim + d];
            }

        Tensor normed = new Tensor(new TensorShape(s, dim), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, 1, s, dim, eps);
        float* np = (float*)normed.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long go = (long)(i / tokensPerGroup) * dim;
            for (int d = 0; d < dim; d++) np[i * dim + d] = np[i * dim + d] * (1f + scp[go + d]) + shp[go + d];
        }
        shift.Dispose(); scale.Dispose();

        Tensor proj = new Tensor(new TensorShape(s, outVec), DType.F32);
        backend.Linear(proj, normed, projOutW, projOutB);
        normed.Dispose();
        return proj;
    }

    /// <summary>Row-concatenates <paramref name="top"/> <c>[a, dim]</c> over <paramref name="bottom"/> <c>[b, dim]</c> → <c>[a+b, dim]</c>.</summary>
    public static Tensor ConcatRows(Tensor top, Tensor bottom, int dim)
    {
        int a = (int)top.Shape[0], b = (int)bottom.Shape[0];
        Tensor o = new Tensor(new TensorShape(a + b, dim), DType.F32);
        Buffer.MemoryCopy((float*)top.DataPointer, (float*)o.DataPointer, (long)a * dim * 4, (long)a * dim * 4);
        Buffer.MemoryCopy((float*)bottom.DataPointer, (float*)o.DataPointer + (long)a * dim, (long)b * dim * 4, (long)b * dim * 4);
        return o;
    }

    /// <summary>Copies a tensor's contiguous data into a fresh F32 <c>[rows, cols]</c> view (Conv3d weight → linear).</summary>
    public static Tensor Reshape2d(Tensor t, int rows, int cols)
    {
        Tensor src = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        Tensor o = new Tensor(new TensorShape(rows, cols), DType.F32);
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)o.DataPointer, (long)rows * cols * 4, (long)rows * cols * 4);
        if (!ReferenceEquals(src, t)) src.Dispose();
        return o;
    }
}
