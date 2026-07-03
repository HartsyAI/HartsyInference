using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Shared sequence-packing / sampling helpers for the Lance image and video pipelines (so they aren't duplicated). Covers the MaPE position build, the logit-normal-shifted timestep grid, the 2-way CFG Euler step, and the channel-last→BCTHW permute for the VAE handoff.</summary>
public static unsafe class LancePipelineCommon
{
    /// <summary>Text (und) modality role tag.</summary>
    public const int TextRole = 0;
    /// <summary>VAE (gen) modality role tag.</summary>
    public const int VaeRole = 1;
    /// <summary>MaPE temporal rebase for gen/noisy tokens (upstream <c>shift_position_ids</c> type-4 → 1000 range). VALIDATION-GATED: confirm exact per-role offsets vs the checkpoint.</summary>
    public const int MapeGenTemporalBase = 1000;

    /// <summary>Builds the packed-sequence positions + role partition: <c>[text(und) | VAE(gen)]</c>. Text tokens get 1-D positions <c>(i,i,i)</c>; VAE tokens get the 3-D grid with the gen temporal axis rebased to <see cref="MapeGenTemporalBase"/>. Returns <c>posIds[seq,3]</c>, the und (text) index list, and the gen (VAE) index list.</summary>
    public static (Tensor Pos, int[] Und, int[] Gen) BuildSequence(int numText, int gridT, int gridH, int gridW)
    {
        int nVae = gridT * gridH * gridW;
        int seq = numText + nVae;
        Tensor pos = new Tensor(new TensorShape(seq, 3), DType.F32);
        float* p = (float*)pos.DataPointer;

        for (int i = 0; i < numText; i++) { p[i * 3 + 0] = i; p[i * 3 + 1] = i; p[i * 3 + 2] = i; }

        int idx = 0;
        for (int ti = 0; ti < gridT; ti++)
            for (int hi = 0; hi < gridH; hi++)
                for (int wi = 0; wi < gridW; wi++)
                {
                    long off = (long)(numText + idx) * 3;
                    p[off + 0] = MapeGenTemporalBase + ti;
                    p[off + 1] = hi;
                    p[off + 2] = wi;
                    idx++;
                }

        int[] und = new int[numText];
        for (int i = 0; i < numText; i++) und[i] = i;
        int[] gen = new int[nVae];
        for (int i = 0; i < nVae; i++) gen[i] = numText + i;
        return (pos, und, gen);
    }

    /// <summary>Logit-normal (SD3-style) shifted timestep grid from 1→0, length steps+1.</summary>
    public static float[] BuildShiftedTimesteps(int steps, float shift)
    {
        float[] t = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float lin = 1.0f - (float)i / steps;
            t[i] = shift * lin / (1.0f + (shift - 1.0f) * lin);
        }
        return t;
    }

    /// <summary>In-place 2-way text-CFG Euler step: <c>z -= (uncond + cfg·(cond − uncond))·dt</c>.</summary>
    public static void EulerCfgStep(Tensor latents, Tensor cond, Tensor uncond, float cfg, float dt)
    {
        long n = latents.Shape.ElementCount;
        float* z = (float*)latents.DataPointer;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float v = u[i] + cfg * (c[i] - u[i]);
            z[i] -= v * dt;
        }
    }

    /// <summary>Folds 2-way CFG into <paramref name="cond"/> in place: <c>cond ← uncond + cfg·(cond − uncond)</c>.
    /// Use this to produce the single combined velocity that a stateful scheduler (e.g.
    /// <c>FlowUniPCMultistepScheduler</c>) steps with, instead of <see cref="EulerCfgStep"/> which combines and steps
    /// in one pass.</summary>
    public static void CfgCombineInPlace(Tensor cond, Tensor uncond, float cfg)
    {
        long n = cond.Shape.ElementCount;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;
        for (long i = 0; i < n; i++) c[i] = u[i] + cfg * (c[i] - u[i]);
    }

    /// <summary>CFG with guidance-renormalization (Lin et al. 2023, "Common Diffusion Noise Schedules…"). Folds
    /// <c>v_cfg = uncond + cfg·(cond − uncond)</c>, then rescales <c>v_cfg</c> so its mean+std match the raw
    /// <b>conditional</b> prediction (which is on-distribution), blended by <paramref name="rescale"/> in [0,1].
    /// Corrects the mean/std inflation that high CFG induces — the DC drift that turns fp8-quantized DiTs' output dark
    /// at cfg≥5. <paramref name="rescale"/>=0 is byte-identical to plain <see cref="CfgCombineInPlace"/> (so fp16
    /// models with the flag off are unchanged); ~0.7 tames the drift while preserving the guidance direction.</summary>
    public static void CfgCombineRenormInPlace(Tensor cond, Tensor uncond, float cfg, float rescale)
    {
        if (rescale <= 0f) { CfgCombineInPlace(cond, uncond, cfg); return; }
        long n = cond.Shape.ElementCount;
        float* c = (float*)cond.DataPointer;
        float* u = (float*)uncond.DataPointer;

        // Stats of the raw conditional prediction (the on-distribution target) BEFORE we overwrite cond.
        double sumC = 0; for (long i = 0; i < n; i++) sumC += c[i];
        double meanC = sumC / n;
        double varC = 0; for (long i = 0; i < n; i++) { double d = c[i] - meanC; varC += d * d; }
        double stdC = Math.Sqrt(varC / n);

        // Fold CFG in place + accumulate its stats.
        double sumCfg = 0; for (long i = 0; i < n; i++) { c[i] = u[i] + cfg * (c[i] - u[i]); sumCfg += c[i]; }
        double meanCfg = sumCfg / n;
        double varCfg = 0; for (long i = 0; i < n; i++) { double d = c[i] - meanCfg; varCfg += d * d; }
        double stdCfg = Math.Sqrt(varCfg / n);

        // Rescale v_cfg to the conditional's mean+std, then blend back by `rescale`.
        float factor = (float)(stdC / Math.Max(stdCfg, 1e-8));
        float mC = (float)meanC, mCfg = (float)meanCfg, phi = rescale;
        for (long i = 0; i < n; i++)
        {
            float matched = (c[i] - mCfg) * factor + mC;
            c[i] = phi * matched + (1f - phi) * c[i];
        }
    }

    /// <summary>Channel-last <c>[T,H,W,C]</c> → <c>[1,C,T,H,W]</c> for the VAE decode handoff.</summary>
    public static Tensor ChannelLastToBcthw(Tensor cl)
    {
        int t = (int)cl.Shape[0], h = (int)cl.Shape[1], w = (int)cl.Shape[2], c = (int)cl.Shape[3];
        Tensor outT = new Tensor(new TensorShape([1L, c, t, h, w]), DType.F32);
        float* s = (float*)cl.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    for (int ci = 0; ci < c; ci++)
                    {
                        long src = (((long)ti * h + hi) * w + wi) * c + ci;
                        long dst = (((long)ci * t + ti) * h + hi) * w + wi;
                        d[dst] = s[src];
                    }
        return outT;
    }

    /// <summary>Inverse of <see cref="ChannelLastToBcthw"/>: <c>[1,C,T,H,W]</c> → channel-last <c>[T,H,W,C]</c>. Used by img2img to feed the VAE-encoded source into <c>LanceLatentPatch.Patchify</c>.</summary>
    public static Tensor BcthwToChannelLast(Tensor bcthw)
    {
        int c = (int)bcthw.Shape[1], t = (int)bcthw.Shape[2], h = (int)bcthw.Shape[3], w = (int)bcthw.Shape[4];
        Tensor outT = new Tensor(new TensorShape([(long)t, h, w, c]), DType.F32);
        float* s = (float*)bcthw.DataPointer;
        float* d = (float*)outT.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    for (int ci = 0; ci < c; ci++)
                    {
                        long src = (((long)ci * t + ti) * h + hi) * w + wi;
                        long dst = (((long)ti * h + hi) * w + wi) * c + ci;
                        d[dst] = s[src];
                    }
        return outT;
    }
}
