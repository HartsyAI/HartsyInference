using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>ACE-Step v1.5 guidance + sampler-side correction, ported verbatim from upstream
/// <c>models/common/apg_guidance.py</c> and <c>dcw_correction.py</c>/<c>dcw_primitives.py</c>. All tensors are
/// the pipeline's <c>[1, T, C]</c> latents/velocities (C = 64); reductions run in double like upstream's
/// <c>project()</c>. APG norms/projections use dims=[1] (per-channel over TIME) — NOT whole-tensor.</summary>
public static unsafe class AceStep15Guidance
{
    /// <summary>APG's running momentum of the cond−uncond diff (<c>MomentumBuffer</c>, momentum −0.75):
    /// <c>avg = diff + m·avg</c>. One instance per generation; updates only on guided steps.</summary>
    public sealed class MomentumBuffer(float momentum = -0.75f)
    {
        private float[]? _avg;
        public void Update(float[] diff)
        {
            if (_avg is null)
            {
                _avg = (float[])diff.Clone();
                return;
            }
            for (int i = 0; i < diff.Length; i++) _avg[i] = diff[i] + momentum * _avg[i];
        }
        public float[]? RunningAverage => _avg;
    }

    /// <summary>APG (<c>apg_forward</c>, dims=[1]): momentum-accumulated diff, per-channel time-axis norm clamp
    /// (threshold 2.5), projection of the diff onto/against the cond velocity (time axis, double precision), and
    /// <c>guided = cond + (g−1)·orthogonal</c> (eta = 0 upstream default drops the parallel term).</summary>
    public static Tensor ApgForward(Tensor cond, Tensor uncond, float guidanceScale, MomentumBuffer momentum,
        float normThreshold = 2.5f)
    {
        int t = (int)cond.Shape[1], c = (int)cond.Shape[2];
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        float[] diff = new float[t * c];
        for (int i = 0; i < diff.Length; i++) diff[i] = cp[i] - up[i];
        momentum.Update(diff);
        diff = (float[])momentum.RunningAverage!.Clone();

        // Per-channel norm over time (dims=[1], keepdim): clamp to normThreshold.
        if (normThreshold > 0)
        {
            for (int ch = 0; ch < c; ch++)
            {
                double ss = 0;
                for (int ti = 0; ti < t; ti++) { double d = diff[ti * c + ch]; ss += d * d; }
                double norm = Math.Sqrt(ss);
                if (norm > normThreshold)
                {
                    float f = (float)(normThreshold / norm);
                    for (int ti = 0; ti < t; ti++) diff[ti * c + ch] *= f;
                }
            }
        }

        // project(diff, cond, dims=[1]) in double: v1 = cond normalized over time per channel;
        // parallel = (Σ_t diff·v1)·v1; orthogonal = diff − parallel.
        Tensor outT = new(cond.Shape, DType.F32);
        float* op = (float*)outT.DataPointer;
        double g1 = guidanceScale - 1.0;
        for (int ch = 0; ch < c; ch++)
        {
            double ss = 0;
            for (int ti = 0; ti < t; ti++) { double v = cp[ti * c + ch]; ss += v * v; }
            double inv = 1.0 / Math.Max(Math.Sqrt(ss), 1e-12);
            double dot = 0;
            for (int ti = 0; ti < t; ti++) dot += diff[ti * c + ch] * (cp[ti * c + ch] * inv);
            for (int ti = 0; ti < t; ti++)
            {
                double v1 = cp[ti * c + ch] * inv;
                double orth = diff[ti * c + ch] - dot * v1;
                op[ti * c + ch] = (float)(cp[ti * c + ch] + g1 * orth);
            }
        }
        return outT;
    }

    /// <summary>Plain CFG blend (<c>cfg_forward</c>) — used by Heun correctors upstream.</summary>
    public static Tensor CfgForward(Tensor cond, Tensor uncond, float g)
    {
        Tensor outT = new(cond.Shape, DType.F32);
        float* op = (float*)outT.DataPointer;
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        for (long i = 0; i < cond.Shape.ElementCount; i++) op[i] = up[i] + g * (cp[i] - up[i]);
        return outT;
    }

    /// <summary>ADG (<c>adg_forward</c>, flow-matching form): per-row (time-frame) angle between the cond/uncond
    /// clean estimates <c>x̂ = x − t·v</c>, angle scaled by <c>max(g−1,0)+1e-3</c> and clipped to ±π/6, rotated
    /// clean estimate re-expressed as a velocity <c>(x − x̂_new)/t</c>.</summary>
    public static Tensor AdgForward(Tensor latents, Tensor cond, Tensor uncond, float sigma, float guidanceScale,
        float angleClip = 3.14f / 6f)
    {
        int t = (int)cond.Shape[1], c = (int)cond.Shape[2];
        float* xp = (float*)latents.DataPointer;
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        double w = Math.Max(guidanceScale - 1.0, 0.0) + 1e-3;
        Tensor outT = new(cond.Shape, DType.F32);
        float* op = (float*)outT.DataPointer;

        if (c > 512) throw new ArgumentException($"ADG expects the 64-ch latent (got {c} channels).");
        // Clean estimates per frame (double, matching upstream's .to(float) on f32 — full precision here).
        Span<double> hatC = stackalloc double[c];
        Span<double> hatU = stackalloc double[c];
        for (int ti = 0; ti < t; ti++)
        {
            long off = (long)ti * c;
            double nc = 0, nu = 0, dotCU = 0;
            for (int ch = 0; ch < c; ch++)
            {
                double hc = xp[off + ch] - sigma * cp[off + ch];
                double hu = xp[off + ch] - sigma * up[off + ch];
                hatC[ch] = hc; hatU[ch] = hu;
                nc += hc * hc; nu += hu * hu; dotCU += hc * hu;
            }
            double cos = dotCU / (Math.Sqrt(nc) * Math.Sqrt(nu) + 1e-30);
            cos = Math.Clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6);
            double theta = Math.Acos(cos);
            double thetaNew = Math.Clamp(w * theta, -angleClip, angleClip);

            // proj/perp of (x̂_c − x̂_u) against x̂_u per frame.
            double dotDU = 0, nuSq = nu;
            for (int ch = 0; ch < c; ch++) dotDU += (hatC[ch] - hatU[ch]) * hatU[ch];
            double projK = dotDU / (nuSq + 1e-8);
            double sinT = Math.Sin(theta);
            double sinNew = Math.Sin(thetaNew);
            double cosNew = Math.Cos(thetaNew);
            double perpScale = sinT > 1e-3 ? sinNew / sinT : w;
            for (int ch = 0; ch < c; ch++)
            {
                double perp = (hatC[ch] - hatU[ch]) - projK * hatU[ch];
                double latentNew = cosNew * hatC[ch] + perp * perpScale;
                op[off + ch] = (float)((xp[off + ch] - latentNew) / sigma);
            }
        }
        return outT;
    }

    /// <summary>DCW — Differential Correction in Wavelet domain (<c>DCWCorrector</c>): after each sampler step,
    /// pushes the latent's frequency band(s) away from the predicted clean sample via a 1-level Haar DWT along
    /// time. Scalers are t-modulated: low band <c>t·scaler</c>, high band <c>(1−t)·highScaler</c>.</summary>
    public sealed class DcwCorrector(bool enabled, string mode = "double", float scaler = 0.05f, float highScaler = 0.02f)
    {
        public bool IsActive => enabled && (mode == "double" ? scaler != 0f || highScaler != 0f : scaler != 0f);

        /// <summary>Applies the correction in place on <paramref name="xNext"/> <c>[1, T, C]</c>.</summary>
        public void Apply(Tensor xNext, Tensor denoised, float tCurr)
        {
            if (!IsActive) return;
            float lowS = tCurr * scaler;
            float highS = (1f - tCurr) * scaler;
            float doubleHighS = (1f - tCurr) * highScaler;
            switch (mode)
            {
                case "pix":
                {
                    float* xp = (float*)xNext.DataPointer;
                    float* yp = (float*)denoised.DataPointer;
                    for (long i = 0; i < xNext.Shape.ElementCount; i++) xp[i] += scaler * (xp[i] - yp[i]);
                    return;
                }
                case "low": HaarCorrect(xNext, denoised, lowS, 0f); return;
                case "high": HaarCorrect(xNext, denoised, 0f, highS); return;
                default: HaarCorrect(xNext, denoised, lowS, doubleHighS); return;   // "double"
            }
        }

        // 1-level Haar along T per channel: L[k]=(x2k+x2k+1)/√2, H[k]=(x2k−x2k+1)/√2 (zero-pad odd T);
        // corrected bands xB += s·(xB − yB); inverse transform back, trimmed to T.
        private static void HaarCorrect(Tensor x, Tensor y, float lowS, float highS)
        {
            int t = (int)x.Shape[1], c = (int)x.Shape[2];
            int half = (t + 1) / 2;
            float* xp = (float*)x.DataPointer;
            float* yp = (float*)y.DataPointer;
            const double R = 0.70710678118654752;   // 1/√2
            Span<double> xl = new double[half];
            Span<double> xh = new double[half];
            for (int ch = 0; ch < c; ch++)
            {
                for (int k = 0; k < half; k++)
                {
                    double xa = xp[(long)(2 * k) * c + ch];
                    double xb = 2 * k + 1 < t ? xp[(long)(2 * k + 1) * c + ch] : 0.0;
                    double ya = yp[(long)(2 * k) * c + ch];
                    double yb = 2 * k + 1 < t ? yp[(long)(2 * k + 1) * c + ch] : 0.0;
                    double xL = (xa + xb) * R, xH = (xa - xb) * R;
                    double yL = (ya + yb) * R, yH = (ya - yb) * R;
                    xl[k] = xL + lowS * (xL - yL);
                    xh[k] = xH + highS * (xH - yH);
                }
                for (int k = 0; k < half; k++)
                {
                    double a = (xl[k] + xh[k]) * R, b = (xl[k] - xh[k]) * R;
                    xp[(long)(2 * k) * c + ch] = (float)a;
                    if (2 * k + 1 < t) xp[(long)(2 * k + 1) * c + ch] = (float)b;
                }
            }
        }
    }
}
