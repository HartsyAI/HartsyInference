using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Shared helpers for classifier-free guidance pipelines that run a UNet/transformer twice per step (uncond + cond) and combine the two outputs. Centralizes the batch-slice + CFG-combine operations duplicated across SD1.5, SDXL, SD3, and every legacy CFG pipeline.</summary>
public static unsafe class CfgHelper
{
    /// <summary>Extracts a single element along the batch dimension of a 3-D tensor [B, seqLen, hiddenSize] → [1, seqLen, hiddenSize]. F32 only — callers needing other dtypes should cast first.</summary>
    public static Tensor SliceBatchElement(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(1, seqLen, hiddenSize);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int elements = seqLen * hiddenSize;
        int srcOffset = batchIdx * elements;
        for (int i = 0; i < elements; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }
        return slice;
    }

    /// <summary>Extracts the first <paramref name="realLen"/> rows of batch element <paramref name="batchIdx"/> from a
    /// right-padded <c>[B, fullSeqLen, hiddenSize]</c> tensor → <c>[1, realLen, hiddenSize]</c>. Drops the pad rows so
    /// downstream cross-attention sees only real tokens (equivalent to masking the pad positions, which text-conditioned
    /// DiTs require — attending unmasked padding dilutes the caption). F32 only.</summary>
    public static Tensor SliceBatchElementPrefix(Tensor tensor, int batchIdx, int fullSeqLen, int realLen, int hiddenSize)
    {
        if (realLen <= 0 || realLen > fullSeqLen)
            throw new ArgumentException($"realLen {realLen} must be in (0, {fullSeqLen}].", nameof(realLen));
        Tensor slice = new Tensor(new TensorShape(1, realLen, hiddenSize), DType.F32);
        float* src = (float*)tensor.DataPointer + (long)batchIdx * fullSeqLen * hiddenSize;
        float* dst = (float*)slice.DataPointer;
        long n = (long)realLen * hiddenSize;
        for (long i = 0; i < n; i++) dst[i] = src[i];
        return slice;
    }

    /// <summary>Extracts a single element along the batch dimension of a 2-D tensor [B, dim] → [1, dim]. F32 only.</summary>
    public static Tensor SliceBatchElement1D(Tensor tensor, int batchIdx, int dim)
    {
        TensorShape shape = new TensorShape(1, dim);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int srcOffset = batchIdx * dim;
        for (int i = 0; i < dim; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }
        return slice;
    }

    /// <summary>Whether classifier-free guidance has any effect at this scale. At <c>scale ≤ 1</c> the combine reduces to the conditional output (<c>uncond + 1·(cond − uncond) = cond</c>), so the unconditional forward pass is wasted work. Guidance-distilled checkpoints (Flux-dev, SDXL-Turbo, LCM/TCD) run at scale 0-1; gating the uncond pass on this halves per-step compute and activation memory for them. A small epsilon avoids running a pass that contributes nothing at scale ≈ 1.</summary>
    public static bool IsGuidanceActive(float scale) => scale > 1.0f + 1e-4f;

    /// <summary>Applies classifier-free guidance: <c>output = uncond + scale * (cond - uncond)</c> (the standard convention, anchored on the unconditional output). Both inputs must be F32 and have identical shape; output is allocated F32 with that same shape. The inputs are NOT disposed — caller owns the lifetime.</summary>
    public static Tensor ApplyCfg(Tensor uncond, Tensor cond, float scale) => CombineCfg(uncond, cond, scale, condAnchored: false);

    /// <summary>Applies cond-anchored classifier-free guidance: <c>output = cond + scale * (cond - uncond)</c>. This is algebraically <see cref="ApplyCfg"/> with the anchor swapped to the conditional output, the convention used by Krea 2 and Chroma. Because the baseline is <c>cond</c> rather than <c>uncond</c>, a given guidance_scale produces stronger guidance than the standard convention: e.g. Krea 2's <c>guidance_scale = 4.5</c> here ≈ <c>5.5</c> under <see cref="ApplyCfg"/> (the cond-anchored formula adds one extra unit of <c>(cond - uncond)</c> relative to the uncond-anchored one). Both inputs must be F32 and have identical shape; output is allocated F32 with that same shape. The inputs are NOT disposed — caller owns the lifetime.</summary>
    // VALIDATION-PENDING: verify cond-anchored formula vs Krea 2 / Chroma reference pipelines (cond + scale*(cond - uncond)).
    public static Tensor ApplyCfgCondAnchored(Tensor cond, Tensor uncond, float scale) => CombineCfg(uncond, cond, scale, condAnchored: true);

    /// <summary>Core CFG combine shared by <see cref="ApplyCfg"/> and <see cref="ApplyCfgCondAnchored"/>. When <paramref name="condAnchored"/> is false: <c>uncond + scale*(cond - uncond)</c>; when true: <c>cond + scale*(cond - uncond)</c>.</summary>
    private static Tensor CombineCfg(Tensor uncond, Tensor cond, float scale, bool condAnchored)
    {
        if (uncond.DType != DType.F32 || cond.DType != DType.F32)
            throw new ArgumentException($"ApplyCfg requires F32 inputs; got uncond={uncond.DType}, cond={cond.DType}. Cast via DtypeCastHelper.EnsureF32 first.");
        if (!uncond.Shape.Equals(cond.Shape))
            throw new ArgumentException($"ApplyCfg shape mismatch: uncond={uncond.Shape}, cond={cond.Shape}.");
        Tensor output = new Tensor(uncond.Shape, DType.F32);
        float* uncPtr = (float*)uncond.DataPointer;
        float* conPtr = (float*)cond.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)uncond.ElementCount;
        for (int i = 0; i < count; i++)
        {
            float anchor = condAnchored ? conPtr[i] : uncPtr[i];
            outPtr[i] = anchor + scale * (conPtr[i] - uncPtr[i]);
        }
        return output;
    }

    /// <summary>Applies classifier-free guidance with Lumina-2.0's <c>cfg_normalization</c>: combine <c>guided = uncond + scale·(cond − uncond)</c>, then rescale the guided prediction back to the conditional's per-token L2 norm over the last dimension — <c>guided ·= ‖cond‖₂ / (‖guided‖₂ + eps)</c>, where each norm reduces over the final axis and is broadcast across it.
    /// <para>VALIDATION-PENDING: verify against diffusers <c>Lumina2Pipeline</c> (<c>pipeline_lumina2.py</c> lines 750-755): <c>cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)</c>; <c>noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)</c>; <c>noise_pred = noise_pred * (cond_norm / noise_norm)</c>. Diffusers uses no epsilon; the <paramref name="eps"/> here guards the division and is negligible at the default 1e-12. The reduction axis is the last tensor dim — the W axis for the unpatchified <c>[B, C, H, W]</c> velocity.</para>
    /// Both inputs must be F32 with identical shape; output is a new F32 tensor with that shape. Inputs are NOT disposed — caller owns the lifetime.</summary>
    public static Tensor ApplyCfgNormalized(Tensor uncond, Tensor cond, float scale, float eps = 1e-12f)
    {
        if (uncond.DType != DType.F32 || cond.DType != DType.F32)
            throw new ArgumentException($"ApplyCfgNormalized requires F32 inputs; got uncond={uncond.DType}, cond={cond.DType}. Cast via DtypeCastHelper.EnsureF32 first.");
        if (!uncond.Shape.Equals(cond.Shape))
            throw new ArgumentException($"ApplyCfgNormalized shape mismatch: uncond={uncond.Shape}, cond={cond.Shape}.");

        Tensor output = new Tensor(uncond.Shape, DType.F32);
        float* uncPtr = (float*)uncond.DataPointer;
        float* conPtr = (float*)cond.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // The last dimension is the per-token feature axis the L2 norm reduces over; everything before it is the token grid.
        int lastDim = (int)uncond.Shape[uncond.Shape.Rank - 1];
        long tokens = uncond.ElementCount / lastDim;

        for (long tok = 0; tok < tokens; tok++)
        {
            long baseOffset = tok * lastDim;

            // Pass 1: combine guidance and accumulate ‖cond‖₂ and ‖guided‖₂ over this token's last-dim slice.
            double condSq = 0.0;
            double guidedSq = 0.0;
            for (int d = 0; d < lastDim; d++)
            {
                long idx = baseOffset + d;
                float unc = uncPtr[idx];
                float con = conPtr[idx];
                float guided = unc + scale * (con - unc);
                outPtr[idx] = guided;
                condSq += (double)con * con;
                guidedSq += (double)guided * guided;
            }

            // Pass 2: rescale guided back to the conditional's L2 norm, broadcast across the last dim.
            float ratio = (float)(Math.Sqrt(condSq) / (Math.Sqrt(guidedSq) + eps));
            for (int d = 0; d < lastDim; d++)
                outPtr[baseOffset + d] *= ratio;
        }
        return output;
    }

    /// <summary>CFG-Rescale (Hartsy's "hartsyinference"-flagged param — see <c>docs/07-Parameters-And-Feature-Flags.md</c>
    /// on the extension side for why this duplicates rather than reads a Comfy-side param): rescales an already
    /// CFG-combined <paramref name="guided"/> prediction back toward the conditional prediction's per-token L2
    /// dispersion, blended by <paramref name="rescaleWeight"/>. High CFG scales inflate the guided prediction's
    /// magnitude relative to the conditional's, which is what drives the oversaturation/burnt-highlights look —
    /// this pulls it back down. At <paramref name="rescaleWeight"/> = 0 this is a no-op (returns a copy of
    /// <paramref name="guided"/>); at 1.0 it's a full renormalize to <c>‖cond‖</c> (this is the same per-token
    /// last-dim L2 reduction <see cref="ApplyCfgNormalized"/> uses for Lumina2's <c>cfg_normalization</c>, kept as
    /// a separate method rather than refactored into this one to avoid risking that pipeline's already-validated
    /// numerics). Not the same operation as ComfyUI's RescaleCFG node (which reduces std over all non-batch dims,
    /// not L2 over the last dim per token) — this is Hartsy's own formulation, values are not directly comparable.
    /// Both inputs must be F32 with identical shape; output is a new F32 tensor. Inputs are NOT disposed — caller
    /// owns the lifetime.</summary>
    public static Tensor ApplyCfgRescale(Tensor guided, Tensor cond, float rescaleWeight, float eps = 1e-12f)
    {
        if (guided.DType != DType.F32 || cond.DType != DType.F32)
            throw new ArgumentException($"ApplyCfgRescale requires F32 inputs; got guided={guided.DType}, cond={cond.DType}. Cast via DtypeCastHelper.EnsureF32 first.");
        if (!guided.Shape.Equals(cond.Shape))
            throw new ArgumentException($"ApplyCfgRescale shape mismatch: guided={guided.Shape}, cond={cond.Shape}.");

        Tensor output = new Tensor(guided.Shape, DType.F32);
        float* guiPtr = (float*)guided.DataPointer;
        float* conPtr = (float*)cond.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        if (rescaleWeight <= 0f)
        {
            long total = guided.ElementCount;
            for (long i = 0; i < total; i++) outPtr[i] = guiPtr[i];
            return output;
        }

        int lastDim = (int)guided.Shape[guided.Shape.Rank - 1];
        long tokens = guided.ElementCount / lastDim;

        for (long tok = 0; tok < tokens; tok++)
        {
            long baseOffset = tok * lastDim;
            double condSq = 0.0;
            double guidedSq = 0.0;
            for (int d = 0; d < lastDim; d++)
            {
                long idx = baseOffset + d;
                condSq += (double)conPtr[idx] * conPtr[idx];
                guidedSq += (double)guiPtr[idx] * guiPtr[idx];
            }
            float ratio = (float)(Math.Sqrt(condSq) / (Math.Sqrt(guidedSq) + eps));
            for (int d = 0; d < lastDim; d++)
            {
                long idx = baseOffset + d;
                float rescaled = guiPtr[idx] * ratio;
                outPtr[idx] = rescaleWeight * rescaled + (1f - rescaleWeight) * guiPtr[idx];
            }
        }
        return output;
    }

    /// <summary>Dual (text + image) classifier-free guidance for reference-image edit pipelines:
    /// <c>output = uncond + imageScale·(imageCond − uncond) + textScale·(cond − imageCond)</c>, elementwise.
    /// This is the OmniGen2 triple-pass combine (<c>pipeline_omnigen2.py</c>:
    /// <c>model_pred_uncond + image_guidance_scale·(model_pred_ref − model_pred_uncond) +
    /// text_guidance_scale·(model_pred − model_pred_ref)</c>, where <c>model_pred</c> = positive text + refs,
    /// <c>model_pred_ref</c> = negative text + refs, <c>model_pred_uncond</c> = negative text, no refs).
    /// Boogu-Image's double guidance <c>cond + (tg−1)·(cond − dropText) + (ig−1)·(dropText − dropAll)</c> expands to
    /// the same expression (<c>tg·cond + (ig−tg)·mid + (1−ig)·uncond</c>) with <c>mid = dropText</c>,
    /// <c>uncond = dropAll</c>, so both pipelines share this combine. All inputs must be F32 with identical shape;
    /// output is a new F32 tensor. Inputs are NOT disposed — caller owns the lifetime.</summary>
    public static Tensor ApplyDualCfg(Tensor cond, Tensor imageCond, Tensor uncond, float textScale, float imageScale)
    {
        if (cond.DType != DType.F32 || imageCond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new ArgumentException(
                $"ApplyDualCfg requires F32 inputs; got cond={cond.DType}, imageCond={imageCond.DType}, uncond={uncond.DType}.");
        if (!cond.Shape.Equals(imageCond.Shape) || !cond.Shape.Equals(uncond.Shape))
            throw new ArgumentException(
                $"ApplyDualCfg shape mismatch: cond={cond.Shape}, imageCond={imageCond.Shape}, uncond={uncond.Shape}.");
        Tensor output = new Tensor(cond.Shape, DType.F32);
        float* c = (float*)cond.DataPointer;
        float* m = (float*)imageCond.DataPointer;
        float* u = (float*)uncond.DataPointer;
        float* o = (float*)output.DataPointer;
        long n = cond.ElementCount;
        for (long i = 0; i < n; i++)
            o[i] = u[i] + imageScale * (m[i] - u[i]) + textScale * (c[i] - m[i]);
        return output;
    }

    /// <summary>TCFG (Tangential Damping Classifier-free Guidance, https://huggingface.co/papers/2503.18137 —
    /// Hartsy's "hartsyinference"-flagged param, see the same duplication-vs-Comfy-flag rationale documented on
    /// <see cref="ApplyCfgRescale"/>): before the standard CFG combine, drops the component of the unconditional
    /// prediction that is "tangential" to (orthogonal from, in the row-space sense below) the conditional
    /// prediction, then combines as usual: <c>output = uncondFiltered + scale·(cond − uncondFiltered)</c>.
    /// <para>Reference impl (diffusers <c>TangentialClassifierFreeGuidance</c>) stacks <c>[cond, uncond]</c> into a
    /// per-batch-item 2×N matrix (N = flattened non-batch size), SVDs it, zeroes the second right-singular vector,
    /// and reprojects <c>uncond</c> through the remaining one. For a 2-row matrix this SVD has a closed form: the
    /// right singular vectors come from eigendecomposing the 2×2 Gram matrix <c>G = [[cond·cond, cond·uncond],
    /// [cond·uncond, uncond·uncond]]</c>, and reprojecting through only the dominant one collapses algebraically to
    /// <c>uncondFiltered = alpha·cond + beta·uncond</c> for two per-batch-item scalars — so this is implemented as
    /// two dot-product reduction passes plus a scalar 2×2 eigensolve, not a general SVD routine. Verified
    /// bit-exact (max abs err ~1e-14) against <c>numpy.linalg.svd</c> on random vectors before this was trusted —
    /// see <c>SdxlTcfgRealWeightTests</c>'s class doc for the derivation-check summary and the real-checkpoint
    /// calibration finding (TCFG is a genuine trajectory steer, not a subtle nudge — large pixel diff at real
    /// generation isn't itself a correctness red flag the way it is for step-cache).</para>
    /// Both inputs must be F32 with identical shape; output is a new F32 tensor with that shape. Inputs are NOT
    /// disposed — caller owns the lifetime. Composes with <see cref="ApplyCfgRescale"/> the same way the reference
    /// does (TCFG combine first, rescale applied to its output) — callers wanting both should call this, then feed
    /// the result into <see cref="ApplyCfgRescale"/>.</summary>
    public static Tensor ApplyTcfg(Tensor cond, Tensor uncond, float scale, float eps = 1e-8f)
    {
        if (uncond.DType != DType.F32 || cond.DType != DType.F32)
            throw new ArgumentException($"ApplyTcfg requires F32 inputs; got cond={cond.DType}, uncond={uncond.DType}. Cast via DtypeCastHelper.EnsureF32 first.");
        if (!uncond.Shape.Equals(cond.Shape))
            throw new ArgumentException($"ApplyTcfg shape mismatch: cond={cond.Shape}, uncond={uncond.Shape}.");

        Tensor output = new Tensor(uncond.Shape, DType.F32);
        float* conPtr = (float*)cond.DataPointer;
        float* uncPtr = (float*)uncond.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        int batch = (int)uncond.Shape[0];
        long perBatch = uncond.ElementCount / batch;

        for (int b = 0; b < batch; b++)
        {
            long baseOffset = (long)b * perBatch;

            // Gram matrix of the 2-row [cond; uncond] matrix over this batch item's flattened elements.
            double a = 0.0, g = 0.0, d = 0.0;
            for (long i = 0; i < perBatch; i++)
            {
                long idx = baseOffset + i;
                double c = conPtr[idx];
                double u = uncPtr[idx];
                a += c * c;
                g += c * u;
                d += u * u;
            }

            // Closed-form eigendecomposition of the symmetric 2x2 Gram matrix [[a, g], [g, d]]; (u0, u1) is the
            // unit eigenvector for the larger eigenvalue lambda0 (the dominant right-singular direction).
            double trace = a + d;
            double disc = Math.Sqrt(Math.Max(0.0, (trace * 0.5) * (trace * 0.5) - (a * d - g * g)));
            double lambda0 = trace * 0.5 + disc;
            double u0, u1;
            if (Math.Abs(g) > 1e-20)
            {
                u0 = g;
                u1 = lambda0 - a;
            }
            else
            {
                u0 = a >= d ? 1.0 : 0.0;
                u1 = a >= d ? 0.0 : 1.0;
            }
            double norm = Math.Sqrt(u0 * u0 + u1 * u1);
            if (norm > 1e-20) { u0 /= norm; u1 /= norm; } else { u0 = 1.0; u1 = 0.0; }

            // uncondFiltered = alpha*cond + beta*uncond — the algebraic collapse of "project uncond onto the
            // dominant singular direction, drop the rest" (see doc comment above for the derivation).
            double alpha, beta;
            if (lambda0 > eps)
            {
                double k = (u0 * g + u1 * d) / lambda0;
                alpha = u0 * k;
                beta = u1 * k;
            }
            else
            {
                // Degenerate (near-zero-energy) batch item — fall back to unfiltered uncond rather than divide
                // by ~0; real diffusion predictions never hit this in practice.
                alpha = 0.0;
                beta = 1.0;
            }

            for (long i = 0; i < perBatch; i++)
            {
                long idx = baseOffset + i;
                double c = conPtr[idx];
                double u = uncPtr[idx];
                double uncondFiltered = alpha * c + beta * u;
                outPtr[idx] = (float)(uncondFiltered + scale * (c - uncondFiltered));
            }
        }
        return output;
    }

    /// <summary>Concatenates two [B, seqLen, dimA] and [B, seqLen, dimB] tensors along the last dimension → [B, seqLen, dimA + dimB]. Used by SDXL-family pipelines to stitch CLIP-L and CLIP-G hidden states into the 2048-wide text embedding the UNet expects. F32 only.</summary>
    public static Tensor ConcatLastDim(Tensor a, Tensor b)
    {
        if (a.Shape.Rank != 3 || b.Shape.Rank != 3)
            throw new ArgumentException($"ConcatLastDim expects rank-3 tensors; got {a.Shape.Rank} and {b.Shape.Rank}.");
        if (a.Shape[0] != b.Shape[0] || a.Shape[1] != b.Shape[1])
            throw new ArgumentException($"ConcatLastDim requires matching [batch, seqLen]; got {a.Shape} vs {b.Shape}.");
        int batch = (int)a.Shape[0];
        int seqLen = (int)a.Shape[1];
        int dimA = (int)a.Shape[2];
        int dimB = (int)b.Shape[2];
        int dimOut = dimA + dimB;
        Tensor output = new Tensor(new TensorShape(batch, seqLen, dimOut), DType.F32);
        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int aOffset = (bIdx * seqLen + s) * dimA;
                int bOffset = (bIdx * seqLen + s) * dimB;
                int outOffset = (bIdx * seqLen + s) * dimOut;
                for (int d = 0; d < dimA; d++) outPtr[outOffset + d] = aPtr[aOffset + d];
                for (int d = 0; d < dimB; d++) outPtr[outOffset + dimA + d] = bPtr[bOffset + d];
            }
        }
        return output;
    }
}
