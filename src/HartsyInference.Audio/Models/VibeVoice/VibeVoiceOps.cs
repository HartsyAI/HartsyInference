using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Low-level helpers specific to VibeVoice (acoustic + semantic VAEs and the per-token diffusion head), operating on F32 tensors via unsafe <c>float*</c> indexing.</summary>
/// <remarks>Weight tensors are pre-cast to F32 at load time so the pointer arithmetic never has to
/// reason about BF16 / FP16 layouts.
///
/// <para>New ops here, relative to <see cref="Whisper.WhisperOps"/> and
/// <see cref="F5Tts.F5Ops"/>:
/// <list type="bullet">
///   <item><b>CausalConv1d</b> — left-pad-only Conv1D with constant-zero padding,
///         optional dilation, and optional grouping (groups=channels gives depthwise mode
///         needed by every <c>Block1D</c> mixer). Stride and kernel are read from the
///         config; the padding budget per the Python reference is
///         <c>(kernel-1)*dilation - (stride-1)</c>.</item>
///   <item><b>CausalConvTranspose1d</b> — strided transpose conv with right-trim removing
///         the trailing <c>kernel - stride</c> samples (matches <c>trim_right_ratio=1.0</c>
///         in <c>SConvTranspose1d</c>). No grouping — used by the acoustic decoder only.</item>
///   <item><b>RmsNormChannelsFirst</b> — RMSNorm over the channel axis of
///         <c>[B, C, T]</c> with a per-channel learnable scale. <c>IBackend.RmsNorm</c>
///         normalizes the last dim; <c>Block1D</c> works in channels-first, so a dedicated
///         primitive avoids the round-trip transpose-norm-transpose.</item>
///   <item><b>LayerScaleApplyCF</b> — multiplies every <c>[B, c, T]</c> slice by the
///         scalar <c>gamma[c]</c>. The ConvNeXt blocks apply this after the mixer and
///         after the FFN.</item>
///   <item><b>AdaLnZeroDiffusionHead</b> — DiT-style <c>x * (1 + scale) + shift</c> over
///         the latent dim for the per-token diffusion head, where <c>x</c>, <c>shift</c>,
///         <c>scale</c> are all <c>[N, D]</c> (no time axis — the head operates on a
///         single latent frame at a time).</item>
/// </list></para></remarks>
internal static unsafe class VibeVoiceOps
{
    /// <summary>Causal Conv1D with constant zero-pad on the left; output length is <c>ceil((T + extra_right_pad) / stride)</c> when the left pad budget covers the kernel footprint exactly.</summary>
    /// <remarks>
    /// <para>Layout: input <c>[B, C_in, T]</c>, weight <c>[C_out, C_in / groups, K]</c>,
    /// bias <c>[C_out]</c>, output <c>[B, C_out, T_out]</c>.</para>
    ///
    /// <para>Pass <paramref name="groups"/> = <paramref name="cIn"/> for depthwise mode
    /// (<c>C_in == C_out</c>, one weight row per channel). <paramref name="dilation"/> is
    /// always 1 for the published VibeVoice checkpoints but the kernel handles arbitrary
    /// dilation for future variants.</para>
    ///
    /// <para>Out-of-bound left taps read as zero (constant-pad). The caller is responsible
    /// for any right-side stride-alignment padding — pass <paramref name="extraRightPad"/>
    /// computed as <c>get_extra_padding_for_conv1d(T, K, stride, padTotal)</c>.</para></remarks>
    public static void CausalConv1d(
        Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int batch, int cIn, int cOut, int tIn, int kernel, int stride,
        int dilation, int groups, int extraRightPad)
    {
        if (cIn % groups != 0 || cOut % groups != 0)
            throw new ArgumentException($"Channel counts must divide evenly by groups (cIn={cIn}, cOut={cOut}, groups={groups}).");

        int inPerGroup = cIn / groups;
        int outPerGroup = cOut / groups;
        int padTotal = (kernel - 1) * dilation - (stride - 1);
        int tPadded = tIn + padTotal + extraRightPad;
        int tOut = (tPadded - dilation * (kernel - 1) - 1) / stride + 1;

        if (output.Shape.Rank != 3 || (int)output.Shape[0] != batch || (int)output.Shape[1] != cOut || (int)output.Shape[2] != tOut)
            throw new ArgumentException($"CausalConv1d output shape mismatch: expected [{batch}, {cOut}, {tOut}], got {output.Shape}.");

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < cOut; oc++)
            {
                int group = oc / outPerGroup;
                int icStart = group * inPerGroup;
                float biasV = bp is null ? 0f : bp[oc];
                int wRow = oc * inPerGroup * kernel;
                int outBase = (b * cOut + oc) * tOut;

                for (int j = 0; j < tOut; j++)
                {
                    // jPadded is the index in the (logically) left-padded input at which
                    // the leftmost kernel tap reads. Subtracting padTotal gives the index
                    // into the unpadded input; negative values fall in the zero-pad region.
                    int jPadded = j * stride;
                    int srcLeftmost = jPadded - padTotal;
                    float acc = biasV;
                    for (int ic = 0; ic < inPerGroup; ic++)
                    {
                        int inCh = icStart + ic;
                        int inBase = (b * cIn + inCh) * tIn;
                        int wBase = wRow + ic * kernel;
                        for (int k = 0; k < kernel; k++)
                        {
                            int src = srcLeftmost + k * dilation;
                            if ((uint)src < (uint)tIn)
                                acc += ip[inBase + src] * wp[wBase + k];
                        }
                    }
                    op[outBase + j] = acc;
                }
            }
        }
    }

    /// <summary>Computes the same right-side stride alignment as the Python reference's <c>get_extra_padding_for_conv1d</c>: additional right-side zeros so the (T + padTotal + extra) effective length is exactly <c>(ceil(n_frames) - 1) * stride + kernel - padTotal</c>.</summary>
    public static int GetExtraRightPadding(int tIn, int kernel, int stride, int padTotal)
    {
        // n_frames = (T - K + padTotal) / stride + 1 (using float division)
        float nFrames = ((float)tIn - kernel + padTotal) / stride + 1f;
        int idealLength = ((int)MathF.Ceiling(nFrames) - 1) * stride + (kernel - padTotal);
        int extra = idealLength - tIn;
        return extra < 0 ? 0 : extra;
    }

    /// <summary>Causal ConvTranspose1D with right-trim, mirroring the math of <c>SConvTranspose1d</c>.</summary>
    /// <remarks>Input <c>[B, C_in, T_in]</c> → strided transposed conv
    /// produces <c>[B, C_out, (T_in-1)*stride + K]</c>, then we trim
    /// <c>padTotal = K - stride</c> samples from the right (because
    /// <c>trim_right_ratio = 1.0</c> sends all the asymmetric pad budget to the right).
    ///
    /// <para>Weight layout follows PyTorch's <c>nn.ConvTranspose1d</c>:
    /// <c>[C_in, C_out, K]</c>. Bias <c>[C_out]</c>. Output <c>[B, C_out, T_in*stride]</c>
    /// after trim.</para>
    ///
    /// <para>No groups support — the VibeVoice decoder is always fully-connected per
    /// stage. No dilation — always 1.</para></remarks>
    public static void CausalConvTranspose1d(
        Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int batch, int cIn, int cOut, int tIn, int kernel, int stride)
    {
        int tFull = (tIn - 1) * stride + kernel;     // raw transposed-conv output length
        int padTotal = kernel - stride;
        int tOut = tFull - padTotal;                 // after right-trim with trim_right_ratio=1.0

        if (output.Shape.Rank != 3 || (int)output.Shape[0] != batch || (int)output.Shape[1] != cOut || (int)output.Shape[2] != tOut)
            throw new ArgumentException($"CausalConvTranspose1d output shape mismatch: expected [{batch}, {cOut}, {tOut}], got {output.Shape}.");

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;

        // Initialize output with bias (broadcast across time).
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < cOut; oc++)
            {
                float biasV = bp is null ? 0f : bp[oc];
                int outBase = (b * cOut + oc) * tOut;
                for (int j = 0; j < tOut; j++) op[outBase + j] = biasV;
            }
        }

        // Accumulate input-driven contributions. For each input position i, the kernel
        // writes to output positions [i*stride + k] for k in [0, K). Skip positions that
        // fall in the trimmed right-pad region (j >= tOut).
        for (int b = 0; b < batch; b++)
        {
            for (int ic = 0; ic < cIn; ic++)
            {
                int inBase = (b * cIn + ic) * tIn;
                for (int oc = 0; oc < cOut; oc++)
                {
                    int wBase = (ic * cOut + oc) * kernel;
                    int outBase = (b * cOut + oc) * tOut;
                    for (int i = 0; i < tIn; i++)
                    {
                        float xv = ip[inBase + i];
                        int outRow = i * stride;
                        for (int k = 0; k < kernel; k++)
                        {
                            int j = outRow + k;
                            if (j < tOut) op[outBase + j] += xv * wp[wBase + k];
                        }
                    }
                }
            }
        }
    }

    /// <summary>GPU-resident channels-first RMSNorm over a <c>[B, C, T]</c> tensor with a per-channel scale <paramref name="weight"/> (<c>[C]</c>).</summary>
    /// <remarks>Composed from existing backend ops (transpose → last-dim <see cref="IBackend.RmsNorm"/>
    /// → transpose) so the whole VAE block stays on-device — avoiding the per-op device→host
    /// sync the host <see cref="RmsNormChannelsFirst"/> forced. <see cref="IBackend.RmsNorm"/>
    /// uses <c>1/sqrt(meanSq + eps)</c>, matching the host reference bit-for-bit.</remarks>
    /// <returns>A fresh <c>[B, C, T]</c> tensor; caller disposes.</returns>
    public static Tensor RmsNormChannelsFirstGpu(IBackend backend, Tensor x, Tensor weight, int batch, int channels, int t, float eps)
    {
        Tensor cl = new(new TensorShape(batch, t, channels), DType.F32);
        backend.Transpose2D(cl, x, channels, t);
        Tensor normed = new(cl.Shape, DType.F32);
        backend.RmsNorm(normed, cl, weight, eps);
        cl.Dispose();
        Tensor cf = new(new TensorShape(batch, channels, t), DType.F32);
        backend.Transpose2D(cf, normed, t, channels);
        normed.Dispose();
        return cf;
    }

    /// <summary>GPU-resident per-channel scale of a channels-first <c>[B, C, T]</c> tensor by <paramref name="scaleWeight"/> shaped as a <c>[C, 1, 1]</c> depthwise conv kernel: <c>out[b,c,t] = x[b,c,t] · scaleWeight[c]</c>, computed via a groups=C, kernel=1 <see cref="IBackend.Conv1d"/> (ConvNeXt "layer scale") to avoid a host round-trip.</summary>
    /// <returns>A fresh tensor; caller disposes.</returns>
    public static Tensor ChannelScaleGpu(IBackend backend, Tensor x, Tensor scaleWeight, int batch, int channels, int t)
    {
        Tensor output = new(new TensorShape(batch, channels, t), DType.F32);
        backend.Conv1d(output, x, scaleWeight, null, 1, 0, 0, 1, channels);
        return output;
    }

    /// <summary>Copies a per-channel scale vector (<c>[C]</c>, any layout) into a fresh owning <c>[C, 1, 1]</c> depthwise-conv kernel for <see cref="ChannelScaleGpu"/>.</summary>
    public static unsafe Tensor ToChannelScaleWeight(Tensor gamma, int channels)
    {
        using Tensor g = gamma.DType == DType.F32 ? null! : gamma.CastTo(DType.F32);
        float* src = (float*)(g is null ? gamma.DataPointer : g.DataPointer);
        Tensor w = new(new TensorShape(channels, 1, 1), DType.F32);
        float* dst = (float*)w.DataPointer;
        for (int c = 0; c < channels; c++) dst[c] = src[c];
        return w;
    }

    /// <summary>RMSNorm over the channel axis of a <c>[B, C, T]</c> tensor with a per-channel learnable scale <c>weight</c> (shape <c>[C]</c>); host-side per-(batch, time) RMS over C in F32.</summary>
    /// <remarks>This matches Python's <c>ConvRMSNorm</c> which transposes to channels-last,
    /// runs RMSNorm on the channel dim, then transposes back. We collapse that into a
    /// single pass to save bandwidth in the hot codec loop.</remarks>
    public static void RmsNormChannelsFirst(Tensor output, Tensor input, Tensor weight, int batch, int channels, int t, float eps)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int tt = 0; tt < t; tt++)
            {
                // Mean of squares across channels at this (b, tt).
                double sumSq = 0d;
                for (int c = 0; c < channels; c++)
                {
                    float v = ip[(b * channels + c) * t + tt];
                    sumSq += (double)v * v;
                }
                float invRms = 1f / MathF.Sqrt((float)(sumSq / channels) + eps);

                for (int c = 0; c < channels; c++)
                {
                    int idx = (b * channels + c) * t + tt;
                    op[idx] = ip[idx] * invRms * wp[c];
                }
            }
        }
    }

    /// <summary>Multiplies a channels-first <c>[B, C, T]</c> tensor by a per-channel learnable scale <c>gamma</c> of shape <c>[C]</c>, in place; ConvNeXt's "layer scale" applied after the mixer and after the FFN inside <c>Block1D</c>.</summary>
    public static void LayerScaleApplyCF(Tensor x, Tensor gamma, int batch, int channels, int t)
    {
        float* xp = (float*)x.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float scale = gp[c];
                int rowBase = (b * channels + c) * t;
                for (int j = 0; j < t; j++) xp[rowBase + j] *= scale;
            }
        }
    }

    /// <summary>Diffusion-head AdaLN modulation: <c>x = x * (1 + scale) + shift</c>, with <c>x</c>, <paramref name="scale"/> and <paramref name="shift"/> all shape <c>[N, D]</c> (the head denoises one latent frame per LM step, so the time axis collapses to 1).</summary>
    public static void AdaLnModulate(Tensor x, Tensor shift, Tensor scale, int n, int d)
    {
        float* xp = (float*)x.DataPointer;
        float* sp = (float*)shift.DataPointer;
        float* zp = (float*)scale.DataPointer;
        for (int i = 0; i < n; i++)
        {
            int rowBase = i * d;
            for (int k = 0; k < d; k++)
            {
                xp[rowBase + k] = xp[rowBase + k] * (1f + zp[rowBase + k]) + sp[rowBase + k];
            }
        }
    }

    /// <summary>Diffusion-head gated residual add: <c>out = residual + gate * h</c>, all shape <c>[N, D]</c>; safe when <paramref name="output"/> aliases <paramref name="residual"/>.</summary>
    public static void AdaLnGatedAdd(Tensor output, Tensor residual, Tensor gate, Tensor h, int n, int d)
    {
        float* op = (float*)output.DataPointer;
        float* rp = (float*)residual.DataPointer;
        float* gp = (float*)gate.DataPointer;
        float* hp = (float*)h.DataPointer;
        for (int i = 0; i < n; i++)
        {
            int rowBase = i * d;
            for (int k = 0; k < d; k++)
            {
                op[rowBase + k] = rp[rowBase + k] + gp[rowBase + k] * hp[rowBase + k];
            }
        }
    }

    // ── Pipeline-level tensor plumbing ──────────────────────────────────────
    // Shared between VibeVoicePipeline (1.5B/7B) and VibeVoiceStreamingPipeline (Realtime-0.5B) — both run
    // an AR loop over per-frame acoustic-VAE latents with CFG-batched DDPM denoising.

    /// <summary>[B, T, dim] → [B, 1, dim] of the last T position.</summary>
    public static Tensor SliceLastFrame(Tensor hidden, int dim)
    {
        int batch = (int)hidden.Shape[0];
        int t = (int)hidden.Shape[1];
        Tensor result = new(new TensorShape(batch, 1, dim), DType.F32);
        float* sp = (float*)hidden.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long srcOff = ((long)b * t + (t - 1)) * dim;
            long dstOff = (long)b * dim;
            Buffer.MemoryCopy(sp + srcOff, dp + dstOff, dim * 4, dim * 4);
        }
        return result;
    }

    /// <summary>Ensures shape <c>[1, 1, dim]</c>; clones if already so.</summary>
    public static Tensor ExpandTo3D(Tensor x, int dim)
    {
        if (x.Shape.Rank == 3 && (int)x.Shape[1] == 1) return CopyOf(x);
        if (x.Shape.Rank == 2)
        {
            Tensor t = new(new TensorShape(1, 1, dim), DType.F32);
            Buffer.MemoryCopy((void*)x.DataPointer, (void*)t.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
            return t;
        }
        return CopyOf(x);
    }

    public static Tensor CopyOf(Tensor src)
    {
        Tensor copy = new(src.Shape, DType.F32);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)copy.DataPointer, src.ElementCount * 4, src.ElementCount * 4);
        return copy;
    }

    /// <summary>Draws fresh decorrelated Gaussian noise from one persistent stream (matching <c>torch.randn</c> per frame) — never reseed per frame, that correlates adjacent latents.</summary>
    public static Tensor SampleNoise(int dim, ref uint rng)
    {
        Tensor t = new(new TensorShape(1, 1, dim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < dim; i++)
            p[i] = HartsyInference.Audio.Dsp.DeterministicRng.NextGaussian(ref rng);
        return t;
    }

    public static Tensor AddEmbeds(Tensor a, Tensor? b)
    {
        Tensor result = CopyOf(a);
        if (b is null) return result;
        long n = result.ElementCount;
        float* rp = (float*)result.DataPointer;
        float* bp = (float*)b.DataPointer;
        for (long i = 0; i < n; i++) rp[i] += bp[i];
        return result;
    }

    public static void NormalizeLatentInPlace(Tensor latent, float scale, float bias)
    {
        long n = latent.ElementCount;
        float* p = (float*)latent.DataPointer;
        for (long i = 0; i < n; i++) p[i] = (p[i] + bias) * scale;
    }

    public static void UnnormalizeLatentInPlace(Tensor latent, float scale, float bias)
    {
        long n = latent.ElementCount;
        float* p = (float*)latent.DataPointer;
        float invScale = 1f / scale;
        for (long i = 0; i < n; i++) p[i] = p[i] * invScale - bias;
    }

    public static Tensor ReNormalizeLatent(Tensor latent, float scale, float bias)
    {
        Tensor copy = CopyOf(latent);
        long n = copy.ElementCount;
        float* p = (float*)copy.DataPointer;
        for (long i = 0; i < n; i++) p[i] = (p[i] + bias) * scale;
        return copy;
    }

    public static float[] TensorToPcm(Tensor frame)
    {
        // [1, 1, T] → float[T]
        int t = (int)frame.Shape[2];
        float[] result = new float[t];
        float* p = (float*)frame.DataPointer;
        fixed (float* dst = result) Buffer.MemoryCopy(p, dst, t * 4, t * 4);
        return result;
    }

    public static Tensor PcmToTensor(float[] pcm)
    {
        Tensor t = new(new TensorShape(1, 1, pcm.Length), DType.F32);
        fixed (float* src = pcm) Buffer.MemoryCopy(src, (void*)t.DataPointer, pcm.Length * 4, pcm.Length * 4);
        return t;
    }

    /// <summary>CFG combine from a batched <c>[1, 2, latent]</c> head output (row 0 = cond, row 1 = uncond): <c>eps = uncond + cfg_scale * (cond - uncond)</c>. Latents are tiny (usually 64) so the tail stays host.</summary>
    public static Tensor CombineCfgBatched(Tensor vBatched, float cfg, int latent)
    {
        Tensor result = new(new TensorShape(1, 1, latent), DType.F32);
        float* v = (float*)vBatched.DataPointer;
        float* r = (float*)result.DataPointer;
        for (int i = 0; i < latent; i++) r[i] = v[latent + i] + cfg * (v[i] - v[latent + i]);
        return result;
    }

    public static float ReadScalar(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        if (!w.TryGetValue(key, out Tensor? t)) return 1.0f;
        if (t.DType == DType.F32) return *(float*)t.DataPointer;
        using Tensor f = t.CastTo(DType.F32);     // checkpoint stores these as bf16 scalars
        return *(float*)f.DataPointer;
    }

    /// <summary>Runs the CFG-batched DDPM denoise loop shared by every VibeVoice variant: batches the conditional/unconditional halves into one N=2 head forward per step (the head is FFN-only, no cross-frame mixing, so each row is exactly its own pass), combining as <c>eps = uncond + cfg_scale·(cond−uncond)</c>.</summary>
    /// <param name="negCond">Null, or <paramref name="cfgScale"/> == 1, runs single-stream (no CFG).</param>
    /// <returns>An owned tensor the caller disposes.</returns>
    public static Tensor DenoiseLatent(IBackend backend, VibeVoiceDiffusionHead diffusionHead, int numSteps,
        float cfgScale, Tensor noiseLatent, Tensor cond, Tensor? negCond, int latentDim, int hiddenDim)
    {
        using VibeVoiceCosineDpmSolver scheduler = new();
        scheduler.SetTimesteps(numSteps);

        bool useCfg = negCond is not null && MathF.Abs(cfgScale - 1f) > 1e-6f;
        Tensor speech = noiseLatent;
        bool ownsSpeech = false;
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        Tensor? condB = null;
        if (useCfg)
        {
            condB = new(new TensorShape(1, 2, hiddenDim), DType.F32);
            backend.Concat(condB, [cond, negCond!], dim: 1);
        }
        for (int step = 0; step < timesteps.Length; step++)
        {
            Tensor vPred;
            if (useCfg)
            {
                Tensor speechB = new(new TensorShape(1, 2, latentDim), DType.F32);
                backend.Concat(speechB, [speech, speech], dim: 1);
                float[] tBatch2 = [timesteps[step], timesteps[step]];
                Tensor vB = diffusionHead.Forward(backend, speechB, tBatch2, condB!);
                vPred = CombineCfgBatched(vB, cfgScale, latentDim);
                speechB.Dispose();
                vB.Dispose();
            }
            else
            {
                float[] tBatch = [timesteps[step]];
                vPred = diffusionHead.Forward(backend, speech, tBatch, cond);
            }
            Tensor next = scheduler.Step(vPred, speech, step);
            vPred.Dispose();
            if (ownsSpeech) speech.Dispose();
            speech = next;
            ownsSpeech = true;
        }
        condB?.Dispose();

        if (!ownsSpeech)
        {
            return CopyOf(speech);      // defensive: no steps ran somehow
        }
        return speech;
    }
}
