using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Low-level helpers specific to VibeVoice (acoustic + semantic VAEs and the per-token diffusion head), operating on F32 tensors via unsafe <c>float*</c> indexing.</summary>
/// <remarks>Weight tensors are pre-cast to F32 at load time so the pointer arithmetic never has to
/// reason about BF16 / FP16 layouts. The codec/diffusion-head math itself lives on the backend
/// (<see cref="RmsNormChannelsFirstGpu"/>, <see cref="ChannelScaleGpu"/>, and the <c>IBackend</c> conv ops the
/// <c>SConv1d</c>/<c>SConvTranspose1d</c> wrappers call) — what remains here is padding arithmetic and
/// pipeline-level tensor plumbing.</remarks>
internal static unsafe class VibeVoiceOps
{
    /// <summary>Computes the same right-side stride alignment as the Python reference's <c>get_extra_padding_for_conv1d</c>: additional right-side zeros so the (T + padTotal + extra) effective length is exactly <c>(ceil(n_frames) - 1) * stride + kernel - padTotal</c>.</summary>
    public static int GetExtraRightPadding(int tIn, int kernel, int stride, int padTotal)
    {
        // n_frames = (T - K + padTotal) / stride + 1 (using float division)
        float nFrames = ((float)tIn - kernel + padTotal) / stride + 1f;
        int idealLength = ((int)MathF.Ceiling(nFrames) - 1) * stride + (kernel - padTotal);
        int extra = idealLength - tIn;
        return extra < 0 ? 0 : extra;
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
