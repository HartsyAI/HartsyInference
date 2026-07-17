using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Lance latent↔token handoff between the Wan2.2 VAE latent and the transformer (upstream einops <c>(t pt)(h ph)(w pw) c → (t h w)(pt ph pw c)</c> with <c>latent_patch_size=(1,2,2)</c>). Operates on the <b>channel-last</b> latent <c>[Tf, Hf, Wf, C]</c> form the upstream uses (the pipeline permutes the VAE's <c>[B,C,T,H,W]</c> into/out of this). Token feature dim = <c>pt·ph·pw·C</c> (= 1·2·2·48 = 192 for Lance).</summary>
public static unsafe class LanceLatentPatch
{
    /// <summary>Patchify: channel-last latent <c>[t·pt, h·ph, w·pw, C]</c> → tokens <c>[t·h·w, pt·ph·pw·C]</c>. Token order is <c>(t,h,w)</c> row-major; feature order is <c>(pt,ph,pw,c)</c>.</summary>
    public static Tensor Patchify(Tensor latentCl, int pt, int ph, int pw)
    {
        int tf = (int)latentCl.Shape[0], hf = (int)latentCl.Shape[1], wf = (int)latentCl.Shape[2], c = (int)latentCl.Shape[3];
        if (tf % pt != 0 || hf % ph != 0 || wf % pw != 0)
            throw new ArgumentException($"latent {tf}x{hf}x{wf} not divisible by patch ({pt},{ph},{pw}).");
        int t = tf / pt, h = hf / ph, w = wf / pw;
        int featDim = pt * ph * pw * c;
        int numTokens = t * h * w;

        Tensor tokens = new Tensor(new TensorShape(numTokens, featDim), DType.F32);
        float* src = (float*)latentCl.DataPointer;
        float* dst = (float*)tokens.DataPointer;

        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                {
                    long token = ((long)ti * h + hi) * w + wi;
                    for (int pti = 0; pti < pt; pti++)
                        for (int phi = 0; phi < ph; phi++)
                            for (int pwi = 0; pwi < pw; pwi++)
                                for (int ci = 0; ci < c; ci++)
                                {
                                    int feat = ((pti * ph + phi) * pw + pwi) * c + ci;
                                    long srcOff = (((long)(ti * pt + pti) * hf + (hi * ph + phi)) * wf + (wi * pw + pwi)) * c + ci;
                                    dst[token * featDim + feat] = src[srcOff];
                                }
                }
        // latentCl is JIT-dead once DataPointer has been read: if the caller passed an unrooted temporary,
        // a GC during the copy loop finalizes it and frees `src` mid-read (same class as Tensor.Reshape's fix).
        GC.KeepAlive(latentCl);
        return tokens;
    }

    /// <summary>Unpatchify (inverse of <see cref="Patchify"/>): tokens <c>[t·h·w, pt·ph·pw·C]</c> → channel-last latent <c>[t·pt, h·ph, w·pw, C]</c>.</summary>
    public static Tensor Unpatchify(Tensor tokens, int t, int h, int w, int pt, int ph, int pw, int c)
    {
        int featDim = pt * ph * pw * c;
        if ((int)tokens.Shape[1] != featDim)
            throw new ArgumentException($"token feature {tokens.Shape[1]} != {featDim}.");
        if ((int)tokens.Shape[0] != t * h * w)
            throw new ArgumentException($"token count {tokens.Shape[0]} != {t * h * w}.");
        int tf = t * pt, hf = h * ph, wf = w * pw;

        Tensor latent = new Tensor(new TensorShape([(long)tf, hf, wf, c]), DType.F32);
        float* src = (float*)tokens.DataPointer;
        float* dst = (float*)latent.DataPointer;

        for (int ti = 0; ti < t; ti++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                {
                    long token = ((long)ti * h + hi) * w + wi;
                    for (int pti = 0; pti < pt; pti++)
                        for (int phi = 0; phi < ph; phi++)
                            for (int pwi = 0; pwi < pw; pwi++)
                                for (int ci = 0; ci < c; ci++)
                                {
                                    int feat = ((pti * ph + phi) * pw + pwi) * c + ci;
                                    long dstOff = (((long)(ti * pt + pti) * hf + (hi * ph + phi)) * wf + (wi * pw + pwi)) * c + ci;
                                    dst[dstOff] = src[token * featDim + feat];
                                }
                }
        // Same premature-finalization guard as Patchify, for the borrowed `tokens` input.
        GC.KeepAlive(tokens);
        return latent;
    }
}
