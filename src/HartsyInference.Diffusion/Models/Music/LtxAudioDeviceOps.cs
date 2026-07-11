using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>Device-resident layout helpers for the LTX-2 audio decode tail (audio VAE + vocoder), built purely
/// from existing <see cref="IBackend"/> ops so the CPU and CUDA backends share one code path. These replace the
/// host <c>DataPointer</c> pad/crop/transpose loops that forced a full-pipeline D2H drain between every GPU conv
/// (the round-1 video-VAE lesson applied to the audio half).</summary>
public static class LtxAudioDeviceOps
{
    /// <summary>Edge-replicate ("replicate") padding of <c>[1, C, T]</c> along time. Runs as transpose →
    /// row-slice the first/last time rows → one contiguous concat (the repeated edge rows share one device
    /// buffer) → transpose back, so no per-channel copies are issued.</summary>
    public static Tensor ReplicatePadTime(IBackend backend, Tensor x, int padL, int padR)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        Tensor xt = new Tensor(new TensorShape(t, c), DType.F32);
        backend.Transpose2D(xt, x, c, t);
        Tensor first = new Tensor(new TensorShape(1, c), DType.F32);
        backend.SliceRows(first, xt, 0);
        Tensor last = new Tensor(new TensorShape(1, c), DType.F32);
        backend.SliceRows(last, xt, t - 1);
        Tensor[] parts = new Tensor[padL + 1 + padR];
        for (int i = 0; i < padL; i++) parts[i] = first;
        parts[padL] = xt;
        for (int i = 0; i < padR; i++) parts[padL + 1 + i] = last;
        Tensor paddedT = new Tensor(new TensorShape(t + padL + padR, c), DType.F32);
        backend.Concat(paddedT, parts, 0);
        Tensor outT = new Tensor(new TensorShape(1, c, t + padL + padR), DType.F32);
        backend.Transpose2D(outT, paddedT, t + padL + padR, c);
        xt.Dispose();
        first.Dispose();
        last.Dispose();
        paddedT.Dispose();
        return outT;
    }

    /// <summary>Crops <paramref name="left"/>/<paramref name="right"/> samples off the time axis of
    /// <c>[1, C, T]</c> (one per-row contiguous slice — time is the innermost dim).</summary>
    public static Tensor CropTime(IBackend backend, Tensor x, int left, int right)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        Tensor outT = new Tensor(new TensorShape(1, c, t - left - right), DType.F32);
        backend.SliceLastDim(outT, x, left);
        return outT;
    }

    /// <summary>Right-pads the time axis of <c>[1, C, T]</c> with zeros.</summary>
    public static Tensor PadRightZero(IBackend backend, Tensor x, int pad)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        Tensor zeros = new Tensor(new TensorShape(1, c, pad), DType.F32);
        backend.Fill(zeros, 0f);
        Tensor outT = new Tensor(new TensorShape(1, c, t + pad), DType.F32);
        backend.Concat(outT, [x, zeros], 2);
        zeros.Dispose();
        return outT;
    }

    /// <summary>Zero-pads the top of the time (height) axis of NCHW <c>[B, C, H, W]</c> — the audio VAE's causal
    /// conv padding. Permutes time-outer (<c>[B, H, C, W]</c>), where the pad is one contiguous concat, then
    /// permutes back.</summary>
    public static Tensor PadTimeTopZero(IBackend backend, Tensor x, int padTop)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor xt = new Tensor(new TensorShape(b, h, c, w), DType.F32);
        backend.Permute0213(xt, x, c, h, w);
        Tensor zeros = new Tensor(new TensorShape(b, padTop, c, w), DType.F32);
        backend.Fill(zeros, 0f);
        Tensor cat = new Tensor(new TensorShape(b, h + padTop, c, w), DType.F32);
        backend.Concat(cat, [zeros, xt], 1);
        Tensor outT = new Tensor(new TensorShape(b, c, h + padTop, w), DType.F32);
        backend.Permute0213(outT, cat, h + padTop, c, w);
        xt.Dispose();
        zeros.Dispose();
        cat.Dispose();
        return outT;
    }

    /// <summary>Drops the first time (height) row of NCHW <c>[1, C, H, W]</c> → <c>[1, C, H-1, W]</c> (the audio
    /// VAE upsampler's causal alignment). Batch-1 only: the permuted time-outer layout makes the drop a single
    /// leading-rows slice.</summary>
    public static Tensor DropFirstTimeRow(IBackend backend, Tensor x)
    {
        if (x.Shape[0] != 1)
            throw new ArgumentException($"DropFirstTimeRow requires batch 1, got {x.Shape[0]}.", nameof(x));
        int c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor xt = new Tensor(new TensorShape(1, h, c, w), DType.F32);
        backend.Permute0213(xt, x, c, h, w);
        Tensor sliced = new Tensor(new TensorShape(1, h - 1, c, w), DType.F32);
        backend.SliceRows(sliced, xt, c);
        Tensor outT = new Tensor(new TensorShape(1, c, h - 1, w), DType.F32);
        backend.Permute0213(outT, sliced, h - 1, c, w);
        xt.Dispose();
        sliced.Dispose();
        return outT;
    }

    /// <summary>Flattens a mel tensor <c>[1, C, frames, melBins]</c> into vocoder-generator input
    /// <c>[1, C·melBins, frames]</c> (batched last-two-dim transpose: <c>out[c·bins+bin, t] = in[c, t, bin]</c>).</summary>
    public static Tensor FlattenMel(IBackend backend, Tensor mel)
    {
        int c = (int)mel.Shape[1], frames = (int)mel.Shape[2], melBins = (int)mel.Shape[3];
        Tensor outT = new Tensor(new TensorShape(1, (long)c * melBins, frames), DType.F32);
        backend.Permute0213(outT, mel, frames, melBins, 1);
        return outT;
    }
}
