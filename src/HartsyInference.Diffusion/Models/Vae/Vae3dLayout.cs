using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Shared layout helpers for 3D video VAEs: convert between the channel-major <c>[B, C, T, H, W]</c> tensor form and the per-frame <c>[B·T, C, H, W]</c> form that 2D conv/attention ops consume. Used by every block that processes frames independently (attention, spatial resample). Reusable across Wan/LTX video VAEs.
///
/// <para><b>GPU-resident overloads:</b> the <c>backend</c>-taking overloads keep the whole VAE decode on-device — the
/// host overloads below read <see cref="Tensor.DataPointer"/>, which forces a D2H sync + H2D re-upload around every
/// layout op (thousands per clip) and strands the GPU at ~10% util. For <c>B==1</c> (the only case any video VAE
/// decode hits) these map onto the existing <c>Permute0213</c>/<c>SliceRows</c>/<c>Concat</c> GPU kernels; they fall
/// back to the host path for <c>B&gt;1</c>.</para></summary>
public static unsafe class Vae3dLayout
{
    // ---- GPU-resident overloads (B==1) --------------------------------------------------------------------------

    /// <summary><c>[1, C, T, H, W] → [T, C, H, W]</c> on-device (Permute0213: swap the C and T axes). Host fallback for B&gt;1.</summary>
    public static Tensor ToFrames(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        if (b != 1) return ToFrames(x, b, c, t, h, w);
        Tensor outT = new Tensor(new TensorShape(t, c, h, w), x.DType);
        backend.Permute0213(outT, x, c, t, h * w);          // [C,T,HW] → [T,C,HW]
        return outT;
    }

    /// <summary><c>[T, C, H, W] → [1, C, T, H, W]</c> on-device (inverse of <see cref="ToFrames(IBackend,Tensor)"/>). Host fallback for B&gt;1.</summary>
    public static Tensor FromFrames(IBackend backend, Tensor x, int b, int c, int t, int h, int w)
    {
        if (b != 1) return FromFrames(x, b, c, t, h, w);
        Tensor outT = new Tensor(new TensorShape([1L, c, t, h, w]), x.DType);
        backend.Permute0213(outT, x, t, c, h * w);          // [T,C,HW] → [C,T,HW]
        return outT;
    }

    /// <summary>On-device temporal concat (<c>Concat</c> along T). Host fallback when <paramref name="backend"/> is null.</summary>
    public static Tensor ConcatFrames(IBackend backend, IReadOnlyList<Tensor> parts)
    {
        int b = (int)parts[0].Shape[0], c = (int)parts[0].Shape[1], h = (int)parts[0].Shape[3], w = (int)parts[0].Shape[4];
        int totalT = 0; foreach (Tensor p in parts) totalT += (int)p.Shape[2];
        Tensor outT = new Tensor(new TensorShape([(long)b, c, totalT, h, w]), parts[0].DType);
        Tensor[] arr = parts as Tensor[] ?? System.Linq.Enumerable.ToArray(parts);
        backend.Concat(outT, arr, dim: 2);
        return outT;
    }

    /// <summary>On-device temporal slice for <c>B==1</c>: <c>[1,C,T,H,W] → [1,C,count,H,W]</c> via frame-major reorder +
    /// contiguous row slice. Host fallback for B&gt;1.</summary>
    public static Tensor SliceFrames(IBackend backend, Tensor x, int tStart, int count)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[3], w = (int)x.Shape[4];
        if (b != 1) return SliceFrames(x, tStart, count);
        using Tensor fm = ToFrames(backend, x);              // [T,C,H,W] contiguous frame-major
        using Tensor slice = new Tensor(new TensorShape(count, c, h, w), x.DType);
        backend.SliceRows(slice, fm, tStart * c * h);        // contiguous block [tStart..tStart+count) frames
        return FromFrames(backend, slice, 1, c, count, h, w);
    }

    /// <summary>On-device: prepend the last frame of <paramref name="prefix"/> before <paramref name="x"/> along T.</summary>
    public static Tensor PrependLastFrameOf(IBackend backend, Tensor prefix, Tensor x)
    {
        int pt = (int)prefix.Shape[2];
        using Tensor last = SliceFrames(backend, prefix, pt - 1, 1);
        return ConcatFrames(backend, [last, x]);
    }

    /// <summary>On-device: prepend a zero frame before <paramref name="x"/> along T.</summary>
    public static Tensor PrependZeroFrame(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[3], w = (int)x.Shape[4];
        using Tensor zero = new Tensor(new TensorShape([(long)b, c, 1, h, w]), x.DType);
        System.Runtime.InteropServices.NativeMemory.Clear(zero.DataPointer, (nuint)(zero.Shape.ElementCount * x.DType.SizeInBytes));
        return ConcatFrames(backend, [zero, x]);
    }

    // ---- Host overloads (B>1 / CPU fallback) --------------------------------------------------------------------

    /// <summary><c>[B, C, T, H, W] → [B·T, C, H, W]</c> (frames batched).</summary>
    public static Tensor ToFrames(Tensor x, int b, int c, int t, int h, int w)
    {
        int es = x.DType.SizeInBytes;
        Tensor outT = new Tensor(new TensorShape(b * t, c, h, w), x.DType);
        byte* s = (byte*)x.DataPointer;
        byte* d = (byte*)outT.DataPointer;
        long frame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
                for (int ci = 0; ci < c; ci++)
                {
                    long src = (((long)bi * c + ci) * t + ti) * frame;
                    long dst = (((long)bi * t + ti) * c + ci) * frame;
                    Buffer.MemoryCopy(s + src * es, d + dst * es, frame * es, frame * es);
                }
        return outT;
    }

    /// <summary>Slices <paramref name="count"/> temporal frames starting at <paramref name="tStart"/> from a <c>[B,C,T,H,W]</c> tensor → <c>[B,C,count,H,W]</c>.</summary>
    public static Tensor SliceFrames(Tensor x, int tStart, int count)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int es = x.DType.SizeInBytes;
        Tensor outT = new Tensor(new TensorShape([(long)b, c, count, h, w]), x.DType);
        byte* s = (byte*)x.DataPointer;
        byte* d = (byte*)outT.DataPointer;
        long frame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < count; ti++)
                {
                    long src = (((long)bi * c + ci) * t + (tStart + ti)) * frame;
                    long dst = (((long)bi * c + ci) * count + ti) * frame;
                    Buffer.MemoryCopy(s + src * es, d + dst * es, frame * es, frame * es);
                }
        return outT;
    }

    /// <summary>Concatenates a list of <c>[B,C,Ti,H,W]</c> tensors along the temporal axis → <c>[B,C,ΣTi,H,W]</c>.</summary>
    public static Tensor ConcatFrames(IReadOnlyList<Tensor> parts)
    {
        int b = (int)parts[0].Shape[0], c = (int)parts[0].Shape[1], h = (int)parts[0].Shape[3], w = (int)parts[0].Shape[4];
        int totalT = 0;
        foreach (Tensor p in parts) totalT += (int)p.Shape[2];
        int es = parts[0].DType.SizeInBytes;
        Tensor outT = new Tensor(new TensorShape([(long)b, c, totalT, h, w]), parts[0].DType);
        byte* d = (byte*)outT.DataPointer;
        long frame = (long)h * w;
        int tOff = 0;
        foreach (Tensor p in parts)
        {
            int pt = (int)p.Shape[2];
            byte* s = (byte*)p.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int ci = 0; ci < c; ci++)
                    for (int ti = 0; ti < pt; ti++)
                    {
                        long src = (((long)bi * c + ci) * pt + ti) * frame;
                        long dst = (((long)bi * c + ci) * totalT + (tOff + ti)) * frame;
                        Buffer.MemoryCopy(s + src * es, d + dst * es, frame * es, frame * es);
                    }
            tOff += pt;
        }
        return outT;
    }

    /// <summary>Prepends a single frame (the last frame of <paramref name="prefix"/>) before <paramref name="x"/> along T → <c>[B,C,1+Tx,H,W]</c>.</summary>
    public static Tensor PrependLastFrameOf(Tensor prefix, Tensor x)
    {
        int pt = (int)prefix.Shape[2];
        using Tensor last = SliceFrames(prefix, pt - 1, 1);
        return ConcatFrames([last, x]);
    }

    /// <summary>Prepends a zero frame before <paramref name="x"/> along T → <c>[B,C,1+Tx,H,W]</c>.</summary>
    public static Tensor PrependZeroFrame(Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor zero = new Tensor(new TensorShape([(long)b, c, 1, h, w]), x.DType);
        System.Runtime.InteropServices.NativeMemory.Clear(zero.DataPointer, (nuint)(zero.Shape.ElementCount * x.DType.SizeInBytes));
        Tensor outT = ConcatFrames([zero, x]);
        zero.Dispose();
        return outT;
    }

    /// <summary><c>[B·T, C, H, W] → [B, C, T, H, W]</c> (inverse of <see cref="ToFrames"/>; H/W may differ from the input to ToFrames after a spatial op).</summary>
    public static Tensor FromFrames(Tensor x, int b, int c, int t, int h, int w)
    {
        int es = x.DType.SizeInBytes;
        Tensor outT = new Tensor(new TensorShape([(long)b, c, t, h, w]), x.DType);
        byte* s = (byte*)x.DataPointer;
        byte* d = (byte*)outT.DataPointer;
        long frame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
                for (int ci = 0; ci < c; ci++)
                {
                    long src = (((long)bi * t + ti) * c + ci) * frame;
                    long dst = (((long)bi * c + ci) * t + ti) * frame;
                    Buffer.MemoryCopy(s + src * es, d + dst * es, frame * es, frame * es);
                }
        return outT;
    }
}
