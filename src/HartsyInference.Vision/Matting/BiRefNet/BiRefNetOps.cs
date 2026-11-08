using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Tensor helpers the BiRefNet modules share: NCHW conv / bilinear / concat / activation calls routed
/// through <see cref="IBackend"/>, the host channel-broadcast multiply, and the upstream <c>image2patches</c> split.
/// Every helper returns a fresh tensor the caller disposes, except <see cref="Own"/>, which tracks ownership.</summary>
internal static unsafe class BiRefNetOps
{
    /// <summary>Stride-1 convolution with symmetric zero padding; output size follows the kernel.</summary>
    public static Tensor Conv(IBackend backend, Tensor x, Tensor w, Tensor? b, int pad)
    {
        int k = (int)w.Shape[2];
        int oh = (int)x.Shape[2] + 2 * pad - k + 1, ow = (int)x.Shape[3] + 2 * pad - k + 1;
        Tensor o = new(new TensorShape(1, (int)w.Shape[0], oh, ow), DType.F32);
        backend.Conv2D(o, x, w, b, 1, 1, pad, pad);
        return o;
    }

    /// <summary>Elementwise ReLU into a fresh tensor (a zero-slope LeakyReLU).</summary>
    public static Tensor Relu(IBackend backend, Tensor x)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.LeakyRelu(o, x, 0f);
        return o;
    }

    /// <summary>Elementwise sigmoid into a fresh tensor.</summary>
    public static Tensor Sigmoid(IBackend backend, Tensor x)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.Sigmoid(o, x);
        return o;
    }

    /// <summary>Elementwise sum into a fresh tensor.</summary>
    public static Tensor Add(IBackend backend, Tensor a, Tensor b)
    {
        Tensor o = new(a.Shape, DType.F32);
        backend.Add(o, a, b);
        return o;
    }

    /// <summary>Bilinear resize with <c>align_corners=True</c>, the mode every upstream <c>F.interpolate</c> call uses.</summary>
    public static Tensor Resize(IBackend backend, Tensor x, int h, int w)
    {
        Tensor o = new(new TensorShape(1, (int)x.Shape[1], h, w), DType.F32);
        backend.InterpolateBilinear2D(o, x, alignCorners: true);
        return o;
    }

    /// <summary>Channel concatenation of same-size NCHW tensors, in argument order.</summary>
    public static Tensor Concat(IBackend backend, params Tensor[] parts)
    {
        int channels = 0;
        foreach (Tensor part in parts) channels += (int)part.Shape[1];
        Tensor o = new(new TensorShape(1, channels, (int)parts[0].Shape[2], (int)parts[0].Shape[3]), DType.F32);
        backend.Concat(o, parts, 1);
        return o;
    }

    /// <summary>Multiplies every channel of <paramref name="x"/> <c>[1,C,H,W]</c> by the single-channel map <paramref name="gate"/> <c>[1,1,H,W]</c>.</summary>
    public static Tensor MulBroadcastGate(Tensor x, Tensor gate)
    {
        int c = (int)x.Shape[1], hw = (int)x.Shape[2] * (int)x.Shape[3];
        Tensor o = new(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer, gp = (float*)gate.DataPointer, op = (float*)o.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            long off = (long)ch * hw;
            for (int i = 0; i < hw; i++) op[off + i] = xp[off + i] * gp[i];
        }
        return o;
    }

    /// <summary>Upstream <c>image2patches</c> with <c>'b c (hg h) (wg w) -> b (c hg wg) h w'</c>: splits a
    /// <c>[1,3,H,W]</c> image into a <c>grid×grid</c> lattice of tiles stacked on channels, channel index
    /// <c>c·grid² + i·grid + j</c>.</summary>
    public static Tensor Patchify(Tensor image, int grid)
    {
        int c = (int)image.Shape[1], h = (int)image.Shape[2], w = (int)image.Shape[3];
        int th = h / grid, tw = w / grid;
        Tensor o = new(new TensorShape(1, c * grid * grid, th, tw), DType.F32);
        float* ip = (float*)image.DataPointer, op = (float*)o.DataPointer;
        for (int ch = 0; ch < c; ch++)
            for (int gi = 0; gi < grid; gi++)
                for (int gj = 0; gj < grid; gj++)
                {
                    long outCh = (long)ch * grid * grid + gi * grid + gj;
                    for (int y = 0; y < th; y++)
                        for (int xx = 0; xx < tw; xx++)
                            op[(outCh * th + y) * tw + xx] = ip[((long)ch * h + gi * th + y) * w + gj * tw + xx];
                }
        return o;
    }

    /// <summary>Takes ownership of a weight, converting it to F32 when needed. A tensor cast here is added to
    /// <paramref name="owned"/> so the model frees it; an F32 input is returned as-is and stays with its loader.</summary>
    public static Tensor Own(Tensor t, List<Tensor> owned)
    {
        if (t.DType == DType.F32) return t;
        Tensor cast = t.CastTo(DType.F32);
        owned.Add(cast);
        return cast;
    }
}
