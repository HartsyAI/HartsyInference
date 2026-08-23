using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Host-side tensor primitives shared by the VAE encoders/decoders (one copy, per the shared-primitive rule).</summary>
internal static unsafe class VaeOps
{
    /// <summary>Optional bias lookup: the tensor at <paramref name="key"/>, or null when the checkpoint carries none.</summary>
    internal static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? b : null;

    /// <summary>Host copy of <paramref name="x"/> into a fresh tensor of the same shape and dtype.</summary>
    internal static Tensor Clone(Tensor x)
    {
        Tensor copy = new Tensor(x.Shape, x.DType);
        long bytes = x.DType.ComputeByteCount(x.ElementCount);
        Buffer.MemoryCopy(x.DataPointer, copy.DataPointer, bytes, bytes);
        return copy;
    }

    /// <summary>Copies channels <c>[start, start + count)</c> of an F32 <c>[B, C, T, H, W]</c> tensor into a new <c>[B, count, T, H, W]</c> tensor.</summary>
    internal static Tensor SliceChannels(Tensor x, int start, int count)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor o = new Tensor(new TensorShape([(long)b, count, t, h, w]), DType.F32);
        long per = (long)t * h * w;
        float* sp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
            Buffer.MemoryCopy(
                sp + ((long)bi * c + start) * per,
                op + (long)bi * count * per,
                (long)count * per * 4, (long)count * per * 4);
        return o;
    }
}
