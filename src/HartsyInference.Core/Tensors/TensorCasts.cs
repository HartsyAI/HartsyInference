namespace HartsyInference.Core.Tensors;

/// <summary>Host-side dtype-cast primitives shared across packages. Unlike backend-routed cast helpers (e.g. Diffusion's <c>DtypeCastHelper</c>), these never touch a backend and never dispose the source — safe for borrowed weights-dict tensors.</summary>
public static class TensorCasts
{
    /// <summary>Returns <paramref name="t"/> as F32, host-casting if needed. On the pass-through (already-F32) path the caller does not own the result.</summary>
    public static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    /// <summary>Looks up <paramref name="key"/> in <paramref name="weights"/> and returns it as F32 via <see cref="EnsureF32"/>.</summary>
    public static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> weights, string key) => EnsureF32(weights[key]);

    /// <summary>As <see cref="LoadF32"/> but returns null when <paramref name="key"/> is absent, for optional weights.</summary>
    public static Tensor? LoadF32Opt(IReadOnlyDictionary<string, Tensor> weights, string key)
        => weights.TryGetValue(key, out Tensor? t) ? EnsureF32(t) : null;

    /// <summary>Truncating F32→BF16 bit pattern, matching <see cref="Tensor.CastTo"/> bit for bit (not round-to-nearest-even).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort F32ToBf16Bits(float value) => (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);

    /// <summary>Returns a NEW owned F32 <c>[in, out]</c>→<c>[out, in]</c> relabel of an F32 rank-2 GGUF weight — the bytes are already row-major <c>[out, in]</c>, so only the shape changes, but the data is copied so the result never aliases the loader's mmap; contrast <c>GgufModelLoader.RelabelRank2ToPyTorchOrder</c>, which is a dtype-preserving metadata-only reshape returning borrowed views.</summary>
    public static unsafe Tensor RelabelRank2Copy(Tensor t)
    {
        Tensor outp = new(new TensorShape((int)t.Shape[1], (int)t.Shape[0]), DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)outp.DataPointer, outp.ElementCount * 4, t.ElementCount * 4);
        return outp;
    }
}
