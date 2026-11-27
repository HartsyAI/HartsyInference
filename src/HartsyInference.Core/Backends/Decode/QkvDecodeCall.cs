using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Decode;

/// <summary>One <see cref="IBackend.QkvDecodeEpilogue"/> invocation: a decode step's q, k and v sections, the optional QK norm,
/// the rotation and the caches the step scatters into. The factories cover the three ways a transformer packs its projection
/// output; <see cref="WithNorm"/> adds the norm.</summary>
/// <param name="QOut">Where the rotated q heads go: F32, head-major <c>[1, Heads, 1, HeadDim]</c>.</param>
/// <param name="KCache">The key cache <c>[1, KvHeads, maxSeq, HeadDim]</c>, F32 or F16; slot <c>position</c> of every head is written.</param>
/// <param name="VCache">The value cache, the same shape and dtype as <paramref name="KCache"/>.</param>
/// <param name="Q">The q section, <c>Heads·HeadDim</c> F32 elements.</param>
/// <param name="K">The k section, <c>KvHeads·HeadDim</c> F32 elements.</param>
/// <param name="V">The v section, <c>KvHeads·HeadDim</c> F32 elements.</param>
/// <param name="Heads">Query heads.</param>
/// <param name="KvHeads">Key and value heads.</param>
/// <param name="HeadDim">Elements per head.</param>
/// <param name="Cos">The cosine table <c>[rows, HeadDim]</c>, F32, one row per position; row <c>position</c> is read.</param>
/// <param name="Sin">The sine table, the same shape as <paramref name="Cos"/>.</param>
/// <param name="RotaryDim">Rotated width of each head for <see cref="RotaryPairing.RotateHalf"/>; 0 rotates the whole head.
/// <see cref="RotaryPairing.Interleaved"/> always rotates the whole head and takes a partial width from identity table rows, so it is 0.</param>
/// <param name="Pairing">Which elements of a head pair up; the table row holds one entry per element (rotate-half) or per pair, in its first half (interleaved).</param>
/// <param name="DevicePosition">The device position buffer from <see cref="IBackend.AllocDevicePos"/>; its query offset (second value) is the step's position.</param>
/// <param name="Norm">The RMS normalization applied to q and k before the rotation.</param>
/// <param name="QNormWeight">The q norm weight, F32: <c>[HeadDim]</c> per head or <c>[Heads·HeadDim]</c> full width.</param>
/// <param name="KNormWeight">The k norm weight, F32: <c>[HeadDim]</c> per head or <c>[KvHeads·HeadDim]</c> full width.</param>
/// <param name="Eps">The norm epsilon.</param>
public readonly record struct QkvDecodeCall(
    Tensor QOut, Tensor KCache, Tensor VCache,
    QkvSection Q, QkvSection K, QkvSection V,
    int Heads, int KvHeads, int HeadDim,
    Tensor Cos, Tensor Sin, int RotaryDim, RotaryPairing Pairing, ulong DevicePosition,
    QkvNormKind Norm = QkvNormKind.None, Tensor? QNormWeight = null, Tensor? KNormWeight = null, float Eps = 0f)
{
    /// <summary>A fused projection output <c>[q | k | v]</c>.</summary>
    public static QkvDecodeCall Fused(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qkv, int heads, int kvHeads, int headDim,
        Tensor cos, Tensor sin, int rotaryDim, RotaryPairing pairing, ulong devicePosition)
    {
        long nq = (long)heads * headDim, nkv = (long)kvHeads * headDim;
        return new(qOut, kCache, vCache, new QkvSection(qkv, 0), new QkvSection(qkv, nq), new QkvSection(qkv, nq + nkv),
            heads, kvHeads, headDim, cos, sin, rotaryDim, pairing, devicePosition);
    }

    /// <summary>A projection output <c>[q | k]</c> with v projected on its own.</summary>
    public static QkvDecodeCall QkPlusV(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor v, int heads, int kvHeads, int headDim,
        Tensor cos, Tensor sin, int rotaryDim, RotaryPairing pairing, ulong devicePosition)
        => new(qOut, kCache, vCache, new QkvSection(qk, 0), new QkvSection(qk, (long)heads * headDim), v,
            heads, kvHeads, headDim, cos, sin, rotaryDim, pairing, devicePosition);

    /// <summary>q, k and v each projected on their own.</summary>
    public static QkvDecodeCall Separate(Tensor qOut, Tensor kCache, Tensor vCache, Tensor q, Tensor k, Tensor v, int heads, int kvHeads, int headDim,
        Tensor cos, Tensor sin, int rotaryDim, RotaryPairing pairing, ulong devicePosition)
        => new(qOut, kCache, vCache, q, k, v, heads, kvHeads, headDim, cos, sin, rotaryDim, pairing, devicePosition);

    /// <summary>The same call with an RMS QK norm before the rotation.</summary>
    public QkvDecodeCall WithNorm(QkvNormKind kind, Tensor qWeight, Tensor kWeight, float eps)
        => this with { Norm = kind, QNormWeight = qWeight, KNormWeight = kWeight, Eps = eps };

    /// <summary>Elements of the q section.</summary>
    public long QWidth => (long)Heads * HeadDim;

    /// <summary>Elements of the k and of the v section.</summary>
    public long KvWidth => (long)KvHeads * HeadDim;

    /// <summary>Slots in each cache.</summary>
    public long CacheLength => KCache.Shape[2];
}
