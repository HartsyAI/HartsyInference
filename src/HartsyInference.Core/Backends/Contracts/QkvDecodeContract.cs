using HartsyInference.Core.Backends.Decode;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.QkvDecodeEpilogue"/>: F32 sections that fit their tensors, a head-major F32 q output
/// <c>[1, Heads, 1, HeadDim]</c>, two caches of one dtype and shape <c>[1, KvHeads, maxSeq, HeadDim]</c>, tables with a row per cache
/// slot and <c>HeadDim</c> entries, a rotated width the pairing allows, norm weights of the width the norm kind reads, and a device
/// position buffer. Every implementation validates through here.</summary>
public static class QkvDecodeContract
{
    public static void Validate(in QkvDecodeCall call)
    {
        ArgumentNullException.ThrowIfNull(call.QOut);
        ArgumentNullException.ThrowIfNull(call.KCache);
        ArgumentNullException.ThrowIfNull(call.VCache);
        ArgumentNullException.ThrowIfNull(call.Cos);
        ArgumentNullException.ThrowIfNull(call.Sin);
        if (call.Heads <= 0 || call.KvHeads <= 0 || call.HeadDim <= 0 || (call.HeadDim & 1) != 0)
            throw new ArgumentException($"A decode epilogue needs positive head counts and an even head dim; got heads {call.Heads}, kv heads {call.KvHeads}, head dim {call.HeadDim}.", nameof(call));
        if (call.DevicePosition == 0)
            throw new ArgumentException("A decode epilogue reads its slot from a device position buffer.", nameof(call));
        RequireSection(call.Q, call.QWidth, "q");
        RequireSection(call.K, call.KvWidth, "k");
        RequireSection(call.V, call.KvWidth, "v");
        if (call.QOut.DType != DType.F32 || !call.QOut.Shape.Equals(new TensorShape(1, call.Heads, 1, call.HeadDim)))
            throw new ArgumentException($"The q output is F32 [1, {call.Heads}, 1, {call.HeadDim}]; got {call.QOut.DType} {call.QOut.Shape}.", nameof(call));
        if (call.KCache.DType != call.VCache.DType || (call.KCache.DType != DType.F32 && call.KCache.DType != DType.F16))
            throw new NotSupportedException($"The caches are both F32 or both F16; got k {call.KCache.DType}, v {call.VCache.DType}.");
        if (!call.KCache.Shape.Equals(call.VCache.Shape))
            throw new ArgumentException($"The caches share one shape; got k {call.KCache.Shape}, v {call.VCache.Shape}.", nameof(call));
        if (call.KCache.Shape.Rank != 4 || call.KCache.Shape[0] != 1 || call.KCache.Shape[1] != call.KvHeads || call.KCache.Shape[3] != call.HeadDim || call.KCache.Shape[2] < 1)
            throw new ArgumentException($"A cache is [1, {call.KvHeads}, maxSeq, {call.HeadDim}]; got {call.KCache.Shape}.", nameof(call));
        if (call.Cos.DType != DType.F32 || call.Sin.DType != DType.F32)
            throw new NotSupportedException($"The tables are F32; got cos {call.Cos.DType}, sin {call.Sin.DType}.");
        if (!call.Cos.Shape.Equals(call.Sin.Shape) || call.Cos.Shape.Rank != 2 || call.Cos.Shape[1] != call.HeadDim || call.Cos.Shape[0] < call.CacheLength)
            throw new ArgumentException($"The tables are [rows ≥ {call.CacheLength}, {call.HeadDim}] and alike; got cos {call.Cos.Shape}, sin {call.Sin.Shape}.", nameof(call));
        if (call.Pairing == RotaryPairing.Interleaved && call.RotaryDim != 0)
            throw new ArgumentException("Interleaved pairing rotates the whole head; a partial width comes from identity table rows.", nameof(call));
        if (call.RotaryDim < 0 || call.RotaryDim > call.HeadDim || (call.RotaryDim & 1) != 0)
            throw new ArgumentException($"The rotated width is 0 or an even number up to the head dim {call.HeadDim}; got {call.RotaryDim}.", nameof(call));
        switch (call.Norm)
        {
            case QkvNormKind.None:
                if (call.QNormWeight is not null || call.KNormWeight is not null)
                    throw new ArgumentException("Norm weights are given only with a norm kind.", nameof(call));
                break;
            case QkvNormKind.PerHead:
                RequireWeight(call.QNormWeight, call.HeadDim, "q");
                RequireWeight(call.KNormWeight, call.HeadDim, "k");
                break;
            case QkvNormKind.FullWidth:
                RequireWeight(call.QNormWeight, call.QWidth, "q");
                RequireWeight(call.KNormWeight, call.KvWidth, "k");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(call), call.Norm, "Unknown norm kind.");
        }
        if (call.Norm != QkvNormKind.None && !(call.Eps > 0f))
            throw new ArgumentException($"A norm needs a positive epsilon; got {call.Eps}.", nameof(call));
        if (call.QOut.HasOverlappingHostStorageWithoutSync(call.Q.Tensor) || call.QOut.HasOverlappingHostStorageWithoutSync(call.K.Tensor) || call.QOut.HasOverlappingHostStorageWithoutSync(call.V.Tensor))
            throw new ArgumentException("The q output must not overlap a section.", nameof(call));
        if (ReferenceEquals(call.KCache, call.VCache))
            throw new ArgumentException("The key and value caches are distinct tensors.", nameof(call));
    }

    private static void RequireSection(in QkvSection section, long width, string name)
    {
        ArgumentNullException.ThrowIfNull(section.Tensor, name);
        if (section.Tensor.DType != DType.F32)
            throw new NotSupportedException($"The {name} section is F32; got {section.Tensor.DType}.");
        if (section.Offset < 0 || section.Offset + width > section.Tensor.ElementCount)
            throw new ArgumentException($"The {name} section [{section.Offset}, +{width}) leaves its tensor of {section.Tensor.ElementCount} elements.", name);
    }

    private static void RequireWeight(Tensor? weight, long width, string name)
    {
        if (weight is null)
            throw new ArgumentException($"The {name} norm needs a weight.", name);
        if (weight.DType != DType.F32 || weight.ElementCount != width)
            throw new ArgumentException($"The {name} norm weight holds {width} F32 elements; got {weight.DType} {weight.Shape}.", name);
    }
}
