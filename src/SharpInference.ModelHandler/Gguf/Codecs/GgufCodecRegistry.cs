using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Gguf.Codecs;

/// <summary>Static registry of every available <see cref="IGgufCodec"/>. Lookup by <see cref="DType"/>.
///
/// <para>Adding a new codec:</para>
/// <list type="number">
/// <item>Add the corresponding <see cref="DType"/> static instance in <see cref="DType"/>.</item>
/// <item>Implement <see cref="GgufCodecBase"/> (or <see cref="IGgufCodec"/> directly) for the new type.</item>
/// <item>Register here in the static constructor's <see cref="Register"/> call.</item>
/// </list>
///
/// <para>The registry is populated lazily-immutably at first access — no need to call any init method.</para></summary>
public static class GgufCodecRegistry
{
    private static readonly Dictionary<DType, IGgufCodec> _codecs = BuildRegistry();

    private static Dictionary<DType, IGgufCodec> BuildRegistry()
    {
        Dictionary<DType, IGgufCodec> r = new();
        Register(r, new Codec_Q8_0());
        Register(r, new Codec_Q4_K());
        Register(r, new Codec_Q5_K());
        Register(r, new Codec_Q4_0());
        Register(r, new Codec_Q4_1());
        Register(r, new Codec_Q5_0());
        Register(r, new Codec_Q5_1());
        Register(r, new Codec_Q8_1());
        Register(r, new Codec_Q2_K());
        Register(r, new Codec_Q3_K());
        Register(r, new Codec_Q6_K());
        Register(r, new Codec_IQ4_NL());
        return r;
    }

    private static void Register(Dictionary<DType, IGgufCodec> r, IGgufCodec codec)
    {
        if (!codec.DType.IsQuantized)
            throw new InvalidOperationException($"Codec for {codec.DType} cannot register — DType is not quantized.");
        if (r.ContainsKey(codec.DType))
            throw new InvalidOperationException($"Duplicate codec registration for {codec.DType}.");
        r[codec.DType] = codec;
    }

    /// <summary>Looks up a codec by DType. Throws if no codec is registered.</summary>
    public static IGgufCodec Get(DType dtype)
    {
        if (!_codecs.TryGetValue(dtype, out IGgufCodec? c))
            throw new SharpInference.Core.Exceptions.SharpInferenceException(
                $"No GGUF codec registered for {dtype}. Register one in {nameof(GgufCodecRegistry)}.");
        return c;
    }

    /// <summary>Whether a codec exists for this dtype.</summary>
    public static bool Supports(DType dtype) => _codecs.ContainsKey(dtype);

    /// <summary>All registered codec dtypes (read-only).</summary>
    public static IReadOnlyCollection<DType> Registered => _codecs.Keys;
}
