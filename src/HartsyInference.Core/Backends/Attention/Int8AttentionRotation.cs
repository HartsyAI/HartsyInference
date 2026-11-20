namespace HartsyInference.Core.Backends.Attention;

/// <summary>The orthonormal transform INT8 attention applies to each Q row and each K row, along the padded head
/// dim, before they are quantized. Rotating both sides leaves every <c>q · k</c> unchanged in exact arithmetic and
/// spreads a channel that is far larger than the rest over all of them, so one absmax scale no longer crushes the
/// other channels to zero. V is never rotated.</summary>
public enum Int8Rotation
{
    /// <summary>No transform.</summary>
    None,

    /// <summary>Whichever of the transforms below the pinned selection rule gives the call
    /// (<see cref="Int8AttentionRotation.Select"/>).</summary>
    Pinned,

    /// <summary>A 4-point Hadamard on every four adjacent channels.</summary>
    H4,

    /// <summary>One 64-point Hadamard; the padded head dim is 64.</summary>
    H64,

    /// <summary>A fixed sign flip per channel, then one 128-point Hadamard; the padded head dim is 128.</summary>
    SignedH128,

    /// <summary>A 128-point Hadamard on channels 0 to 127 and another on 128 to 255; the padded head dim is 256.</summary>
    PaddedH128,
}

/// <summary>The rotations of INT8 attention as Comfy Kitchen's kernels compute them at the pinned commit
/// (benchmarks/attention/protocol.json), in the same F32 operation order, so the rotated row and therefore its INT8
/// bytes can match a kernel's exactly. Every transform is a Sylvester (natural-order) Walsh-Hadamard,
/// <c>y[i] = n^-1/2 · Σ_j (-1)^popcount(i &amp; j) · x[j]</c>.</summary>
public static class Int8AttentionRotation
{
    /// <summary>The key count up to which every head dim takes <see cref="Int8Rotation.H4"/>.</summary>
    public const int ShortKeyLimit = 256;

    private const float H128Scale = 0.1767766952966369f;

    // One bit per channel of the signed 128-point transform: set keeps the value, clear negates it.
    private static readonly uint[] SignWords = [0x1035997b, 0x8087f5ee, 0xee2e4e1a, 0x71132418];

    /// <summary>The transform the pinned kernels pick for a padded head dim (64, 128 or 256) and a valid key count: the
    /// 4-point one up to <see cref="ShortKeyLimit"/> keys, past that the widest the head dim holds.</summary>
    public static Int8Rotation Select(int paddedHeadDim, int validKeys)
    {
        if (paddedHeadDim is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(paddedHeadDim), paddedHeadDim, "The padded head dim is 64, 128 or 256.");
        if (validKeys <= ShortKeyLimit) return Int8Rotation.H4;
        return paddedHeadDim switch
        {
            64 => Int8Rotation.H64,
            128 => Int8Rotation.SignedH128,
            _ => Int8Rotation.PaddedH128,
        };
    }

    /// <summary><paramref name="rotation"/> with <see cref="Int8Rotation.Pinned"/> resolved for the call's padded head dim
    /// and key count.</summary>
    /// <exception cref="ArgumentException">The transform does not fit the padded head dim.</exception>
    public static Int8Rotation Resolve(Int8Rotation rotation, int paddedHeadDim, int validKeys)
    {
        Int8Rotation resolved = rotation == Int8Rotation.Pinned ? Select(paddedHeadDim, validKeys) : rotation;
        int needs = resolved switch
        {
            Int8Rotation.H64 => 64,
            Int8Rotation.SignedH128 => 128,
            Int8Rotation.PaddedH128 => 256,
            _ => paddedHeadDim,
        };
        if (needs != paddedHeadDim)
            throw new ArgumentException($"{resolved} runs at a padded head dim of {needs}, not {paddedHeadDim}.", nameof(rotation));
        return resolved;
    }

    /// <summary>Whether the signed 128-point transform keeps (true) or negates (false) <paramref name="channel"/>.</summary>
    public static bool KeepsSign(int channel)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(channel);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(channel, 128);
        return ((SignWords[channel >> 5] >> (channel & 31)) & 1) != 0;
    }

    /// <summary>Rotates one padded row in place. <paramref name="rotation"/> is a resolved transform, not
    /// <see cref="Int8Rotation.Pinned"/>, and the row is as long as the padded head dim it runs at.</summary>
    public static void Apply(Span<float> row, Int8Rotation rotation)
    {
        switch (rotation)
        {
            case Int8Rotation.None:
                return;
            case Int8Rotation.H4:
                RequireLength(row, row.Length - row.Length % 4, rotation);
                FourPoint(row);
                return;
            case Int8Rotation.H64:
                RequireLength(row, 64, rotation);
                Hadamard(row, 0.25f);
                return;
            case Int8Rotation.SignedH128:
                RequireLength(row, 128, rotation);
                for (int channel = 0; channel < 128; channel++)
                    if (!KeepsSign(channel)) row[channel] = -row[channel];
                Hadamard(row, H128Scale);
                return;
            case Int8Rotation.PaddedH128:
                RequireLength(row, 256, rotation);
                Hadamard(row[..128], H128Scale);
                Hadamard(row[128..], H128Scale);
                return;
            default:
                throw new ArgumentException($"{rotation} is not a resolved rotation.", nameof(rotation));
        }
    }

    private static void RequireLength(Span<float> row, int length, Int8Rotation rotation)
    {
        if (row.Length != length || length == 0)
            throw new ArgumentException($"{rotation} does not run over a row of {row.Length} channels.", nameof(row));
    }

    /// <summary>The 4-point stage on every four channels, halving included: the whole of <see cref="Int8Rotation.H4"/>
    /// and the first stage of the wider transforms.</summary>
    private static void FourPoint(Span<float> row)
    {
        for (int c = 0; c < row.Length; c += 4)
        {
            float a0 = row[c] + row[c + 1], a1 = row[c] - row[c + 1], a2 = row[c + 2] + row[c + 3], a3 = row[c + 2] - row[c + 3];
            row[c] = (a0 + a2) * 0.5f;
            row[c + 1] = (a1 + a3) * 0.5f;
            row[c + 2] = (a0 - a2) * 0.5f;
            row[c + 3] = (a1 - a3) * 0.5f;
        }
    }

    /// <summary>A Hadamard over the whole span (64 or 128 channels): the 4-point stage, then one radix-2 stage per
    /// higher bit of the channel index in ascending order, then <paramref name="scale"/>, which with the 4-point
    /// stage's halving makes the transform orthonormal. Nothing here is a multiply feeding an add, so a fused
    /// multiply-add cannot change a result.</summary>
    private static void Hadamard(Span<float> row, float scale)
    {
        FourPoint(row);
        for (int stride = 4; stride < row.Length; stride <<= 1)
            for (int block = 0; block < row.Length; block += stride << 1)
                for (int i = block; i < block + stride; i++)
                {
                    float lo = row[i], hi = row[i + stride];
                    row[i] = lo + hi;
                    row[i + stride] = lo - hi;
                }
        for (int c = 0; c < row.Length; c++) row[c] *= scale;
    }
}
