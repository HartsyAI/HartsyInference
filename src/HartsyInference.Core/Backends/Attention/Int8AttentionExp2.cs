namespace HartsyInference.Core.Backends.Attention;

/// <summary>The <c>2^x</c> of INT8 attention on Vulkan: the host twin, operation for operation, of <c>exp2Fixed</c> in
/// <c>HartsyInference.Vulkan/Shaders/exp2_fixed.glsl</c>. A device's own <c>exp2()</c> differs between drivers, so the
/// shaders compute it from a floor, correctly rounded multiplies and adds, and an exact scaling by a power of two; passed
/// as <see cref="Int8AttentionV2Options.Exp2"/>, this makes <see cref="Int8AttentionV2Reference"/> the specification of
/// what they produce. The two must change together.</summary>
public static class Int8AttentionExp2
{
    /// <summary>Arguments above this give +∞, also those below 128 whose true power is finite: the attention pass asks
    /// for nothing above <see cref="Int8AttentionV2Reference.U8Offset"/>.</summary>
    public const float Max = 127f;

    /// <summary>Arguments below this give exactly zero, so no result is subnormal.</summary>
    public const float Min = -125f;

    // ln(2)^n / n!, n = 1 .. 7, as the float nearest each.
    private static readonly float C1 = BitConverter.UInt32BitsToSingle(0x3F317218u);
    private static readonly float C2 = BitConverter.UInt32BitsToSingle(0x3E75FDF0u);
    private static readonly float C3 = BitConverter.UInt32BitsToSingle(0x3D635847u);
    private static readonly float C4 = BitConverter.UInt32BitsToSingle(0x3C1D955Bu);
    private static readonly float C5 = BitConverter.UInt32BitsToSingle(0x3AAEC3FFu);
    private static readonly float C6 = BitConverter.UInt32BitsToSingle(0x39218489u);
    private static readonly float C7 = BitConverter.UInt32BitsToSingle(0x377FE5FEu);

    /// <summary><c>2^x</c> within about one unit in the last place: <c>x = k + r</c> with <c>k</c> the nearest integer, the
    /// degree-7 Taylor polynomial of <c>e^(r ln 2)</c> by Horner's rule in single precision with no fused step, times
    /// <c>2^k</c>.</summary>
    public static float Evaluate(float x)
    {
        if (float.IsNaN(x)) return x;
        if (x > Max) return float.PositiveInfinity;
        if (x < Min) return 0f;
        float k = MathF.Floor(x + 0.5f);
        float r = x - k;
        // Each product is rounded to single before its add. The JIT does that today without the casts; they are a
        // guard against a compiler that would fuse or widen the step.
        float p = C7;
        p = (float)(p * r) + C6;
        p = (float)(p * r) + C5;
        p = (float)(p * r) + C4;
        p = (float)(p * r) + C3;
        p = (float)(p * r) + C2;
        p = (float)(p * r) + C1;
        p = (float)(p * r) + 1f;
        float scale = BitConverter.UInt32BitsToSingle((uint)((int)k + 127) << 23);
        return p * scale;
    }
}
