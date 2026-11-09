using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the uniform fake-quantize contract (<c>docs/BACKEND_OPS.md</c>): each value is clamped to
/// <c>[low, high]</c>, truncated to one of <c>levels</c> evenly spaced steps and mapped back into the range. The
/// <see cref="IBackend"/> default and the oracle the kernel is checked against.</summary>
public static unsafe class FakeQuantizeReference
{
    /// <summary>The quantized value of <paramref name="v"/> over <c>steps = levels − 1</c> intervals of <c>range = high − low</c>.</summary>
    public static float Quantize(float v, float low, float high, float range, float steps)
    {
        if (v > high) v = high; else if (v < low) v = low;
        int b = (int)((v - low) / range * steps);
        return (float)b / steps * range + low;
    }

    public static void Apply(Tensor output, Tensor input, float low, float high, int levels)
    {
        long count = FakeQuantizeContract.Validate(output, input, low, high, levels);
        float range = high - low, steps = levels - 1;
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (long i = 0; i < count; i++) op[i] = Quantize(ip[i], low, high, range, steps);
    }
}
