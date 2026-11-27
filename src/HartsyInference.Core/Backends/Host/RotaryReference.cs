using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the rotary contract (<c>docs/BACKEND_OPS.md</c>): for every pair
/// <c>(x0, x1)</c> with its entries <c>(c0, s0)</c> and <c>(c1, s1)</c>,
/// <c>x0' = x0·c0 − x1·s0</c> and <c>x1' = x1·c1 + x0·s1</c>, each product and sum rounded once. The
/// <see cref="IBackend"/> default and the oracle every kernel is checked against.</summary>
public static unsafe class RotaryReference
{
    public static void Apply(in RotaryCall call)
    {
        RotaryGeometry g = RotaryContract.Validate(call);
        if (g.DevicePosition != 0)
            throw new NotSupportedException("A device position buffer is a GPU decode facility; the host reference takes host-indexed tables.");
        float* cos = (float*)call.Cos.DataPointer;
        float* sin = (float*)call.Sin.DataPointer;
        void* x = call.X.DataPointer;
        if (call.X.DType == DType.F32) Rotate((float*)x, cos, sin, g, v => v, v => v);
        else if (call.X.DType == DType.F16) Rotate((Half*)x, cos, sin, g, v => (float)v, v => (Half)v);
        else Rotate((ushort*)x, cos, sin, g, BFloat16Bits.ToFloat, BFloat16Bits.FromFloat);
    }

    private static void Rotate<T>(T* x, float* cos, float* sin, in RotaryGeometry g, Func<T, float> load, Func<float, T> store) where T : unmanaged
    {
        Span<long> index = stackalloc long[RotaryGeometry.IndexAxes];
        long xo = g.XOffset, co = g.CosOffset, so = g.SinOffset;
        for (long n = 0; n < g.PairCount; n++)
        {
            float x0 = load(x[xo]), x1 = load(x[xo + g.XSecond]);
            float c0 = cos[co], c1 = cos[co + g.CosSecond];
            float s0 = sin[so], s1 = sin[so + g.SinSecond];
            x[xo] = store(x0 * c0 - x1 * s0);
            x[xo + g.XSecond] = store(x1 * c1 + x0 * s1);
            for (int axis = RotaryGeometry.IndexAxes - 1; axis >= 0; axis--)
            {
                index[axis]++;
                xo += g.XStride(axis);
                co += g.CosStride(axis);
                so += g.SinStride(axis);
                if (index[axis] < g.Dim(axis)) break;
                xo -= g.XStride(axis) * g.Dim(axis);
                co -= g.CosStride(axis) * g.Dim(axis);
                so -= g.SinStride(axis) * g.Dim(axis);
                index[axis] = 0;
            }
        }
    }
}
