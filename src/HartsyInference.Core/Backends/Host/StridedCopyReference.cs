using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the strided copy, accumulate and pad contracts (<c>docs/BACKEND_OPS.md</c>): the
/// <see cref="IBackend"/> defaults, and the oracle every accelerator kernel is checked against.</summary>
public static unsafe class StridedCopyReference
{
    /// <summary><c>output[dst(i)] = input[src(i)]</c> over the spec's extents; elements the destination does not address are untouched.</summary>
    public static void Copy(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        StridedCopyGeometry g = StridedCopyContract.ValidateCopy(output, input, spec);
        void* o = output.DataPointer;
        void* i = input.DataPointer;
        switch (g.ElementSize)
        {
            case 1: Copy((byte*)o, (byte*)i, g); break;
            case 2: Copy((ushort*)o, (ushort*)i, g); break;
            case 4: Copy((uint*)o, (uint*)i, g); break;
            default: Copy((ulong*)o, (ulong*)i, g); break;
        }
    }

    /// <summary><c>output[dst(i)] += input[src(i)]</c> in F32 arithmetic, rounded to the storage dtype (bfloat16 ties to even).</summary>
    public static void Accumulate(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        StridedCopyGeometry g = StridedCopyContract.ValidateAccumulate(output, input, spec);
        if (output.DType == DType.F32)
        {
            float* o = (float*)output.DataPointer, i = (float*)input.DataPointer;
            Walk(g, (d, s) => o[d] += i[s]);
        }
        else if (output.DType == DType.F16)
        {
            Half* o = (Half*)output.DataPointer, i = (Half*)input.DataPointer;
            Walk(g, (d, s) => o[d] = (Half)((float)o[d] + (float)i[s]));
        }
        else if (input.DType == DType.F32)
        {
            ushort* o = (ushort*)output.DataPointer;
            float* i = (float*)input.DataPointer;
            Walk(g, (d, s) => o[d] = BFloat16Bits.FromFloat(BFloat16Bits.ToFloat(o[d]) + i[s]));
        }
        else
        {
            ushort* o = (ushort*)output.DataPointer, i = (ushort*)input.DataPointer;
            Walk(g, (d, s) => o[d] = BFloat16Bits.FromFloat(BFloat16Bits.ToFloat(o[d]) + BFloat16Bits.ToFloat(i[s])));
        }
    }

    /// <summary>Fills <paramref name="output"/> from <paramref name="input"/> with the padded positions zero, edge-replicated or reflected per the spec.</summary>
    public static void Pad(Tensor output, Tensor input, in PadSpec spec)
    {
        PadGeometry g = PadContract.Validate(output, input, spec);
        void* o = output.DataPointer;
        void* i = input.DataPointer;
        switch (g.ElementSize)
        {
            case 1: Pad((byte*)o, (byte*)i, g); break;
            case 2: Pad((ushort*)o, (ushort*)i, g); break;
            case 4: Pad((uint*)o, (uint*)i, g); break;
            default: Pad((ulong*)o, (ulong*)i, g); break;
        }
    }

    /// <summary>The input position an output position maps to on one axis, or -1 for a zero-filled one.</summary>
    public static long SourceIndex(long outputIndex, long before, long extent, PadMode mode)
    {
        long index = outputIndex - before;
        if (index >= 0 && index < extent) return index;
        switch (mode)
        {
            case PadMode.Replicate:
                return index < 0 ? 0 : extent - 1;
            case PadMode.Reflect:
                return index < 0 ? -index : 2 * extent - 2 - index;
            default:
                return -1;
        }
    }

    private static void Copy<T>(T* output, T* input, in StridedCopyGeometry g) where T : unmanaged
        => Walk(g, (d, s) => output[d] = input[s]);

    private static void Pad<T>(T* output, T* input, in PadGeometry g) where T : unmanaged
    {
        int rank = g.Rank;
        Span<long> index = stackalloc long[rank];
        Span<long> inputStrides = stackalloc long[rank];
        long stride = 1;
        for (int axis = rank - 1; axis >= 0; axis--)
        {
            inputStrides[axis] = stride;
            stride *= g.InputDim(axis);
        }
        for (long n = 0; n < g.ElementCount; n++)
        {
            long source = 0;
            bool zero = false;
            for (int axis = 0; axis < rank && !zero; axis++)
            {
                long s = SourceIndex(index[axis], g.Before(axis), g.InputDim(axis), g.Mode);
                if (s < 0) zero = true;
                else source += s * inputStrides[axis];
            }
            output[n] = zero ? default : input[source];
            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (++index[axis] < g.OutputDim(axis)) break;
                index[axis] = 0;
            }
        }
    }

    /// <summary>Visits every (destination, source) element offset pair of the geometry in row-major order of its extents.</summary>
    private static void Walk(in StridedCopyGeometry g, Action<long, long> visit)
    {
        int rank = g.Rank;
        Span<long> index = stackalloc long[rank];
        long d = g.DestinationOffset, s = g.SourceOffset;
        for (long n = 0; n < g.ElementCount; n++)
        {
            visit(d, s);
            for (int axis = rank - 1; axis >= 0; axis--)
            {
                index[axis]++;
                d += g.DestinationStride(axis);
                s += g.SourceStride(axis);
                if (index[axis] < g.Dim(axis)) break;
                d -= g.DestinationStride(axis) * g.Dim(axis);
                s -= g.SourceStride(axis) * g.Dim(axis);
                index[axis] = 0;
            }
        }
    }
}
