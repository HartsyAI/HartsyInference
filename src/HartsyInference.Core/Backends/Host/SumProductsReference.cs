using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the sum-of-products contract (<c>docs/BACKEND_OPS.md</c>): the products of F32 elements, exact in F64, added in F64 in
/// the order of the view. The <see cref="IBackend"/> default and the oracle every kernel is checked against.</summary>
public static unsafe class SumProductsReference
{
    public static void Apply(Tensor output, Tensor a, Tensor b, in SumProductsSpec spec)
    {
        SumProductsGeometry g = SumProductsContract.Validate(output, a, b, spec);
        float* pA = (float*)a.DataPointer, pB = (float*)b.DataPointer;
        void* pOut = output.DataPointer;
        bool wide = output.DType == DType.F64;
        int rank = spec.A.Rank, kept = spec.KeptAxes, inner = rank - 1;
        long run = spec.A[inner], stepA = spec.A.Stride(inner), stepB = spec.B.Stride(inner);
        Span<long> index = stackalloc long[StridedView.MaxRank];
        index.Clear();

        long at = 0;
        for (long group = 0; group < g.Groups; group++)
        {
            double dot = 0, squaredA = 0, squaredB = 0;
            for (long done = 0; done < g.ElementsPerGroup; done += run)
            {
                long ia = spec.A.Offset, ib = spec.B.Offset;
                for (int axis = 0; axis < inner; axis++)
                {
                    ia += index[axis] * spec.A.Stride(axis);
                    ib += index[axis] * spec.B.Stride(axis);
                }
                for (long i = 0; i < run; i++, ia += stepA, ib += stepB)
                {
                    double x = pA[ia], y = pB[ib];
                    dot += x * y;
                    squaredA += x * x;
                    squaredB += y * y;
                }
                for (int axis = inner - 1; axis >= 0; axis--)
                {
                    if (++index[axis] < spec.A[axis]) break;
                    index[axis] = 0;
                }
            }
            if ((spec.Terms & ProductTerms.Dot) != 0) Store(pOut, at++, dot, wide);
            if ((spec.Terms & ProductTerms.SquaredA) != 0) Store(pOut, at++, squaredA, wide);
            if ((spec.Terms & ProductTerms.SquaredB) != 0) Store(pOut, at++, squaredB, wide);
        }
    }

    private static void Store(void* output, long at, double value, bool wide)
    {
        if (wide) ((double*)output)[at] = value;
        else ((float*)output)[at] = (float)value;
    }
}
