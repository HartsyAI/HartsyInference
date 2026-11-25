using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;

namespace HartsyInference.Cuda;

// Sums of products over strided views (Kernels/reduction/sum_products.cu). The geometry is one by-value kernel parameter, so a launch captures into
// a graph without a device-side spec buffer.
public sealed partial class CudaKernels
{
    /// <summary>Most blocks one sum-of-products launch spans; the kernels grid-stride past it.</summary>
    private const uint SumProductsMaxBlocks = 1u << 20;

    /// <summary>Partial sums a large group is cut into, capped so one block adds them.</summary>
    internal const int SumProductsMaxPartials = 256;

    /// <summary>Elements a block sums on its own. A longer group takes partials: one block would walk it with too few threads.</summary>
    internal const long SumProductsBlockElements = 64L * BlockSize;

    private CudaModule? _sumProductsModule;
    private nint _sumProductsGroupsF32, _sumProductsPartialsF32, _sumProductsFinalizeF32;

    /// <summary>Kernel parameter of the sum-of-products kernels; mirrors <c>SumProductsArgs</c> in <c>sum_products.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct SumProductsArgs
    {
        public ulong Groups;
        public ulong PerGroup;
        public fixed long Dims[StridedView.MaxRank];
        public fixed long StrideA[StridedView.MaxRank];
        public fixed long StrideB[StridedView.MaxRank];
        public long OffsetA;
        public long OffsetB;
        public int Rank;
        public int Terms;
        public int Wide;
        public uint PartialsPerGroup;
    }

    private void LoadReductionKernels()
    {
        _sumProductsModule = LoadOwnedModule(Ptx("sum_products"));
        _sumProductsGroupsF32 = _sumProductsModule.GetFunction("sum_products_groups_f32");
        _sumProductsPartialsF32 = _sumProductsModule.GetFunction("sum_products_partials_f32");
        _sumProductsFinalizeF32 = _sumProductsModule.GetFunction("sum_products_finalize_f32");
    }

    /// <summary>Partial sums per group for groups of <paramref name="elementsPerGroup"/> elements; 1 means the groups need no scratch.</summary>
    internal static int SumProductsPartials(long elementsPerGroup) => elementsPerGroup <= SumProductsBlockElements
        ? 1 : (int)Math.Min(SumProductsMaxPartials, 1L + (elementsPerGroup - 1L) / SumProductsBlockElements);

    /// <summary>Bytes of scratch the partial sums take: three F64 per partial.</summary>
    internal static nuint SumProductsScratchBytes(long groups, int partials) => checked((nuint)(groups * partials * 3 * sizeof(double)));

    /// <summary>Sums the groups of <paramref name="g"/> into <paramref name="output"/>: one launch, or two through <paramref name="scratch"/> when
    /// <paramref name="partials"/> is above 1.</summary>
    public unsafe void LaunchSumProducts(ulong output, ulong a, ulong b, ulong scratch, int partials, in SumProductsGeometry g, ProductTerms terms,
        bool wide, nint stream)
    {
        SumProductsArgs args = default;
        args.Groups = (ulong)g.Groups;
        args.PerGroup = (ulong)g.ElementsPerGroup;
        args.Rank = g.Axes.Rank;
        for (int axis = 0; axis < g.Axes.Rank; axis++)
        {
            args.Dims[axis] = g.Axes.Dim(axis);
            args.StrideA[axis] = g.Axes.SourceStride(axis);
            args.StrideB[axis] = g.Axes.DestinationStride(axis);
        }
        args.OffsetA = g.Axes.SourceOffset;
        args.OffsetB = g.Axes.DestinationOffset;
        args.Terms = (int)terms;
        args.Wide = wide ? 1 : 0;
        args.PartialsPerGroup = (uint)partials;

        ulong outArg = output, aArg = a, bArg = b, scratchArg = scratch;
        uint groupBlocks = (uint)Math.Min(g.Groups, SumProductsMaxBlocks);
        if (partials == 1)
        {
            uint threads = 32;
            while (threads < (ulong)g.ElementsPerGroup && threads < BlockSize) threads <<= 1;
            void** direct = stackalloc void*[4];
            direct[0] = &outArg; direct[1] = &aArg; direct[2] = &bArg; direct[3] = &args;
            CudaDriverApi.cuLaunchKernel(_sumProductsGroupsF32, groupBlocks, 1, 1, threads, 1, 1, SharedBytes(threads), stream, (nint)direct, 0).ThrowOnError();
            return;
        }

        void** partial = stackalloc void*[4];
        partial[0] = &scratchArg; partial[1] = &aArg; partial[2] = &bArg; partial[3] = &args;
        uint partialBlocks = (uint)Math.Min(checked(g.Groups * partials), SumProductsMaxBlocks);
        CudaDriverApi.cuLaunchKernel(_sumProductsPartialsF32, partialBlocks, 1, 1, BlockSize, 1, 1, SharedBytes(BlockSize), stream, (nint)partial, 0).ThrowOnError();

        void** finalize = stackalloc void*[3];
        finalize[0] = &outArg; finalize[1] = &scratchArg; finalize[2] = &args;
        CudaDriverApi.cuLaunchKernel(_sumProductsFinalizeF32, groupBlocks, 1, 1, BlockSize, 1, 1, SharedBytes(BlockSize), stream, (nint)finalize, 0).ThrowOnError();

        static uint SharedBytes(uint threads) => 3 * threads * sizeof(double);
    }
}
