using System.Runtime.InteropServices;
using HartsyInference.Core.Backends.Guidance;

namespace HartsyInference.Cuda;

// The guided Euler step (Kernels/guidance/guided_euler.cu) and the guided top-K logits (Kernels/guidance/topk_guided_logits.cu):
// by-value parameters only, so every launch captures into a graph.
public sealed partial class CudaKernels
{
    /// <summary>Most blocks a guided Euler launch spans; the kernels grid-stride past it.</summary>
    private const uint GuidedEulerMaxBlocks = 1u << 20;

    /// <summary>Partial sums the global norm reduces into: one block each, capped so the finalize stays one block.</summary>
    public const int GuidedEulerMaxPartials = 256;

    /// <summary>Widest logit row the guided top-K kernel masks in shared memory.</summary>
    public const int TopKGuidedMaxWidth = 16448;

    private CudaModule? _guidedEulerModule;
    private CudaModule? _topKGuidedModule;
    private nint _guidedEulerF32, _guidedEulerReduceF32, _guidedEulerFinalizeF32, _guidedEulerScaledF32, _guidedEulerRowsF32;
    private nint _topKGuidedLogitsF32;

    /// <summary>Kernel parameter of the guided Euler kernels; mirrors <c>GuidedEulerArgs</c> in <c>guided_euler.cu</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct GuidedEulerArgs
    {
        public float Guidance;
        public float Delta;
        public int Form;
        public int Reserved;
        public ulong Count;
    }

    /// <summary>Kernel parameter of the norm ratio; mirrors <c>GuidanceNormArgs</c> field for field.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct GuidanceNormArgs
    {
        public float Min;
        public float Max;
        public float Eps;
        public uint PartialCount;
    }

    private void LoadGuidanceKernels()
    {
        _guidedEulerModule = LoadOwnedModule(Ptx("guided_euler"));
        _guidedEulerF32 = _guidedEulerModule.GetFunction("guided_euler_f32");
        _guidedEulerReduceF32 = _guidedEulerModule.GetFunction("guided_euler_reduce_f32");
        _guidedEulerFinalizeF32 = _guidedEulerModule.GetFunction("guided_euler_finalize_f32");
        _guidedEulerScaledF32 = _guidedEulerModule.GetFunction("guided_euler_scaled_f32");
        _guidedEulerRowsF32 = _guidedEulerModule.GetFunction("guided_euler_rows_f32");
        _topKGuidedModule = LoadOwnedModule(Ptx("topk_guided_logits"));
        _topKGuidedLogitsF32 = _topKGuidedModule.GetFunction("topk_guided_logits_f32");
    }

    private static GuidedEulerArgs EulerArgs(in GuidedEulerSpec spec, long count)
        => new() { Guidance = spec.Guidance, Delta = spec.Delta, Form = (int)spec.Form, Count = (ulong)count };

    private static GuidanceNormArgs NormArgs(in GuidanceNorm norm, int partialCount)
        => new() { Min = norm.Min, Max = norm.Max, Eps = norm.Eps, PartialCount = (uint)partialCount };

    /// <summary>The unnormalized step over <paramref name="count"/> elements: one launch.</summary>
    public unsafe void LaunchGuidedEuler(ulong z, ulong cond, ulong uncond, in GuidedEulerSpec spec, long count, nint stream)
    {
        GuidedEulerArgs args = EulerArgs(spec, count);
        ulong zArg = z, condArg = cond, uncondArg = uncond;
        void** a = stackalloc void*[4];
        a[0] = &zArg; a[1] = &condArg; a[2] = &uncondArg; a[3] = &args;
        uint grid = (uint)Math.Min((count + BlockSize - 1) / BlockSize, GuidedEulerMaxBlocks);
        CudaDriverApi.cuLaunchKernel(_guidedEulerF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Blocks the global norm's reduction uses for <paramref name="count"/> elements, which is also its partial count.</summary>
    public static int GuidedEulerPartials(long count) => (int)Math.Min(GuidedEulerMaxPartials, 1L + (count - 1L) / BlockSize);

    /// <summary>Bytes of scratch the global norm needs: two F64 partial arrays and one F32 ratio.</summary>
    public static nuint GuidedEulerScratchBytes(int partialCount) => checked((nuint)(2L * partialCount * sizeof(double) + sizeof(float)));

    /// <summary>The globally normalized step: the partial sums, the ratio and the scaled update, three launches ordered on <paramref name="stream"/>.</summary>
    public unsafe void LaunchGuidedEulerGlobal(ulong z, ulong cond, ulong uncond, ulong scratch, int partialCount, in GuidedEulerSpec spec, long count, nint stream)
    {
        ulong condPartials = scratch;
        ulong guidedPartials = scratch + (ulong)partialCount * sizeof(double);
        ulong scale = guidedPartials + (ulong)partialCount * sizeof(double);
        GuidedEulerArgs args = EulerArgs(spec, count);
        GuidanceNormArgs norm = NormArgs(spec.Norm, partialCount);
        ulong zArg = z, condArg = cond, uncondArg = uncond, condPartialsArg = condPartials, guidedPartialsArg = guidedPartials, scaleArg = scale;
        uint sharedBytes = 2 * BlockSize * sizeof(double);

        void** reduce = stackalloc void*[5];
        reduce[0] = &condArg; reduce[1] = &uncondArg; reduce[2] = &condPartialsArg; reduce[3] = &guidedPartialsArg; reduce[4] = &args;
        CudaDriverApi.cuLaunchKernel(_guidedEulerReduceF32, (uint)partialCount, 1, 1, BlockSize, 1, 1, sharedBytes, stream, (nint)reduce, 0).ThrowOnError();

        void** finalize = stackalloc void*[4];
        finalize[0] = &scaleArg; finalize[1] = &condPartialsArg; finalize[2] = &guidedPartialsArg; finalize[3] = &norm;
        CudaDriverApi.cuLaunchKernel(_guidedEulerFinalizeF32, 1, 1, 1, BlockSize, 1, 1, sharedBytes, stream, (nint)finalize, 0).ThrowOnError();

        void** update = stackalloc void*[5];
        update[0] = &zArg; update[1] = &condArg; update[2] = &uncondArg; update[3] = &scaleArg; update[4] = &args;
        CudaDriverApi.cuLaunchKernel(_guidedEulerScaledF32, (uint)partialCount, 1, 1, BlockSize, 1, 1, 0, stream, (nint)update, 0).ThrowOnError();
    }

    /// <summary>The per-row normalized step: one block reduces and updates each last-dimension row.</summary>
    public unsafe void LaunchGuidedEulerRows(ulong z, ulong cond, ulong uncond, in GuidedEulerSpec spec, long rows, long lastDim, nint stream)
    {
        GuidedEulerArgs args = EulerArgs(spec, rows);
        GuidanceNormArgs norm = NormArgs(spec.Norm, 0);
        ulong zArg = z, condArg = cond, uncondArg = uncond, lastDimArg = (ulong)lastDim;
        void** a = stackalloc void*[6];
        a[0] = &zArg; a[1] = &condArg; a[2] = &uncondArg; a[3] = &args; a[4] = &lastDimArg; a[5] = &norm;
        uint threads = 32;
        while (threads < (ulong)lastDim && threads < BlockSize) threads <<= 1;
        uint grid = (uint)Math.Min(rows, 65_535L);
        CudaDriverApi.cuLaunchKernel(_guidedEulerRowsF32, grid, 1, 1, threads, 1, 1, 2 * threads * sizeof(double), stream, (nint)a, 0).ThrowOnError();
    }

    /// <summary>Guided top-K logits, one block per row; a row holds at most <see cref="TopKGuidedMaxWidth"/> entries.</summary>
    public unsafe void LaunchTopKGuidedLogits(ulong output, ulong cond, ulong uncond, long rows, int width, int topK, float guidance, nint stream)
    {
        if (width > TopKGuidedMaxWidth)
            throw new NotSupportedException($"TopKGuidedLogits rows hold at most {TopKGuidedMaxWidth} entries on CUDA; got {width}.");
        ulong outArg = output, condArg = cond, uncondArg = uncond;
        int widthArg = width, topKArg = topK;
        float guidanceArg = guidance;
        void** a = stackalloc void*[6];
        a[0] = &outArg; a[1] = &condArg; a[2] = &uncondArg; a[3] = &widthArg; a[4] = &topKArg; a[5] = &guidanceArg;
        CudaDriverApi.cuLaunchKernel(_topKGuidedLogitsF32, (uint)rows, 1, 1, 256, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
