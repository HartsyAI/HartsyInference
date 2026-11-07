using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the norm-and-modulate contract (<c>docs/BACKEND_OPS.md</c>): for every row,
/// <c>y = (norm(x) · w + b) · scale + shift</c> with each absent term dropped, the norm's sums accumulated in F64 and every
/// product rounded once in F32. The <see cref="IBackend"/> default and the oracle every kernel is checked against.</summary>
public static unsafe class NormModulateReference
{
    public static void Apply(in NormModulateCall call)
    {
        NormModulateGeometry g = NormModulateContract.Validate(call);
        ModulationSpec mod = call.Modulation;
        float* w = call.Weight is null ? null : (float*)call.Weight.DataPointer;
        float* b = call.Bias is null ? null : (float*)call.Bias.DataPointer;
        float* scale = mod.Scale is null ? null : (float*)mod.Scale.DataPointer;
        float* shift = mod.Shift is null ? null : (float*)mod.Shift.DataPointer;
        int* index = mod.RowIndex is null ? null : (int*)mod.RowIndex.DataPointer;
        long modRows = index is null ? 0 : NormModulateContract.ModulationRowsFor(mod, g.Width);
        Func<long, float> load = Loader(call.Input);
        Action<long, float> store = Storer(call.Output);
        int n = g.Width;
        for (long r = 0; r < g.Rows; r++)
        {
            long at = g.RowBase(r);
            float factor = 0f, mean = 0f;
            if (call.Kind != NormKind.None)
            {
                double sum = 0, sumSq = 0;
                for (int d = 0; d < n; d++)
                {
                    double v = load(at + d * g.Spatial);
                    sum += v;
                    sumSq += v * v;
                }
                if (call.Kind == NormKind.LayerNorm)
                {
                    double rowMean = sum / n;
                    double variance = 0;
                    for (int d = 0; d < n; d++)
                    {
                        double diff = load(at + d * g.Spatial) - rowMean;
                        variance += diff * diff;
                    }
                    mean = (float)rowMean;
                    factor = (float)(1.0 / Math.Sqrt(variance / n + call.Eps));
                }
                else if (call.Kind == NormKind.RmsNorm)
                {
                    factor = (float)(1.0 / Math.Sqrt(sumSq / n + call.Eps));
                }
                else
                {
                    factor = MathF.Sqrt(n) / MathF.Max((float)Math.Sqrt(sumSq), call.Eps);
                }
            }
            long m = index is null ? g.ModulationRow(r) : index[r];
            if (index is not null && (m < 0 || m >= modRows))
                throw new ArgumentOutOfRangeException(nameof(call), m, $"Row {r} indexes modulation row {m} of {modRows}.");
            long scaleAt = g.ScaleOffset + m * g.RowStride, shiftAt = g.ShiftOffset + m * g.RowStride;
            for (int d = 0; d < n; d++)
            {
                long i = at + d * g.Spatial;
                float v = load(i);
                if (call.Kind == NormKind.LayerNorm) v = (v - mean) * factor;
                else if (call.Kind != NormKind.None) v *= factor;
                if (w != null) v *= w[d];
                if (b != null) v += b[d];
                if (scale != null) v *= mod.Mode == ScaleMode.OnePlus ? 1f + scale[scaleAt + d] : scale[scaleAt + d];
                if (shift != null) v += shift[shiftAt + d];
                store(i, v);
            }
        }
    }

    private static Func<long, float> Loader(Tensor t)
    {
        void* p = t.DataPointer;
        if (t.DType == DType.F32) return i => ((float*)p)[i];
        if (t.DType == DType.F16) return i => (float)((Half*)p)[i];
        return i => BFloat16Bits.ToFloat(((ushort*)p)[i]);
    }

    private static Action<long, float> Storer(Tensor t)
    {
        void* p = t.DataPointer;
        if (t.DType == DType.F32) return (i, v) => ((float*)p)[i] = v;
        if (t.DType == DType.F16) return (i, v) => ((Half*)p)[i] = (Half)v;
        return (i, v) => ((ushort*)p)[i] = BFloat16Bits.FromFloat(v);
    }
}
