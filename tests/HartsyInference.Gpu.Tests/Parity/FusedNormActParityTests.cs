using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Gpu.Tests.Parity;

/// <summary>The fused DiT ops (<see cref="IBackend.SituGlu"/>, <see cref="IBackend.RmsNormModulate"/>) and the
/// Gated-DeltaNet prefill (<see cref="IBackend.SsmConvSplitPrefill"/>, <see cref="IBackend.SsmDeltaPrefill"/>) on every
/// GPU backend against their Core host references: the mixed F32-in / F16-out modulate a DiT block feeds its F16 branch
/// from, inputs wide enough to saturate SiTU's soft caps, and a prefill split into chunks that must carry the conv and
/// recurrent state across the call boundary.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class FusedNormActParityTests(ITestOutputHelper output)
{
    public static IEnumerable<object[]> SituCases()
    {
        foreach (object[] kind in BackendGate.GpuKinds)
        {
            yield return [kind[0], 3, 40, false];
            yield return [kind[0], 5, 1000, false];
            yield return [kind[0], 4, 96, true];
        }
    }

    public static IEnumerable<object[]> ModulateCases()
    {
        foreach (object[] kind in BackendGate.GpuKinds)
        {
            yield return [kind[0], false, false, 1];
            yield return [kind[0], false, true, 1];
            yield return [kind[0], true, true, 1];
            yield return [kind[0], false, false, 2];
        }
    }

    public static IEnumerable<object[]> PrefillCases()
    {
        foreach (object[] kind in BackendGate.GpuKinds)
        {
            yield return [kind[0], 2, 4, 16, 4, 7, 5];
            yield return [kind[0], 1, 2, 128, 4, 9, 1];
        }
    }

    [Theory]
    [MemberData(nameof(SituCases))]
    public void SituGlu_MatchesReference(string kind, int rows, int inner, bool f16)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        DType dtype = f16 ? DType.F16 : DType.F32;
        using Tensor input = Random(new TensorShape(rows, 2 * inner), seed: 3, scale: 40f, dtype);
        using Tensor actual = new(new TensorShape(rows, inner), dtype);
        using Tensor expected = new(new TensorShape(rows, inner), dtype);
        backend.SituGlu(actual, input);
        SituGluReference.Apply(expected, input);
        TensorAssert.Close(actual, expected, rtol: f16 ? 2e-3f : 1e-5f, atol: f16 ? 2e-2f : 1e-5f, because: $"SituGlu on {kind}");
    }

    [Theory]
    [MemberData(nameof(ModulateCases))]
    public void RmsNormModulate_MatchesReference(string kind, bool inF16, bool outF16, int batches)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        const int seq = 6, dim = 3072;
        DType outType = outF16 ? DType.F16 : DType.F32;
        using Tensor input = Random(new TensorShape(batches, seq, dim), seed: 5, scale: 3f, inF16 ? DType.F16 : DType.F32);
        using Tensor weight = Random(new TensorShape(dim), seed: 6, scale: 1f, DType.F32);
        using Tensor scale = Random(new TensorShape(batches, dim), seed: 7, scale: 0.5f, DType.F32);
        using Tensor shift = Random(new TensorShape(batches, dim), seed: 8, scale: 0.5f, DType.F32);
        float rtol = outF16 ? 2e-3f : 1e-5f, atol = outF16 ? 2e-2f : 1e-5f;
        foreach (Tensor? maybeShift in new[] { shift, null })
        {
            using Tensor actual = new(input.Shape, outType);
            using Tensor expected = new(input.Shape, outType);
            backend.RmsNormModulate(actual, input, weight, scale, maybeShift, 1e-6f);
            RmsNormModulateReference.Apply(expected, input, weight, scale, maybeShift, 1e-6f);
            TensorAssert.Close(actual, expected, rtol, atol, because: $"RmsNormModulate on {kind}, shift={(maybeShift is null ? "none" : "set")}");
        }
    }

    /// <summary>A two-chunk prefill on the device against a one-shot prefill on the host: equal outputs prove the scan,
    /// equal final state and carry prove both survive the chunk boundary.</summary>
    [Theory]
    [MemberData(nameof(PrefillCases))]
    public void DeltaNetPrefill_InChunks_MatchesOneShotReference(string kind, int kHeads, int vHeads, int headDim, int kernel,
        int first, int second)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        DeltaNetCase c = new(kHeads, vHeads, headDim, kernel, first + second);
        using Tensor expected = new(new TensorShape(c.Seq, c.ValueDim), DType.F32);
        using Tensor hostHistory = c.InitialHistory();
        using Tensor hostState = c.InitialState();
        c.Run(null, 0, c.Seq, hostHistory, hostState, expected);

        using Tensor actual = new(new TensorShape(c.Seq, c.ValueDim), DType.F32);
        using Tensor deviceHistory = c.InitialHistory();
        using Tensor deviceState = c.InitialState();
        c.Run(backend, 0, first, deviceHistory, deviceState, actual);
        c.Run(backend, first, second, deviceHistory, deviceState, actual);

        TensorAssert.Close(actual, expected, rtol: 1e-4f, atol: 1e-5f, because: $"DeltaNet prefill output on {kind}");
        TensorAssert.Close(deviceState, hostState, rtol: 1e-4f, atol: 1e-5f, because: $"DeltaNet final state on {kind}");
        TensorAssert.Close(deviceHistory, hostHistory, rtol: 0f, atol: 0f, because: $"DeltaNet conv carry on {kind}");
        c.Dispose();
    }

    /// <summary>Random operands for one gated-DeltaNet layer over <c>Seq</c> tokens, with a decay that stays in (0, 1).</summary>
    private sealed class DeltaNetCase : IDisposable
    {
        private const float Eps = 1e-6f;
        private readonly int _kHeads, _vHeads, _headDim, _kernel;
        private readonly Tensor _mixed, _z, _alpha, _beta, _taps, _dtBias, _ssmA, _normW;

        public DeltaNetCase(int kHeads, int vHeads, int headDim, int kernel, int seq)
        {
            _kHeads = kHeads;
            _vHeads = vHeads;
            _headDim = headDim;
            _kernel = kernel;
            Seq = seq;
            int keyDim = kHeads * headDim;
            ValueDim = vHeads * headDim;
            ConvDim = 2 * keyDim + ValueDim;
            _mixed = Random(new TensorShape(seq, ConvDim), seed: 11, scale: 1f, DType.F32);
            _z = Random(new TensorShape(seq, ValueDim), seed: 12, scale: 1f, DType.F32);
            _alpha = Random(new TensorShape(seq, vHeads), seed: 13, scale: 1f, DType.F32);
            _beta = Random(new TensorShape(seq, vHeads), seed: 14, scale: 1f, DType.F32);
            _taps = Random(new TensorShape(ConvDim, kernel), seed: 15, scale: 0.5f, DType.F32);
            _dtBias = Random(new TensorShape(vHeads), seed: 16, scale: 0.5f, DType.F32);
            _ssmA = Random(new TensorShape(vHeads), seed: 17, scale: 1f, DType.F32);
            Span<float> decay = _ssmA.AsSpan<float>();
            for (int i = 0; i < decay.Length; i++) decay[i] = -MathF.Abs(decay[i]) - 0.1f;
            _normW = Random(new TensorShape(headDim), seed: 18, scale: 1f, DType.F32);
        }

        public int Seq { get; }

        public int ValueDim { get; }

        public int ConvDim { get; }

        public Tensor InitialHistory() => Random(new TensorShape((_kernel - 1) * ConvDim), seed: 19, scale: 1f, DType.F32);

        public Tensor InitialState() => Random(new TensorShape(_vHeads * _headDim * _headDim), seed: 20, scale: 0.1f, DType.F32);

        /// <summary>Runs tokens <c>[start, start+count)</c> through both prefill ops (the host reference when
        /// <paramref name="backend"/> is null) and writes the gated rows into <paramref name="output"/>.</summary>
        public void Run(IBackend? backend, int start, int count, Tensor history, Tensor state, Tensor output)
        {
            int keyDim = _kHeads * _headDim;
            float qScale = 1f / MathF.Sqrt(_headDim);
            using Tensor mixed = Rows(_mixed, start, count);
            using Tensor z = Rows(_z, start, count);
            using Tensor alpha = Rows(_alpha, start, count);
            using Tensor beta = Rows(_beta, start, count);
            using Tensor q = new(new TensorShape(count, keyDim), DType.F32);
            using Tensor k = new(new TensorShape(count, keyDim), DType.F32);
            using Tensor v = new(new TensorShape(count, ValueDim), DType.F32);
            using Tensor o = new(new TensorShape(count, ValueDim), DType.F32);
            int repeat = _vHeads / _kHeads;
            if (backend is null)
            {
                GatedDeltaNetReference.ConvSplit(q, k, v, mixed, history, _taps, _kernel, _kHeads, _headDim, _vHeads, _headDim, Eps, qScale);
                GatedDeltaNetReference.DeltaScan(o, state, q, k, v, z, alpha, beta, _dtBias, _ssmA, _normW, _vHeads, _headDim, _headDim, repeat, Eps);
            }
            else
            {
                backend.SsmConvSplitPrefill(q, k, v, mixed, history, _taps, _kernel, _kHeads, _headDim, _vHeads, _headDim, Eps, qScale);
                backend.SsmDeltaPrefill(o, state, q, k, v, z, alpha, beta, _dtBias, _ssmA, _normW, _vHeads, _headDim, _headDim, repeat, Eps);
            }
            o.AsReadOnlySpan<float>().CopyTo(output.AsSpan<float>().Slice(start * ValueDim, count * ValueDim));
        }

        public void Dispose()
        {
            foreach (Tensor t in new[] { _mixed, _z, _alpha, _beta, _taps, _dtBias, _ssmA, _normW }) t.Dispose();
        }

        private static Tensor Rows(Tensor source, int start, int count)
        {
            int width = (int)(source.ElementCount / source.Shape[0]);
            Tensor slice = new(new TensorShape(count, width), DType.F32);
            source.AsReadOnlySpan<float>().Slice(start * width, count * width).CopyTo(slice.AsSpan<float>());
            return slice;
        }
    }

    private static Tensor Random(TensorShape shape, int seed, float scale, DType dtype)
    {
        Tensor f32 = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> span = f32.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        if (dtype == DType.F32) return f32;
        Tensor cast = f32.CastTo(dtype);
        f32.Dispose();
        return cast;
    }
}
