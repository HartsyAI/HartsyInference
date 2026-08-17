using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Pre-norm neighborhood-attention + SwiGLU block of the LTX-2.5 video decoder, covering both variants: the
/// deterministic stage-1..4 <c>NABlock</c> (<c>x += attn(norm1(x))</c>, <c>x += mlp(norm2(x))</c>) and the stage-5
/// <c>DiffusionNABlock</c>, which additionally adds a projection of the stage-1..4 context and modulates each norm by
/// AdaLN scale/shift. Both residual adds are ungated in either variant — the AdaLN gate slots (2, 5, 6) are unused,
/// folded away at export.</summary>
internal sealed class LtxVideo25NaBlock
{
    private const int ModulationChunks = 7;
    private const int ScaleMsaChunk = 0, ShiftMsaChunk = 1, ScaleMlpChunk = 3, ShiftMlpChunk = 4;

    private readonly int _dim;
    private readonly int _hidden;
    private readonly int _kernelT;
    private readonly float _eps;
    private readonly bool _isDiffusion;
    private readonly LtxVideo25NeighborhoodAttention3d _attention;

    /// <summary>Per-instance bisect tap (attention internals + block stages); set via
    /// <see cref="SetInnerTap"/> so the nested attention's tap stays in lockstep.</summary>
    internal Action<string, Tensor>? Tap { get; private set; }

    internal void SetInnerTap(Action<string, Tensor>? tap)
    {
        Tap = tap;
        _attention.Tap = tap;
    }

    private Tensor? _norm1Weight, _norm2Weight;
    private Tensor? _gateWeight, _upWeight, _downWeight;
    private Tensor? _contextProjWeight, _contextProjBias, _scaleShiftTable;

    public LtxVideo25NaBlock(int dim, (int T, int H, int W) kernel, LtxVideo25DiffusionDecoderConfig config, bool isDiffusion)
    {
        _dim = dim;
        _hidden = LtxVideo25DiffusionDecoderConfig.SwiGluHidden(dim);
        _kernelT = kernel.T;
        _eps = config.NormEps;
        _isDiffusion = isDiffusion;
        _attention = new LtxVideo25NeighborhoodAttention3d(dim, kernel, config);
    }

    public void LoadWeights(LtxVideo25WeightScope scope, string prefix)
    {
        _norm1Weight = scope.F32($"{prefix}.norm1.weight");
        _norm2Weight = scope.F32($"{prefix}.norm2.weight");
        _attention.LoadWeights(scope, $"{prefix}.attn");
        _gateWeight = scope.Raw($"{prefix}.mlp.w_gate.weight");
        _upWeight = scope.Raw($"{prefix}.mlp.w_up.weight");
        _downWeight = scope.Raw($"{prefix}.mlp.w_down.weight");
        if (_gateWeight.Shape[0] != _hidden || _gateWeight.Shape[1] != _dim)
            throw new InvalidOperationException($"'{prefix}.mlp.w_gate.weight' is {_gateWeight.Shape}, expected [{_hidden}, {_dim}].");
        if (!_isDiffusion) return;
        _contextProjWeight = scope.Raw($"{prefix}.context_proj.weight");
        _contextProjBias = scope.OptionalF32($"{prefix}.context_proj.bias");
        _scaleShiftTable = scope.F32($"{prefix}.scale_shift_table");
        if (_scaleShiftTable.ElementCount != (long)ModulationChunks * _dim)
            throw new InvalidOperationException($"'{prefix}.scale_shift_table' has {_scaleShiftTable.ElementCount} entries, expected {ModulationChunks * _dim}.");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _attention.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _norm1Weight, _norm2Weight, _gateWeight, _upWeight, _downWeight,
                                      _contextProjWeight, _contextProjBias, _scaleShiftTable })
        {
            if (t is not null) yield return t;
        }
    }

    /// <summary>Updates <paramref name="x"/> <c>[t·h·w, dim]</c> in place. <paramref name="context"/> and
    /// <paramref name="modulation"/> (<c>[7, dim]</c> AdaLN chunks, before this block's own table is added) are
    /// required for a diffusion block and ignored by a deterministic one.</summary>
    /// <param name="chunkBytes">Per-chunk activation budget; see <see cref="LtxVideo25TemporalChunks"/>. The three
    /// sub-passes run to completion one after another rather than fused per chunk, because the attention's halo
    /// reads rows that a fused loop would already have written a later pass's residual into.</param>
    public void Forward(IBackend backend, Tensor x, Tensor? context, Tensor? modulation, int t, int h, int w,
        long chunkBytes)
    {
        if (_isDiffusion && (context is null || modulation is null))
            throw new ArgumentNullException(nameof(context), "A diffusion block needs both the latent context and the AdaLN modulation.");
        long plane = (long)h * w;
        if (_isDiffusion) AddContextPass(backend, x, context!, t, plane, chunkBytes);
        Modulation msa = _isDiffusion ? Modulation.Build(modulation!, _scaleShiftTable!, ScaleMsaChunk, ShiftMsaChunk, _dim) : default;
        Modulation mlp = _isDiffusion ? Modulation.Build(modulation!, _scaleShiftTable!, ScaleMlpChunk, ShiftMlpChunk, _dim) : default;
        try
        {
            AttentionPass(backend, x, msa, t, h, w, plane, chunkBytes);
            Tap?.Invoke("blk_resid", x);
            MlpPass(backend, x, mlp, t, plane, chunkBytes);
        }
        finally { msa.Dispose(); mlp.Dispose(); }
    }

    /// <summary>One AdaLN slot's <c>1 + scale</c> and <c>shift</c> as <c>[dim]</c> rows, ready to broadcast over every
    /// token. Built once per block call rather than per chunk: the sum is over the shared modulation and this block's
    /// table, neither of which varies across chunks.</summary>
    private readonly record struct Modulation(Tensor? ScalePlus1, Tensor? Shift)
    {
        public static unsafe Modulation Build(Tensor modulation, Tensor scaleShiftTable, int scaleChunk, int shiftChunk, int dim)
        {
            Tensor scalePlus1 = new Tensor(new TensorShape(dim), DType.F32);
            Tensor shift = new Tensor(new TensorShape(dim), DType.F32);
            float* mod = (float*)modulation.DataPointer;
            float* table = (float*)scaleShiftTable.DataPointer;
            float* s = (float*)scalePlus1.DataPointer;
            float* b = (float*)shift.DataPointer;
            for (int i = 0; i < dim; i++)
            {
                s[i] = 1f + (mod[scaleChunk * dim + i] + table[scaleChunk * dim + i]);
                b[i] = mod[shiftChunk * dim + i] + table[shiftChunk * dim + i];
            }
            return new Modulation(scalePlus1, shift);
        }

        public void Dispose() { ScalePlus1?.Dispose(); Shift?.Dispose(); }
    }

    private void AddContextPass(IBackend backend, Tensor x, Tensor context, int frames, long plane, long chunkBytes)
    {
        int step = LtxVideo25TemporalChunks.FramesPerChunk(chunkBytes, plane, 3L * _dim * sizeof(float), frames);
        for (int start = 0; start < frames; start += step)
        {
            int count = Math.Min(step, frames - start);
            int rowOffset = checked((int)((long)start * plane));
            TensorShape shape = new TensorShape((long)count * plane, _dim);
            using Tensor slice = new Tensor(shape, DType.F32);
            using Tensor projected = new Tensor(shape, DType.F32);
            backend.SliceRowsGeneric(slice, context, rowOffset);
            backend.Linear(projected, slice, _contextProjWeight!, _contextProjBias);
            backend.SliceRowsGeneric(slice, x, rowOffset);
            backend.Add(slice, slice, projected);
            backend.ScatterRowsGeneric(x, slice, rowOffset);
        }
    }

    /// <summary>Attention residual, one halo-padded temporal chunk at a time. A chunk's residual write is deferred
    /// until no still-unprocessed chunk's halo covers it, so every chunk reads the same pre-attention <c>x</c> a
    /// single untiled pass would have read.</summary>
    private void AttentionPass(IBackend backend, Tensor x, Modulation modulation, int frames, int h, int w,
        long plane, long chunkBytes)
    {
        // Live at the attention's peak: the normed window, the fused qkv, and q/k/v.
        long bytesPerToken = 7L * _dim * sizeof(float);
        int step = LtxVideo25TemporalChunks.AttentionFramesPerChunk(chunkBytes, plane, bytesPerToken, frames, _kernelT);
        List<(int Start, int End, Tensor Delta)> pending = [];
        for (int start = 0; start < frames; start += step)
        {
            int end = Math.Min(start + step, frames);
            (int windowStart, int windowFrames) = LtxVideo25TemporalChunks.Window(start, end, frames, _kernelT);
            FlushAttention(backend, x, pending, windowStart, plane);
            long windowRows = (long)windowFrames * plane;
            Tensor delta;
            using (Tensor normed = new Tensor(new TensorShape(windowRows, _dim), DType.F32))
            {
                backend.SliceRowsGeneric(normed, x, checked((int)((long)windowStart * plane)));
                backend.RmsNorm(normed, normed, _norm1Weight!, _eps);
                if (_isDiffusion) backend.AffineBroadcastLastDim(normed, normed, modulation.ScalePlus1!, modulation.Shift!);
                delta = _attention.Forward(backend, normed, windowFrames, h, w, windowStart);
            }
            if (start != windowStart || end - start != windowFrames)
            {
                Tensor cropped = new Tensor(new TensorShape((long)(end - start) * plane, _dim), DType.F32);
                backend.SliceRowsGeneric(cropped, delta, checked((int)((long)(start - windowStart) * plane)));
                delta.Dispose();
                delta = cropped;
            }
            pending.Add((start, end, delta));
        }
        FlushAttention(backend, x, pending, frames, plane);
    }

    /// <summary>Adds every pending chunk that ends at or before <paramref name="upTo"/> into <paramref name="x"/>.</summary>
    private void FlushAttention(IBackend backend, Tensor x, List<(int Start, int End, Tensor Delta)> pending,
        int upTo, long plane)
    {
        int kept = 0;
        for (int i = 0; i < pending.Count; i++)
        {
            (int start, int end, Tensor delta) = pending[i];
            if (end > upTo)
            {
                pending[kept++] = pending[i];
                continue;
            }
            int rowOffset = checked((int)((long)start * plane));
            using (Tensor slice = new Tensor(new TensorShape((long)(end - start) * plane, _dim), DType.F32))
            {
                backend.SliceRowsGeneric(slice, x, rowOffset);
                backend.Add(slice, slice, delta);
                backend.ScatterRowsGeneric(x, slice, rowOffset);
            }
            delta.Dispose();
        }
        pending.RemoveRange(kept, pending.Count - kept);
    }

    private void MlpPass(IBackend backend, Tensor x, Modulation modulation, int frames, long plane, long chunkBytes)
    {
        long bytesPerToken = (3L * _dim + 2L * _hidden) * sizeof(float);
        int step = LtxVideo25TemporalChunks.FramesPerChunk(chunkBytes, plane, bytesPerToken, frames);
        for (int start = 0; start < frames; start += step)
        {
            int count = Math.Min(step, frames - start);
            int rowOffset = checked((int)((long)start * plane));
            long rows = (long)count * plane;
            TensorShape shape = new TensorShape(rows, _dim);
            using Tensor slice = new Tensor(shape, DType.F32);
            using Tensor normed = new Tensor(shape, DType.F32);
            backend.SliceRowsGeneric(slice, x, rowOffset);
            backend.RmsNorm(normed, slice, _norm2Weight!, _eps);
            if (_isDiffusion) backend.AffineBroadcastLastDim(normed, normed, modulation.ScalePlus1!, modulation.Shift!);
            // The layer-diff harness compares whole-tensor dumps, so these taps only carry a full tensor when the
            // geometry it runs at leaves the pass un-chunked.
            if (count == frames) Tap?.Invoke("blk_norm2", normed);
            using Tensor mlp = Feedforward(backend, normed, rows);
            if (count == frames) Tap?.Invoke("blk_mlp", mlp);
            backend.Add(slice, slice, mlp);
            backend.ScatterRowsGeneric(x, slice, rowOffset);
        }
    }

    private Tensor Feedforward(IBackend backend, Tensor x, long tokens)
    {
        using Tensor gate = new Tensor(new TensorShape(tokens, _hidden), DType.F32);
        backend.Linear(gate, x, _gateWeight!, null);
        backend.Silu(gate, gate);
        using (Tensor up = new Tensor(new TensorShape(tokens, _hidden), DType.F32))
        {
            backend.Linear(up, x, _upWeight!, null);
            backend.Mul(gate, gate, up);
        }
        Tensor result = new Tensor(new TensorShape(tokens, _dim), DType.F32);
        backend.Linear(result, gate, _downWeight!, null);
        return result;
    }
}
