using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Firefly-gan-vq <c>DownsampleFiniteScalarQuantize</c> (Fish-Speech 1.5). A grouped-<b>residual</b> FSQ
/// (per-book levels <c>[8,5,5,5]</c>, 9 residual quantizers) wrapped in a ConvNeXt down/up-sample stack
/// (<c>downsample_factor [2,2]</c>): the encoder strides the 1024-dim hidden down by 4 into the FSQ digit space,
/// and the decode path runs the inverse — <c>indices → residual-FSQ dequant (sum stages) → fc_post → upsample
/// convs → [1, 1024, T*4]</c>. That 1024-dim tensor is the input to the firefly HiFi-GAN generator.
///
/// <para>Reuses <see cref="Fsq.Dequantize"/> for each residual stage. The residual FSQ has a learned per-stage
/// <c>project_in</c>/<c>project_out</c> (codebook_dim ↔ 4) collapsed here into the standard FSQ inverse since
/// firefly's FSQ uses <c>dim == sum(levels-arity)</c> with identity projections; any per-stage projection weight
/// is applied if present in the checkpoint.</para></summary>
public sealed unsafe class FireflyQuantizer : IDisposable
{
    private const int OutputDim = 1_024;
    private readonly int[] _levels;
    private readonly int _numQuantizers;
    private readonly int _codebookDim;
    private readonly int[] _upsampleFactors;
    private readonly int _upTotal;

    private readonly Tensor?[] _fsqProjInW, _fsqProjInB;       // optional per-stage residual-FSQ in projection
    private readonly Tensor?[] _fsqProjOutW, _fsqProjOutB;     // optional per-stage residual-FSQ out projection
    private Tensor? _fcPostW, _fcPostB;                        // residual-FSQ summed code → upsample-input channels
    private readonly Tensor?[] _upW, _upB;                     // ConvTranspose1d upsample stack
    private readonly Tensor?[] _upResW, _upResB;               // post-upsample ConvNeXt depthwise smoothing conv
    private int _disposed;

    public FireflyQuantizer(int[]? levels = null, int numQuantizers = 9, int[]? upsampleFactors = null,
        int codebookDim = 0)
    {
        _levels = levels ?? [8, 5, 5, 5];
        _numQuantizers = numQuantizers;
        _codebookDim = codebookDim > 0 ? codebookDim : _levels.Length;
        _upsampleFactors = upsampleFactors ?? [2, 2];
        _upTotal = 1;
        for (int i = 0; i < _upsampleFactors.Length; i++) _upTotal *= _upsampleFactors[i];
        _fsqProjInW = new Tensor?[_numQuantizers]; _fsqProjInB = new Tensor?[_numQuantizers];
        _fsqProjOutW = new Tensor?[_numQuantizers]; _fsqProjOutB = new Tensor?[_numQuantizers];
        _upW = new Tensor?[_upsampleFactors.Length]; _upB = new Tensor?[_upsampleFactors.Length];
        _upResW = new Tensor?[_upsampleFactors.Length]; _upResB = new Tensor?[_upsampleFactors.Length];
    }

    /// <summary>Total upsample factor of the ConvNeXt stack (frames in → frames out multiplier). 4 for firefly.</summary>
    public int UpsampleFactor => _upTotal;

    /// <summary>Per-book FSQ vocabulary size (<c>product(levels)</c>). 1000 for firefly.</summary>
    public int CodebookSize => Fsq.VocabSize(_levels);

    /// <summary>Number of residual quantizers (codebooks). 9 for firefly.</summary>
    public int NumQuantizers => _numQuantizers;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "quantizer")
    {
        for (int i = 0; i < _numQuantizers; i++)
        {
            string pin = $"{prefix}.residual_fsq.rvqs.0.layers.{i}.project_in";
            if (w.ContainsKey($"{pin}.weight"))
            {
                _fsqProjInW[i] = WhisperOps.EnsureF32(w[$"{pin}.weight"]);
                if (w.TryGetValue($"{pin}.bias", out Tensor? bin)) _fsqProjInB[i] = WhisperOps.EnsureF32(bin);
            }
            string pout = $"{prefix}.residual_fsq.rvqs.0.layers.{i}.project_out";
            if (w.ContainsKey($"{pout}.weight"))
            {
                _fsqProjOutW[i] = WhisperOps.EnsureF32(w[$"{pout}.weight"]);
                if (w.TryGetValue($"{pout}.bias", out Tensor? bout)) _fsqProjOutB[i] = WhisperOps.EnsureF32(bout);
            }
        }
        if (w.ContainsKey($"{prefix}.fc_post.weight"))
        {
            _fcPostW = WhisperOps.EnsureF32(w[$"{prefix}.fc_post.weight"]);
            if (w.TryGetValue($"{prefix}.fc_post.bias", out Tensor? fpb)) _fcPostB = WhisperOps.EnsureF32(fpb);
        }
        for (int i = 0; i < _upsampleFactors.Length; i++)
        {
            string up = $"{prefix}.upsample.{i}";
            if (w.ContainsKey($"{up}.weight")) { _upW[i] = WhisperOps.EnsureF32(w[$"{up}.weight"]); }
            else if (w.ContainsKey($"{up}.0.weight")) { _upW[i] = WhisperOps.EnsureF32(w[$"{up}.0.weight"]); }
            if (w.TryGetValue($"{up}.bias", out Tensor? ub)) _upB[i] = WhisperOps.EnsureF32(ub);
            else if (w.TryGetValue($"{up}.0.bias", out Tensor? ub0)) _upB[i] = WhisperOps.EnsureF32(ub0);
            string ur = $"{prefix}.upsample_conv.{i}";
            if (w.ContainsKey($"{ur}.weight"))
            {
                _upResW[i] = WhisperOps.EnsureF32(w[$"{ur}.weight"]);
                if (w.TryGetValue($"{ur}.bias", out Tensor? urb)) _upResB[i] = WhisperOps.EnsureF32(urb);
            }
        }
    }

    /// <summary>Dequantizes residual-FSQ <c>codes</c> <c>[numQuantizers, T]</c> and runs the ConvNeXt upsample stack
    /// to produce the firefly HiFi-GAN decoder input <c>[1, 1024, T * upsampleFactor]</c>.</summary>
    public Tensor DequantToDecoderInput(IBackend backend, int[,] codes, int t)
    {
        if (codes.GetLength(0) != _numQuantizers)
            throw new ArgumentException($"codes must have {_numQuantizers} rows, got {codes.GetLength(0)}.");
        if (t <= 0) throw new ArgumentException("t must be positive.", nameof(t));

        // residual-FSQ get_output_from_indices: sum each stage's dequantized vector → [1, T, codebookDim].
        int d = _levels.Length;
        Tensor summed = new(new TensorShape(1, t, _codebookDim), DType.F32);
        float* sp = (float*)summed.DataPointer;
        for (long n = 0; n < summed.ElementCount; n++) sp[n] = 0f;

        Tensor stageCodes = new(new TensorShape(1, t), DType.I32);
        int* scp = (int*)stageCodes.DataPointer;
        Tensor stageVec = new(new TensorShape(1, t, d), DType.F32);
        for (int q = 0; q < _numQuantizers; q++)
        {
            for (int j = 0; j < t; j++) scp[j] = codes[q, j];
            Fsq.Dequantize(stageVec, stageCodes, _levels);
            Tensor projected = ProjectStageOut(backend, stageVec, t);
            float* pvp = (float*)projected.DataPointer;
            for (long n = 0; n < (long)t * _codebookDim; n++) sp[n] += pvp[n];
            if (!ReferenceEquals(projected, stageVec)) projected.Dispose();
        }
        stageVec.Dispose();
        stageCodes.Dispose();

        // [1, T, codebookDim] → [1, codebookDim, T] for the channels-first conv stack.
        Tensor cf = new(new TensorShape(1, _codebookDim, t), DType.F32);
        backend.Transpose2D(cf, summed, t, _codebookDim);
        summed.Dispose();

        // fc_post: 1×1 conv codebookDim → upsample-input channels (the channel count of the first upsample weight).
        Tensor x = ApplyFcPost(backend, cf, t);
        cf.Dispose();

        int curT = t;
        int inCh = (int)x.Shape[1];
        for (int i = 0; i < _upsampleFactors.Length; i++)
        {
            int stride = _upsampleFactors[i];
            if (_upW[i] is null) throw new InvalidOperationException($"FireflyQuantizer upsample.{i} not loaded.");
            int kernel = (int)_upW[i]!.Shape[2];
            int outCh = (int)_upW[i]!.Shape[1];
            int pad = (kernel - stride) / 2;
            int outT = (curT - 1) * stride + (kernel - 1) + 1 - 2 * pad;
            Tensor up = new(new TensorShape(1, outCh, outT), DType.F32);
            backend.ConvTranspose1d(up, x, _upW[i]!, _upB[i], stride, pad, pad, 1, 1);
            x.Dispose();
            x = up;
            curT = outT;
            inCh = outCh;

            if (_upResW[i] is not null)
            {
                Tensor act = new(x.Shape, DType.F32);
                backend.Silu(act, x);
                int rk = (int)_upResW[i]!.Shape[2];
                int rpad = (rk - 1) / 2;
                int rOut = (int)_upResW[i]!.Shape[0];
                int rt = curT + 2 * rpad - (rk - 1);
                Tensor res = new(new TensorShape(1, rOut, rt), DType.F32);
                backend.Conv1d(res, act, _upResW[i]!, _upResB[i], 1, rpad, rpad, 1, 1);
                act.Dispose();
                if (rOut == inCh && rt == curT)
                {
                    float* xp = (float*)x.DataPointer; float* rp = (float*)res.DataPointer;
                    for (long n = 0; n < x.ElementCount; n++) xp[n] += rp[n];
                    res.Dispose();
                }
                else
                {
                    x.Dispose(); x = res; curT = rt; inCh = rOut;
                }
            }
        }

        // The HiFi-GAN expects 1024 input channels; if the upsample stack already lands there, return as-is.
        if (inCh == OutputDim) return x;

        // Otherwise channel-pad/truncate into a 1024-channel tensor (synthetic / partial checkpoints).
        Tensor outp = new(new TensorShape(1, OutputDim, curT), DType.F32);
        float* op = (float*)outp.DataPointer;
        for (long n = 0; n < outp.ElementCount; n++) op[n] = 0f;
        float* src = (float*)x.DataPointer;
        int copyCh = Math.Min(inCh, OutputDim);
        for (int c = 0; c < copyCh; c++)
            for (int j = 0; j < curT; j++) op[(long)c * curT + j] = src[(long)c * curT + j];
        x.Dispose();
        return outp;
    }

    private Tensor ProjectStageOut(IBackend backend, Tensor stageVec, int t)
    {
        if (_fsqProjOutW[0] is null) return stageVec;
        int outDim = (int)_fsqProjOutW[0]!.Shape[0];
        if (outDim != _codebookDim) return stageVec;
        // project_out is a Linear over the last dim; reuse the shared projection helper.
        return WhisperOps.ProjectLinear(backend, stageVec, _fsqProjOutW[0]!, _fsqProjOutB[0], 1, t,
            (int)_fsqProjOutW[0]!.Shape[1], outDim);
    }

    private Tensor ApplyFcPost(IBackend backend, Tensor cf, int t)
    {
        if (_fcPostW is null)
        {
            // No fc_post in the checkpoint: pass the codebook-dim tensor straight into the upsample stack.
            Tensor clone = new(cf.Shape, DType.F32);
            Buffer.MemoryCopy((void*)cf.DataPointer, (void*)clone.DataPointer, cf.ElementCount * 4, cf.ElementCount * 4);
            return clone;
        }
        int outCh = (int)_fcPostW.Shape[0];
        int kernel = _fcPostW.Shape.Rank == 3 ? (int)_fcPostW.Shape[2] : 1;
        int pad = (kernel - 1) / 2;
        int outT = t + 2 * pad - (kernel - 1);
        Tensor x = new(new TensorShape(1, outCh, outT), DType.F32);
        backend.Conv1d(x, cf, _fcPostW, _fcPostB, 1, pad, pad, 1, 1);
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[][] groups = [_fsqProjInW, _fsqProjInB, _fsqProjOutW, _fsqProjOutB, _upW, _upB, _upResW, _upResB];
        foreach (Tensor?[] g in groups) foreach (Tensor? t in g) if (t is not null) yield return t;
        if (_fcPostW is not null) yield return _fcPostW;
        if (_fcPostB is not null) yield return _fcPostB;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
