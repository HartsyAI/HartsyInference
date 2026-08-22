using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>Full F5-TTS DiT denoiser: predicts a vector field (in mel space) given noisy target mel + reference mel + text + flow timestep; the pipeline iterates this 32x under Sway-Sampling Euler to produce a denoised target mel, which is then vocoded by Vocos.</summary>
// Forward signature mirrors the upstream DiT.forward: noisyMel [1, mel_dim, T] is the current
// flow-matching state (the "x" of the ODE), condMel [1, mel_dim, T] is the masked reference mel (zeros
// over the target region), textIds int[T_text] are character token IDs, time float is the current flow
// timestep in [0, 1], and dropAudioCond/dropText are CFG uncond toggles. Output: vectorField
// [1, T, mel_dim], the model's prediction of dx/dt.
public sealed unsafe class F5Dit : IDisposable
{
    private readonly F5TtsConfig _cfg;
    private readonly F5TimestepEmbedding _timeEmb;
    private readonly F5TextEmbedding _textEmb;
    private readonly F5InputEmbed _inputEmb;
    private readonly F5DitBlock[] _blocks;

    // Output head: AdaLayerNorm_Final + Linear(dim → mel_dim)
    private Tensor? _normOutLinW, _normOutLinB;  // [2*dim, dim] / [2*dim]
    private Tensor? _projOutW, _projOutB;        // [mel_dim, dim] / [mel_dim]

    private Tensor? _ropeCos, _ropeSin;   // [maxPos, headDim] backend-ready (first half of each row used)
    private bool _loaded;

    // Per-loop text-embedding cache: two step-invariant tensors (cond/uncond) keyed by sequence length.
    private Tensor? _txtCond, _txtUncond;
    private int _txtCacheLen = -1;
    private int _disposed;

    // ── CUDA-graph step capture (opt-in via HARTSY_F5_GRAPH; ZImage pattern) ───────────────────────────
    // The 22-block core has an identical op sequence every forward (cond/uncond differ only in the embedding
    // stage, which stays outside the graph). Capturing it once and replaying via a single cuGraphLaunch
    // collapses the ~7k per-op host calls that dominate the eager path. Fixed input buffers (_xFixed embedded
    // hidden, _siluFixed timestep-SiLU) are refreshed per forward via CopyInto; the velocity lands in
    // _graphVelocity. Any capture-illegal op trips _graphDead → permanent eager fallback for the session.
    // EXPERIMENTAL / WIP — capture works but replay throws CUDA_ERROR_ILLEGAL_ADDRESS: the block core
    // allocates+frees intermediate tensors per-forward, so pool addresses shift on replay and the recorded
    // kernels read stale pointers. Landing this needs graph-stable scratch: pre-allocate every F5DitBlock
    // intermediate ONCE so the captured region does zero alloc/free (the remaining work). Keep OFF.
    private static readonly bool GraphEnabled = Environment.GetEnvironmentVariable("HARTSY_F5_GRAPH") == "1";
    private bool _graphDead;
    private Tensor? _xFixed, _siluFixed, _graphVelocity;
    private long _graphSig = long.MinValue;
    private int _graphSigCalls;
    private const int GraphCaptureCall = 3;   // calls 1-2 warm the fixed buffers/uploads, call 3 records

    public F5TtsConfig Config => _cfg;

    public F5Dit(F5TtsConfig cfg)
    {
        _cfg = cfg;
        _timeEmb = new F5TimestepEmbedding(cfg);
        _textEmb = new F5TextEmbedding(cfg);
        _inputEmb = new F5InputEmbed(cfg);
        _blocks = new F5DitBlock[cfg.Depth];
        for (int i = 0; i < cfg.Depth; i++) _blocks[i] = new F5DitBlock(cfg);
    }

    /// <summary>Loads from a HuggingFace safetensors dictionary; the upstream prefix is <c>ema_model.transformer</c> for the released F5-TTS v1 base checkpoint (<c>F5TTS_v1_Base/model_1250000.safetensors</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "ema_model.transformer")
    {
        _timeEmb.LoadWeights(w, $"{prefix}.time_embed");
        _textEmb.LoadWeights(w, $"{prefix}.text_embed");
        _inputEmb.LoadWeights(w, $"{prefix}.input_embed");
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(w, $"{prefix}.transformer_blocks.{i}");

        _normOutLinW = WhisperOps.EnsureF32(w[$"{prefix}.norm_out.linear.weight"]);
        _normOutLinB = WhisperOps.EnsureF32(w[$"{prefix}.norm_out.linear.bias"]);
        _projOutW = WhisperOps.EnsureF32(w[$"{prefix}.proj_out.weight"]);
        _projOutB = WhisperOps.EnsureF32(w[$"{prefix}.proj_out.bias"]);

        // Precompute RoPE tables. F5-TTS uses full RoPE (rotary_dim = head_dim = 64) with theta=10000.
        // Repacked into the backend's WanRopeInterleaved "duplicated-pair" layout: freq_i sits at position 2i
        // (and 2i+1) of each [maxPos, headDim] row, so the interleaved pair (2i, 2i+1) reads its angle at index
        // 2i. This lets RoPE run via the on-device WanRopeInterleaved kernel — the previous ApplyRopeInterleaved
        // has no CUDA override, so it yanked q/k to the host and back ~2800× per generation (the F5 slowdown).
        // The tables are long-lived, so weight auto-promotion keeps them GPU-resident.
        const int maxPos = 8_192;
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(_cfg.HeadDim, _cfg.RopeTheta, maxPos);
        int half = _cfg.HeadDim / 2;
        _ropeCos = new Tensor(new TensorShape(maxPos, _cfg.HeadDim), DType.F32);
        _ropeSin = new Tensor(new TensorShape(maxPos, _cfg.HeadDim), DType.F32);
        float* cp = (float*)_ropeCos.DataPointer;
        float* sp = (float*)_ropeSin.DataPointer;
        for (int p = 0; p < maxPos; p++)
        {
            for (int i = 0; i < half; i++)
            {
                int i0 = 2 * i;
                cp[p * _cfg.HeadDim + i0] = cp[p * _cfg.HeadDim + i0 + 1] = cos[p * half + i];
                sp[p * _cfg.HeadDim + i0] = sp[p * _cfg.HeadDim + i0 + 1] = sin[p * half + i];
            }
        }
        _loaded = true;
    }

    /// <summary>One DiT forward pass. Returns vector field of shape <c>[1, T, mel_dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor noisyMel, Tensor condMel, ReadOnlySpan<int> textIds, float timestep, bool dropAudioCond = false, bool dropText = false)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights first.");
        if (noisyMel.Shape.Rank != 3) throw new ArgumentException("noisyMel must be [B, mel_dim, T]");
        int t = (int)noisyMel.Shape[2];

        Tensor timeEmb = _timeEmb.Forward(backend, timestep);
        // Text embedding depends only on (textIds, t, dropText) — invariant across the 32 sampling steps, so
        // there are just two distinct values (cond/uncond). Computing it eagerly every forward reran 4 ConvNeXt
        // blocks whose depthwise conv is a host loop (GPU→host→GPU sync per block); cache the two tensors and
        // reuse them for the whole loop. After the first use each is a pure host-side read (no GPU work).
        Tensor textHidden = GetTextHidden(backend, textIds, t, dropText);
        Tensor x = _inputEmb.Forward(backend, noisyMel, condMel, textHidden, t, dropAudioCond);

        // SiLU(t_emb) once per step, on-device; every block's adaLN Linear consumes it.
        Tensor siluTime = new(timeEmb.Shape, DType.F32);
        backend.Silu(siluTime, timeEmb);
        timeEmb.Dispose();

        // The 22-block core (+ final head) is the expensive, structurally-identical part — graph-captured when
        // enabled; the embedding stage above stays eager (differs per cond/uncond drop flags).
        bool graphMode = GraphEnabled && backend.StepGraphSupported && !_graphDead;
        Tensor vec = graphMode ? RunBlocksGraph(backend, x, siluTime, t) : RunBlocks(backend, x, siluTime, t);
        x.Dispose();
        siluTime.Dispose();
        F5DitBlock.DumpProfile();
        return vec;
    }

    /// <summary>Call once before each sampling loop — the cache keys on sequence length only, so a new generation reusing a length with different text must reset first.</summary>
    public void ResetTextCache()
    {
        _txtCond?.Dispose(); _txtUncond?.Dispose();
        _txtCond = _txtUncond = null;
        _txtCacheLen = -1;
    }

    /// <summary>Returns the cond/uncond text embedding, computing it once per sequence length and reusing the cached tensor across all sampling steps; the returned tensor is owned by this cache and callers must NOT dispose it.</summary>
    private Tensor GetTextHidden(IBackend backend, ReadOnlySpan<int> textIds, int t, bool dropText)
    {
        if (t != _txtCacheLen)
        {
            _txtCond?.Dispose(); _txtUncond?.Dispose();
            _txtCond = _txtUncond = null;
            _txtCacheLen = t;
        }
        if (dropText)
            return _txtUncond ??= _textEmb.Forward(backend, textIds, t, dropText: true);
        return _txtCond ??= _textEmb.Forward(backend, textIds, t, dropText: false);
    }

    /// <summary>The 22 DiT blocks + final AdaLN head + proj_out, eager. Does NOT dispose <paramref name="xIn"/> — the caller (or the captured graph's fixed buffer) owns it.</summary>
    private Tensor RunBlocks(IBackend backend, Tensor xIn, Tensor siluTime, int t)
    {
        Tensor x = xIn;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x, siluTime, t, _ropeCos!, _ropeSin!);
            if (i > 0) x.Dispose();   // never dispose the caller's input
            x = next;
        }
        Tensor outNorm = ApplyFinalAdaLn(backend, x, _normOutLinW!, _normOutLinB!, siluTime, t);
        x.Dispose();
        Tensor vec = WhisperOps.ProjectLinear(backend, outNorm, _projOutW!, _projOutB, 1, t, _cfg.Dim, _cfg.MelDim);
        outNorm.Dispose();
        return vec;
    }

    /// <summary>Graph-captured block core: refreshes the fixed input buffers, then replays the captured cuGraph (or captures on call 3); returns a fresh copy of the velocity so both CFG branches keep independent buffers, and falls back to <see cref="RunBlocks"/> on any capture issue.</summary>
    private Tensor RunBlocksGraph(IBackend backend, Tensor x, Tensor siluTime, int t)
    {
        // (Re)allocate fixed buffers on shape change; the sequence length t is the only varying dim.
        if (_xFixed is null || (int)_xFixed.Shape[1] != t)
        {
            _xFixed?.Dispose(); _siluFixed?.Dispose(); _graphVelocity?.Dispose();
            _xFixed = new Tensor(x.Shape, DType.F32);
            _siluFixed = new Tensor(siluTime.Shape, DType.F32);
            _graphVelocity = new Tensor(new TensorShape(1, t, _cfg.MelDim), DType.F32);
            _graphSig = long.MinValue;   // force re-capture for the new shape
        }
        backend.CopyInto(_xFixed!, x);
        backend.CopyInto(_siluFixed!, siluTime);

        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        long sig = t;
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        _graphSigCalls++;

        // Everything the captured kernels read must be device-resident AND stay pinned across capture→replay
        // (an evicted weight/rope buffer moves → the recorded kernel reads a stale address → ILLEGAL_ADDRESS on
        // replay). Re-pin the full weight set + RoPE tables on each warmup call so the transient activation pool
        // can never reclaim them before the graph is instantiated. (Idempotent — already-cached weights no-op.)
        if (_graphSigCalls <= GraphCaptureCall)
            backend.PreloadWeights(System.Linq.Enumerable.Append(System.Linq.Enumerable.Append(EnumerateWeights(), _ropeCos!), _ropeSin!));

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return Clone(_graphVelocity!);
        }

        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture) backend.StepGraphBegin();
        try
        {
            Tensor vec = RunBlocks(backend, _xFixed!, _siluFixed!, t);
            backend.CopyInto(_graphVelocity!, vec);   // last (captured) op lands velocity at the fixed buffer
            vec.Dispose();
        }
        catch (Exception ex) when (capture)
        {
            backend.StepGraphReset();
            _graphDead = true;
            HartsyInference.Core.Logging.Logs.Warning($"[F5 graph] capture invalidated — eager fallback: {ex.Message}");
            return RunBlocks(backend, x, siluTime, t);
        }
        if (capture)
        {
            try { backend.StepGraphEndAndLaunch(); HartsyInference.Core.Logging.Logs.Info("[F5 graph] DiT block-core captured; replaying via cuGraphLaunch."); }
            catch (Exception ex)
            {
                backend.StepGraphReset();
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[F5 graph] capture failed — eager fallback: {ex.Message}");
                return RunBlocks(backend, x, siluTime, t);
            }
        }
        return Clone(_graphVelocity!);
    }

    /// <summary>Diagnostics — returns copies of the text-stem output, the DiT input embedding, the block-0 output, and the post-all-blocks hidden, for per-component parity checks; caller disposes.</summary>
    public (Tensor TextEmb, Tensor XInput, Tensor Block0, Tensor PreNorm) DebugForward(
        IBackend backend, Tensor noisyMel, Tensor condMel, ReadOnlySpan<int> textIds, float timestep)
    {
        int t = (int)noisyMel.Shape[2];
        Tensor timeEmb = _timeEmb.Forward(backend, timestep);
        Tensor textHidden = _textEmb.Forward(backend, textIds, t, false);
        Tensor textCopy = Clone(textHidden);
        Tensor x = _inputEmb.Forward(backend, noisyMel, condMel, textHidden, t, false);
        textHidden.Dispose();
        Tensor xInputCopy = Clone(x);

        Tensor siluTime = new(timeEmb.Shape, DType.F32);
        backend.Silu(siluTime, timeEmb);
        Tensor block0 = _blocks[0].Forward(backend, x, siluTime, t, _ropeCos!, _ropeSin!);
        x.Dispose();
        Tensor block0Copy = Clone(block0);
        x = block0;
        for (int i = 1; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x, siluTime, t, _ropeCos!, _ropeSin!);
            x.Dispose();
            x = next;
        }
        siluTime.Dispose();
        timeEmb.Dispose();
        Tensor preNorm = Clone(x);
        x.Dispose();
        return (textCopy, xInputCopy, block0Copy, preNorm);
    }

    private static Tensor Clone(Tensor t)
    {
        Tensor c = new(t.Shape, DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)c.DataPointer, t.ElementCount * 4, t.ElementCount * 4);
        return c;
    }

    /// <summary>AdaLayerNorm_Final: <c>x = LN_no_affine(x) * (1 + scale) + shift</c>, where scale/shift come from a single Linear(dim → 2*dim) of <c>silu(time_emb)</c> (chunk order [scale, shift]).</summary>
    private Tensor ApplyFinalAdaLn(IBackend backend, Tensor x, Tensor linW, Tensor linB, Tensor siluTime, int t)
    {
        int dim = _cfg.Dim;
        Tensor mods = WhisperOps.ProjectLinear(backend, siluTime, linW, linB, 1, 1, dim, 2 * dim);
        Tensor scale = new(new TensorShape(1, 1, dim), DType.F32);
        Tensor shift = new(new TensorShape(1, 1, dim), DType.F32);
        backend.SliceLastDim(scale, mods, 0);
        backend.SliceLastDim(shift, mods, dim);
        mods.Dispose();
        backend.AddScalar(scale, scale, 1f);

        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-6f);
        Tensor outT = new(x.Shape, DType.F32);
        backend.AffineBroadcastLastDim(outT, normed, scale, shift);
        normed.Dispose(); scale.Dispose(); shift.Dispose();
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _timeEmb.EnumerateWeights()) yield return t;
        foreach (Tensor t in _textEmb.EnumerateWeights()) yield return t;
        foreach (Tensor t in _inputEmb.EnumerateWeights()) yield return t;
        foreach (F5DitBlock b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_normOutLinW is not null) yield return _normOutLinW;
        if (_normOutLinB is not null) yield return _normOutLinB;
        if (_projOutW is not null) yield return _projOutW;
        if (_projOutB is not null) yield return _projOutB;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(F5Dit));
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
        _xFixed?.Dispose(); _siluFixed?.Dispose(); _graphVelocity?.Dispose();
        _xFixed = _siluFixed = _graphVelocity = null;
        _txtCond?.Dispose(); _txtUncond?.Dispose();
        _txtCond = _txtUncond = null;
    }
}

/// <summary>F5-TTS sinusoidal-into-MLP timestep embedder (<c>freq_embed → Linear(1024) → SiLU → Linear(1024)</c>), outputting a per-batch <c>[1, 1024]</c> embedding for the current flow-matching timestep <c>t ∈ [0, 1]</c>.</summary>
internal sealed unsafe class F5TimestepEmbedding
{
    private readonly F5TtsConfig _cfg;
    private Tensor? _mlp0W, _mlp0B;  // [1024, 256]
    private Tensor? _mlp2W, _mlp2B;  // [1024, 1024]

    /// <summary>Cached most-recent time embedding so the final AdaLN head can reuse it without recomputing the MLP (sub-optimal, but keeps the F5Dit forward simple).</summary>
    public Tensor? LastTimeEmb { get; private set; }

    public F5TimestepEmbedding(F5TtsConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _mlp0W = WhisperOps.EnsureF32(w[$"{prefix}.time_mlp.0.weight"]);
        _mlp0B = WhisperOps.EnsureF32(w[$"{prefix}.time_mlp.0.bias"]);
        _mlp2W = WhisperOps.EnsureF32(w[$"{prefix}.time_mlp.2.weight"]);
        _mlp2B = WhisperOps.EnsureF32(w[$"{prefix}.time_mlp.2.bias"]);
    }

    public Tensor Forward(IBackend backend, float timestep)
    {
        int freqDim = _cfg.TimeFreqEmbedDim;

        // 1. Sinusoidal position embedding of the timestep. F5-TTS uses the standard form:
        //    half = freqDim / 2
        //    factor = log(10000) / (half - 1)
        //    freqs[i] = exp(-factor * i) for i in [0, half)
        //    emb = [sin(t * freqs), cos(t * freqs)]  (concat — sin then cos)
        //
        //    NOTE: x_transformers / lucidrains' SinusPositionEmbedding swaps the order to
        //    [sin, cos] (vs the diffusers/Stable Diffusion convention of [cos, sin]). The
        //    F5-TTS reference uses lucidrains' implementation.
        // Rank-3 [1, 1, freqDim] so ProjectLinear → BatchedMatMul reads a.Shape[2] safely.
        // BatchedMatMul indexes a.Shape[0..2] unconditionally; a rank-2 tensor here used to
        // surface as a hang (out-of-bounds shape index returned a huge garbage K).
        Tensor sinEmb = new(new TensorShape(1, 1, freqDim), DType.F32);
        float* sp = (float*)sinEmb.DataPointer;
        int half = freqDim / 2;
        float factor = MathF.Log(10_000f) / (half - 1);
        // Upstream SinusPositionEmbedding scales the timestep by 1000 before the sinusoid (scale=1000).
        const float scale = 1000f;
        for (int i = 0; i < half; i++)
        {
            float w = MathF.Exp(-factor * i);
            float angle = scale * timestep * w;
            sp[i] = MathF.Sin(angle);
            sp[half + i] = MathF.Cos(angle);
        }

        // 2. Linear(256 → 1024)
        Tensor h1 = WhisperOps.ProjectLinear(backend, sinEmb, _mlp0W!, _mlp0B, 1, 1, freqDim, _cfg.Dim);
        sinEmb.Dispose();

        // 3. SiLU
        F5Ops.SiluInPlace(h1);

        // 4. Linear(1024 → 1024)
        Tensor h2 = WhisperOps.ProjectLinear(backend, h1, _mlp2W!, _mlp2B, 1, 1, _cfg.Dim, _cfg.Dim);
        h1.Dispose();

        // Cache for the final AdaLN head.
        LastTimeEmb?.Dispose();
        LastTimeEmb = new Tensor(h2.Shape, DType.F32);
        Buffer.MemoryCopy((void*)h2.DataPointer, (void*)LastTimeEmb.DataPointer,
            h2.ElementCount * 4, h2.ElementCount * 4);
        return h2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_mlp0W, _mlp0B, _mlp2W, _mlp2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
