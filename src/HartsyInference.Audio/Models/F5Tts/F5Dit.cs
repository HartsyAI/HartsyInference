using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>Full F5-TTS DiT denoiser. Predicts a vector field (in mel space) given
/// noisy target mel + reference mel + text + flow timestep. The pipeline iterates this
/// 32× under Sway-Sampling Euler to produce a denoised target mel, which is then
/// vocoded by Vocos.
///
/// <para>Forward signature mirrors the upstream <c>DiT.forward</c>:
/// <list type="bullet">
///   <item><c>noisyMel [1, mel_dim, T]</c> — current flow-matching state (the "x" of the ODE)</item>
///   <item><c>condMel  [1, mel_dim, T]</c> — masked reference mel (zeros over the target region)</item>
///   <item><c>textIds  int[T_text]</c> — character token IDs</item>
///   <item><c>time     float</c> — current flow timestep in [0, 1]</item>
///   <item><c>dropAudioCond</c>, <c>dropText</c> — CFG uncond toggles</item>
/// </list>
/// Output: <c>vectorField [1, T, mel_dim]</c> — the model's prediction of dx/dt.</para></summary>
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
    private int _disposed;

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

    /// <summary>Loads from a HuggingFace safetensors dictionary. The upstream prefix is
    /// <c>ema_model.transformer</c> for the released F5-TTS v1 base checkpoint
    /// (<c>F5TTS_v1_Base/model_1250000.safetensors</c>).</summary>
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
        // Repacked from the [maxPos, half] table layout into the backend's [maxPos, headDim] stride
        // (first half of each row) so ApplyRopeInterleaved runs on-device — the tensors are long-lived,
        // so weight auto-promotion keeps them GPU-resident.
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
                cp[p * _cfg.HeadDim + i] = cos[p * half + i];
                sp[p * _cfg.HeadDim + i] = sin[p * half + i];
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
        Tensor textHidden = _textEmb.Forward(backend, textIds, t, dropText);
        Tensor x = _inputEmb.Forward(backend, noisyMel, condMel, textHidden, t, dropAudioCond);
        textHidden.Dispose();

        // SiLU(t_emb) once per step, on-device; every block's adaLN Linear consumes it.
        Tensor siluTime = new(timeEmb.Shape, DType.F32);
        backend.Silu(siluTime, timeEmb);

        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x, siluTime, t, _ropeCos!, _ropeSin!);
            x.Dispose();
            x = next;
        }
        timeEmb.Dispose();

        // 5. Final AdaLayerNorm: x = LayerNorm(x, no affine) * (1 + scale) + shift, where
        //    scale and shift come from the timestep through norm_out.linear.
        Tensor outNorm = ApplyFinalAdaLn(backend, x, _normOutLinW!, _normOutLinB!, siluTime, t);
        siluTime.Dispose();
        x.Dispose();

        // 6. proj_out: Linear(dim → mel_dim) → [1, T, mel_dim]
        Tensor vec = WhisperOps.ProjectLinear(backend, outNorm, _projOutW!, _projOutB, 1, t, _cfg.Dim, _cfg.MelDim);
        outNorm.Dispose();
        return vec;
    }

    /// <summary>Diagnostics — returns copies of the text-stem output, the DiT input embedding, the block-0
    /// output, and the post-all-blocks hidden, for per-component parity checks. Caller disposes.</summary>
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

    /// <summary>AdaLayerNorm_Final: <c>x = LN_no_affine(x) * (1 + scale) + shift</c>
    /// where scale, shift come from a single Linear(dim → 2*dim) of <c>silu(time_emb)</c>.
    /// Chunk order in the projection is [scale, shift]. Fully on-device.</summary>
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

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}

/// <summary>F5-TTS sinusoidal-into-MLP timestep embedder. <c>freq_embed → Linear(1024) →
/// SiLU → Linear(1024)</c>. Outputs a per-batch <c>[1, 1024]</c> embedding for the current
/// flow-matching timestep <c>t ∈ [0, 1]</c>.</summary>
internal sealed unsafe class F5TimestepEmbedding
{
    private readonly F5TtsConfig _cfg;
    private Tensor? _mlp0W, _mlp0B;  // [1024, 256]
    private Tensor? _mlp2W, _mlp2B;  // [1024, 1024]

    /// <summary>Cached most-recent time embedding so the final AdaLN head can reuse it
    /// without recomputing the MLP. (Sub-optimal but keeps the F5Dit forward simple.)</summary>
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
