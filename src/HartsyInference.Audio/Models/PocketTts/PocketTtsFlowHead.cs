using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>Per-frame flow-matching / Lagrangian-self-distillation (LSD) head for Pocket-TTS. Given the Flow-LM's
/// per-frame hidden state (the conditioning <c>cond</c>), it samples the next continuous audio latent by running a
/// short flow ODE: starting from Gaussian noise at <c>t = 0</c>, a velocity network <c>v_θ(x, t | cond)</c> is
/// integrated with first-order Euler over <see cref="PocketTtsConfig.LsdDecodeSteps"/> steps to <c>t = 1</c>, with
/// optional latent classifier-free guidance.
///
/// <para>The velocity network is a small SiLU MLP conditioned on the LM hidden plus a sinusoidal embedding of the
/// flow time. The distilled released model uses very few steps (LSD trains the few-step solver), so this stays
/// cheap per frame. All dims come from <see cref="PocketTtsConfig"/> — none are hard-coded.</para>
///
/// <para>Weights (HF-style keys under <c>flow_head.*</c>):
/// <c>time_in.weight/bias</c> (TimeEmbedDim → FlowHeadDim), <c>cond_in.weight/bias</c> (DModel → FlowHeadDim),
/// <c>in.weight/bias</c> (LatentDim → FlowHeadDim), <c>mlp.0.weight/bias</c> (FlowHeadDim → FlowHeadDim),
/// <c>out.weight/bias</c> (FlowHeadDim → LatentDim). Synthesizable for tests.</para></summary>
public sealed unsafe class PocketTtsFlowHead : ICfmEstimator
{
    private readonly PocketTtsConfig _cfg;
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _timeDim;

    private Tensor? _timeW, _timeB;
    private Tensor? _condW, _condB;
    private Tensor? _inW, _inB;
    private Tensor? _mlpW, _mlpB;
    private Tensor? _outW, _outB;

    public int LatentDim => _latentDim;

    public PocketTtsFlowHead(PocketTtsConfig cfg)
    {
        _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        if (cfg.LatentDim <= 0) throw new ArgumentException("PocketTtsConfig.LatentDim must be > 0.", nameof(cfg));
        if (cfg.DModel <= 0) throw new ArgumentException("PocketTtsConfig.DModel must be > 0.", nameof(cfg));
        _latentDim = cfg.LatentDim;
        _hiddenDim = cfg.FlowHeadDim > 0 ? cfg.FlowHeadDim : cfg.DModel;
        _timeDim = cfg.TimeEmbedDim > 0 ? cfg.TimeEmbedDim : _hiddenDim;
    }

    /// <summary>Loads the flow-head weights from an HF-style key dict. <paramref name="prefix"/> is typically
    /// <c>"flow_lm.flow_head"</c> — adjust once the real checkpoint keys are known.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        if (w is null) throw new ArgumentNullException(nameof(w));
        _timeW = WhisperOps.EnsureF32(w[$"{prefix}.time_in.weight"]);
        _timeB = WhisperOps.EnsureF32(w[$"{prefix}.time_in.bias"]);
        _condW = WhisperOps.EnsureF32(w[$"{prefix}.cond_in.weight"]);
        _condB = WhisperOps.EnsureF32(w[$"{prefix}.cond_in.bias"]);
        _inW = WhisperOps.EnsureF32(w[$"{prefix}.in.weight"]);
        _inB = WhisperOps.EnsureF32(w[$"{prefix}.in.bias"]);
        _mlpW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.0.weight"]);
        _mlpB = WhisperOps.EnsureF32(w[$"{prefix}.mlp.0.bias"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.out.weight"]);
        _outB = WhisperOps.EnsureF32(w[$"{prefix}.out.bias"]);
    }

    /// <summary>Samples the next continuous latent <c>[1, LatentDim]</c> conditioned on the Flow-LM hidden
    /// <paramref name="cond"/> (<c>[1, DModel]</c> or <c>[DModel]</c>). Runs an Euler flow solve from Gaussian
    /// noise with latent CFG. Deterministic for a fixed <paramref name="seed"/>.</summary>
    public Tensor SampleLatent(IBackend backend, Tensor cond, int seed)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        EnsureLoaded();

        Tensor x = RandNormal(_latentDim, seed);
        int steps = Math.Max(1, _cfg.LsdDecodeSteps);
        float dt = 1f / steps;
        float cfg = _cfg.LatentCfgScale;

        Tensor? zeroCond = cfg > 0f ? new Tensor(new TensorShape(1, _cfg.DModel), DType.F32) : null;

        for (int step = 0; step < steps; step++)
        {
            float tNow = step * dt;
            Tensor v = Velocity(backend, x, cond, tNow);
            if (cfg > 0f)
            {
                Tensor v0 = Velocity(backend, x, zeroCond!, tNow);
                float* vp = (float*)v.DataPointer;
                float* v0p = (float*)v0.DataPointer;
                float a = 1f + cfg;
                for (int i = 0; i < _latentDim; i++) vp[i] = a * vp[i] - cfg * v0p[i];
                v0.Dispose();
            }
            float* xp = (float*)x.DataPointer;
            float* vvp = (float*)v.DataPointer;
            for (int i = 0; i < _latentDim; i++) xp[i] += dt * vvp[i];
            v.Dispose();
        }

        zeroCond?.Dispose();
        return x;
    }

    /// <summary>OT-CFM <see cref="ICfmEstimator"/> entry: returns <c>v_θ(x, t | cond)</c>. The Pocket-TTS head
    /// only conditions on the LM hidden (<paramref name="mu"/>); <paramref name="spk"/> / <paramref name="cond"/>
    /// are unused (kept for interface compatibility so the shared <c>ConditionalCfm</c> solver can drive it too).</summary>
    public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond, Tensor? attnMask = null)
    {
        EnsureLoaded();
        if (x.ElementCount != _latentDim)
            throw new ArgumentException($"flow x must have {_latentDim} elements, got {x.ElementCount}.", nameof(x));
        if (mu.ElementCount != _cfg.DModel)
            throw new ArgumentException($"flow conditioning must have {_cfg.DModel} elements, got {mu.ElementCount}.", nameof(mu));
        return Velocity(backend, x, mu, t);
    }

    private Tensor Velocity(IBackend backend, Tensor x, Tensor cond, float t)
    {
        // h = in(x) + cond_in(cond) + time_in(timeEmbed(t)); h = silu(mlp(silu(h))); out(h).
        // ProjectLinear needs rank-3 [B, S, in] inputs; reshape the rank-2 [1, dim] tensors to [1, 1, dim].
        Tensor x3 = x.Reshape(new TensorShape(1, 1, _latentDim));
        Tensor cond3 = cond.Reshape(new TensorShape(1, 1, _cfg.DModel));
        Tensor inProj = WhisperOps.ProjectLinear(backend, x3, _inW!, _inB, 1, 1, _latentDim, _hiddenDim);
        Tensor condProj = WhisperOps.ProjectLinear(backend, cond3, _condW!, _condB, 1, 1, _cfg.DModel, _hiddenDim);
        x3.Dispose();
        cond3.Dispose();

        Tensor timeEmbed = SinusoidalTime(t);
        Tensor timeEmbed3 = timeEmbed.Reshape(new TensorShape(1, 1, _timeDim));
        Tensor timeProj = WhisperOps.ProjectLinear(backend, timeEmbed3, _timeW!, _timeB, 1, 1, _timeDim, _hiddenDim);
        timeEmbed3.Dispose();
        timeEmbed.Dispose();

        float* hp = (float*)inProj.DataPointer;
        float* cp = (float*)condProj.DataPointer;
        float* tp = (float*)timeProj.DataPointer;
        for (int i = 0; i < _hiddenDim; i++) hp[i] += cp[i] + tp[i];
        condProj.Dispose();
        timeProj.Dispose();

        Tensor act = new(inProj.Shape, DType.F32);
        backend.Silu(act, inProj);
        inProj.Dispose();

        Tensor mlp = WhisperOps.ProjectLinear(backend, act, _mlpW!, _mlpB, 1, 1, _hiddenDim, _hiddenDim);
        act.Dispose();
        Tensor mlpAct = new(mlp.Shape, DType.F32);
        backend.Silu(mlpAct, mlp);
        mlp.Dispose();

        Tensor v = WhisperOps.ProjectLinear(backend, mlpAct, _outW!, _outB, 1, 1, _hiddenDim, _latentDim);
        mlpAct.Dispose();
        return v;     // [1, 1, LatentDim] — read flat (LatentDim contiguous elements)
    }

    private Tensor SinusoidalTime(float t)
    {
        Tensor emb = new(new TensorShape(1, _timeDim), DType.F32);
        float* p = (float*)emb.DataPointer;
        int half = _timeDim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10_000f) * i / Math.Max(1, half));
            p[i] = MathF.Sin(t * freq);
            p[half + i] = MathF.Cos(t * freq);
        }
        if ((_timeDim & 1) == 1) p[_timeDim - 1] = t;
        return emb;
    }

    private static Tensor RandNormal(int dim, int seed)
    {
        Tensor x = new(new TensorShape(1, dim), DType.F32);
        float* p = (float*)x.DataPointer;
        uint rng = DeterministicRng.Seed(seed);
        for (int i = 0; i < dim; i++) p[i] = DeterministicRng.NextGaussian(ref rng);
        return x;
    }

    private void EnsureLoaded()
    {
        if (_outW is null) throw new InvalidOperationException("PocketTtsFlowHead weights not loaded.");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_timeW is not null) yield return _timeW;
        if (_timeB is not null) yield return _timeB;
        if (_condW is not null) yield return _condW;
        if (_condB is not null) yield return _condB;
        if (_inW is not null) yield return _inW;
        if (_inB is not null) yield return _inB;
        if (_mlpW is not null) yield return _mlpW;
        if (_mlpB is not null) yield return _mlpB;
        if (_outW is not null) yield return _outW;
        if (_outB is not null) yield return _outB;
    }
}
