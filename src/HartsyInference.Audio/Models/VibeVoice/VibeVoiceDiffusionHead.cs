using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>4-layer FFN-only diffusion denoiser predicting the v-target for the acoustic VAE's 64-d latents, conditioned on the LM's per-step hidden state and diffusion timestep; mirrors <c>VibeVoiceDiffusionHead</c> in <c>vibevoice/modular/modular_vibevoice_diffusion_head.py</c>.</summary>
/// <remarks>
/// <para>Forward (per inference step):
/// <code>
///   x = noisy_images_proj(noisy_latents)   # [N, 64] → [N, H]
///   t = t_embedder(timesteps)              # [N]     → [N, H]   sinusoidal+MLP
///   c = cond_proj(condition) + t           # [N, H] + [N, H] = [N, H]
///   for each of 4 HeadLayer:
///     shift, scale, gate = adaLN(c).chunk(3)              # each [N, H]
///     x = x + gate * SwiGLU(modulate(RMSNorm(x), shift, scale))
///   shift, scale = final.adaLN(c).chunk(2)
///   x = final.linear(modulate(RMSNorm_no_affine(x), shift, scale))
///   return x                               # [N, 64]
/// </code></para>
///
/// <para>All Linear layers carry no bias (Python source uses <c>bias=False</c> across the
/// head). The SwiGLU FFN has <c>head_ffn_ratio = 3.0</c> in the published checkpoints.
/// The final layer is zero-initialized; that doesn't matter for inference but is noted
/// here for parity.</para></remarks>
internal sealed unsafe class VibeVoiceDiffusionHead
{
    private readonly VibeVoiceDiffusionHeadConfig _config;
    private readonly string _prefix;
    private readonly int _hiddenSize;
    private readonly int _latentSize;
    private readonly int _ffnDim;

    private Tensor? _noisyProjW;          // [H, latent]
    private Tensor? _condProjW;           // [H, H]
    private Tensor? _tEmbMlp0W;           // [H, 256]
    private Tensor? _tEmbMlp2W;           // [H, H]

    private readonly HeadLayerWeights[] _layers;
    private FinalLayerWeights _finalLayer;
    private Tensor? _onesHidden;          // [H] all-ones weight → affine-free RMSNorm via IBackend.RmsNorm

    public VibeVoiceDiffusionHead(VibeVoiceDiffusionHeadConfig config, string prefix)
    {
        _config = config;
        _prefix = prefix;
        _hiddenSize = config.HiddenSize;
        _latentSize = config.LatentSize;
        _ffnDim = (int)(config.HiddenSize * config.HeadFfnRatio);
        _layers = new HeadLayerWeights[config.HeadLayers];
        _finalLayer = new FinalLayerWeights();
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _noisyProjW = WhisperOps.EnsureF32(w[$"{_prefix}.noisy_images_proj.weight"]);
        _condProjW = WhisperOps.EnsureF32(w[$"{_prefix}.cond_proj.weight"]);
        _tEmbMlp0W = WhisperOps.EnsureF32(w[$"{_prefix}.t_embedder.mlp.0.weight"]);
        _tEmbMlp2W = WhisperOps.EnsureF32(w[$"{_prefix}.t_embedder.mlp.2.weight"]);

        for (int i = 0; i < _layers.Length; i++)
        {
            string lp = $"{_prefix}.layers.{i}";
            _layers[i] = new HeadLayerWeights
            {
                NormW = WhisperOps.EnsureF32(w[$"{lp}.norm.weight"]),
                AdaLnW = WhisperOps.EnsureF32(w[$"{lp}.adaLN_modulation.1.weight"]),
                GateW = WhisperOps.EnsureF32(w[$"{lp}.ffn.gate_proj.weight"]),
                UpW = WhisperOps.EnsureF32(w[$"{lp}.ffn.up_proj.weight"]),
                DownW = WhisperOps.EnsureF32(w[$"{lp}.ffn.down_proj.weight"]),
            };
        }

        _finalLayer = new FinalLayerWeights
        {
            AdaLnW = WhisperOps.EnsureF32(w[$"{_prefix}.final_layer.adaLN_modulation.1.weight"]),
            LinW = WhisperOps.EnsureF32(w[$"{_prefix}.final_layer.linear.weight"]),
        };

        _onesHidden = new Tensor(new TensorShape(_hiddenSize), DType.F32);
        float* ones = (float*)_onesHidden.DataPointer;
        for (int i = 0; i < _hiddenSize; i++) ones[i] = 1f;
    }

    /// <summary>One denoising step over channels-last <c>[1, N, ·]</c> tensors, <paramref name="timesteps"/> being positions inside the 1000-step DDPM schedule.</summary>
    /// <returns>A fresh <c>[1, N, latent_size]</c> tensor with the predicted v-target; caller owns disposal.</returns>
    public Tensor Forward(IBackend backend, Tensor noisyLatents, ReadOnlySpan<float> timesteps, Tensor condition)
    {
        if (_noisyProjW is null) throw new InvalidOperationException($"VibeVoiceDiffusionHead '{_prefix}' weights not loaded.");
        if (noisyLatents.Shape.Rank != 3 || (int)noisyLatents.Shape[2] != _latentSize)
            throw new ArgumentException($"noisyLatents must be [1, N, {_latentSize}], got {noisyLatents.Shape}.", nameof(noisyLatents));
        if (condition.Shape.Rank != 3 || (int)condition.Shape[2] != _hiddenSize)
            throw new ArgumentException($"condition must be [1, N, {_hiddenSize}], got {condition.Shape}.", nameof(condition));

        int n = (int)noisyLatents.Shape[1];
        if (timesteps.Length != n)
            throw new ArgumentException($"timesteps length ({timesteps.Length}) must match N ({n}).", nameof(timesteps));

        // Sinusoidal timestep embedding (host) → then the fully-device core.
        Tensor sinEmb = new(new TensorShape(1, n, SinDim), DType.F32);
        ComputeSinusoidal(sinEmb, timesteps, n);
        Tensor result = RunFromSin(backend, noisyLatents, sinEmb, condition, n);
        sinEmb.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_noisyProjW, _condProjW, _tEmbMlp0W, _tEmbMlp2W];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (HeadLayerWeights l in _layers)
        {
            Tensor?[] arr = [l.NormW, l.AdaLnW, l.GateW, l.UpW, l.DownW];
            foreach (Tensor? t in arr) if (t is not null) yield return t;
        }
        if (_finalLayer.AdaLnW is not null) yield return _finalLayer.AdaLnW;
        if (_finalLayer.LinW is not null) yield return _finalLayer.LinW;
        if (_onesHidden is not null) yield return _onesHidden;
    }

    // ---- Internals --------------------------------------------------------

    private Tensor HeadLayerForward(IBackend backend, Tensor x, Tensor c, HeadLayerWeights l, int n)
    {
        // mod_out = adaLN(c) = Linear(silu(c))   →  [1, N, 3H]
        Tensor cAct = new(c.Shape, DType.F32);
        backend.Silu(cAct, c);
        Tensor mod = WhisperOps.ProjectLinear(backend, cAct, l.AdaLnW!, null, 1, n, _hiddenSize, 3 * _hiddenSize);
        cAct.Dispose();
        // Split into shift_ffn, scale_ffn, gate_ffn — each [1, N, H] (device-resident slices).
        Tensor shift = SliceGpu(backend, mod, n, 0);
        Tensor scale = SliceGpu(backend, mod, n, _hiddenSize);
        Tensor gate = SliceGpu(backend, mod, n, 2 * _hiddenSize);
        mod.Dispose();

        // normed = RMSNorm(x) with elementwise affine, then modulate(normed, shift, scale).
        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, l.NormW!, _config.RmsNormEps);
        Tensor modulated = ModulateGpu(backend, normed, shift, scale);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();

        // SwiGLU FFN.
        Tensor swi = SwiGluForward(backend, modulated, l.GateW!, l.UpW!, l.DownW!, n);
        modulated.Dispose();

        // result = x + gate * swi.
        Tensor result = GatedAddGpu(backend, x, gate, swi);
        gate.Dispose();
        swi.Dispose();
        return result;
    }

    private Tensor FinalLayerForward(IBackend backend, Tensor x, Tensor c, int n)
    {
        // mod = Linear(silu(c))  →  [1, N, 2H]
        Tensor cAct = new(c.Shape, DType.F32);
        backend.Silu(cAct, c);
        Tensor mod = WhisperOps.ProjectLinear(backend, cAct, _finalLayer.AdaLnW!, null, 1, n, _hiddenSize, 2 * _hiddenSize);
        cAct.Dispose();

        Tensor shift = SliceGpu(backend, mod, n, 0);
        Tensor scale = SliceGpu(backend, mod, n, _hiddenSize);
        mod.Dispose();

        // norm_final has elementwise_affine=False — RMSNorm with an all-ones weight = divide by RMS only.
        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, _onesHidden!, _config.RmsNormEps);
        Tensor modulated = ModulateGpu(backend, normed, shift, scale);
        normed.Dispose();
        shift.Dispose();
        scale.Dispose();

        // Linear projection back to latent_size.
        Tensor result = WhisperOps.ProjectLinear(backend, modulated, _finalLayer.LinW!, null, 1, n, _hiddenSize, _latentSize);
        modulated.Dispose();
        return result;
    }

    // adaLN chunk: contiguous [1, N, H] slice of a [1, N, kH] tensor along the last dim (GPU-resident).
    private Tensor SliceGpu(IBackend backend, Tensor src, int n, int startCol)
    {
        Tensor result = new(new TensorShape(1, n, _hiddenSize), DType.F32);
        backend.SliceLastDim(result, src, startCol);
        return result;
    }

    // adaLN modulate: out = x * (1 + scale) + shift (all [1, N, H]), composed from resident elementwise ops.
    private static Tensor ModulateGpu(IBackend backend, Tensor x, Tensor shift, Tensor scale)
    {
        Tensor sc1 = new(scale.Shape, DType.F32);
        backend.AddScalar(sc1, scale, 1f);
        Tensor scaled = new(x.Shape, DType.F32);
        backend.Mul(scaled, x, sc1);
        sc1.Dispose();
        Tensor outp = new(x.Shape, DType.F32);
        backend.Add(outp, scaled, shift);
        scaled.Dispose();
        return outp;
    }

    // gated residual: out = residual + gate * value (all [1, N, H]), resident.
    private static Tensor GatedAddGpu(IBackend backend, Tensor residual, Tensor gate, Tensor value)
    {
        Tensor gv = new(value.Shape, DType.F32);
        backend.Mul(gv, gate, value);
        Tensor outp = new(residual.Shape, DType.F32);
        backend.Add(outp, residual, gv);
        gv.Dispose();
        return outp;
    }

    private Tensor SwiGluForward(IBackend backend, Tensor x, Tensor gateW, Tensor upW, Tensor downW, int n)
    {
        // gate = gate_proj(x)  →  [1, N, ffn]
        Tensor gate = WhisperOps.ProjectLinear(backend, x, gateW, null, 1, n, _hiddenSize, _ffnDim);
        // up = up_proj(x)      →  [1, N, ffn]
        Tensor up = WhisperOps.ProjectLinear(backend, x, upW, null, 1, n, _hiddenSize, _ffnDim);
        // gate = silu(gate)
        Tensor gateAct = new(gate.Shape, DType.F32);
        backend.Silu(gateAct, gate);
        gate.Dispose();
        // mixed = gateAct * up
        Tensor mixed = new(gateAct.Shape, DType.F32);
        backend.Mul(mixed, gateAct, up);
        gateAct.Dispose();
        up.Dispose();
        // result = down_proj(mixed)  →  [1, N, H]
        Tensor result = WhisperOps.ProjectLinear(backend, mixed, downW, null, 1, n, _ffnDim, _hiddenSize);
        mixed.Dispose();
        return result;
    }

    private const int SinDim = 256;

    /// <summary>The fully-device core of one denoising step (no host reads) — CUDA-graph-capturable: noisy_proj + t_embedder MLP + cond_proj → 4 head layers → final.</summary>
    /// <param name="sinEmb">Precomputed sinusoidal timestep embedding <c>[1, N, 256]</c> — the only host-computed input, materialized outside the capture window.</param>
    private Tensor RunFromSin(IBackend backend, Tensor noisyLatents, Tensor sinEmb, Tensor condition, int n)
    {
        // x = noisy_images_proj(noisy_latents)  →  [1, N, H]
        Tensor x = WhisperOps.ProjectLinear(backend, noisyLatents, _noisyProjW!, null, 1, n, _latentSize, _hiddenSize);

        // t = t_embedder MLP: Linear(256, H) → SiLU → Linear(H, H)  →  [1, N, H]
        Tensor t = TimestepMlp(backend, sinEmb, n);

        // condition_proj = cond_proj(condition)  →  [1, N, H]
        Tensor cProj = WhisperOps.ProjectLinear(backend, condition, _condProjW!, null, 1, n, _hiddenSize, _hiddenSize);

        // c = condition_proj + t  →  [1, N, H]
        Tensor c = new(cProj.Shape, DType.F32);
        backend.Add(c, cProj, t);
        cProj.Dispose();
        t.Dispose();

        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = HeadLayerForward(backend, x, c, _layers[i], n);
            x.Dispose();
            x = next;
        }

        Tensor result = FinalLayerForward(backend, x, c, n);
        x.Dispose();
        c.Dispose();
        return result;
    }

    private Tensor TimestepMlp(IBackend backend, Tensor sinEmb, int n)
    {
        Tensor h1 = WhisperOps.ProjectLinear(backend, sinEmb, _tEmbMlp0W!, null, 1, n, SinDim, _hiddenSize);
        Tensor act = new(h1.Shape, DType.F32);
        backend.Silu(act, h1);
        h1.Dispose();
        Tensor result = WhisperOps.ProjectLinear(backend, act, _tEmbMlp2W!, null, 1, n, _hiddenSize, _hiddenSize);
        act.Dispose();
        return result;
    }

    /// <summary>Sinusoidal timestep embedding <c>[1, N, 256] = [cos(t·freqs), sin(t·freqs)]</c> (host).</summary>
    private static void ComputeSinusoidal(Tensor sinEmb, ReadOnlySpan<float> timesteps, int n)
    {
        const int half = SinDim / 2;
        const float maxPeriod = 10_000f;
        float* sp = (float*)sinEmb.DataPointer;
        for (int i = 0; i < n; i++)
        {
            float ts = timesteps[i];
            int rowBase = i * SinDim;
            for (int k = 0; k < half; k++)
            {
                float freq = MathF.Exp(-MathF.Log(maxPeriod) * k / half);
                float arg = ts * freq;
                sp[rowBase + k] = MathF.Cos(arg);
                sp[rowBase + half + k] = MathF.Sin(arg);
            }
        }
    }

    private struct HeadLayerWeights
    {
        public Tensor? NormW;
        public Tensor? AdaLnW;
        public Tensor? GateW;
        public Tensor? UpW;
        public Tensor? DownW;
    }

    private struct FinalLayerWeights
    {
        public Tensor? AdaLnW;
        public Tensor? LinW;
    }
}
