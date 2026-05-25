using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.VibeVoice;

/// <summary>4-layer FFN-only diffusion denoiser. Predicts the v-target for the acoustic
/// VAE's 64-d latents, conditioned on the LM's per-step hidden state and the current
/// diffusion timestep. Mirrors <c>VibeVoiceDiffusionHead</c> in
/// <c>vibevoice/modular/modular_vibevoice_diffusion_head.py</c>.
///
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
/// here for parity.</para></summary>
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
    }

    /// <summary>One denoising step. <paramref name="noisyLatents"/> and
    /// <paramref name="condition"/> are channels-last <c>[1, N, ·]</c> tensors;
    /// <paramref name="timesteps"/> is a length-N float vector (positions inside the
    /// 1000-step DDPM schedule). Returns a fresh <c>[1, N, latent_size]</c> tensor with the
    /// predicted v-target. Caller owns disposal.</summary>
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

        // x = noisy_images_proj(noisy_latents)  →  [1, N, H]
        Tensor x = WhisperOps.ProjectLinear(backend, noisyLatents, _noisyProjW!, null, 1, n, _latentSize, _hiddenSize);

        // t = t_embedder(timesteps)  →  [1, N, H]
        Tensor t = BuildTimestepEmbedding(backend, timesteps, n);

        // condition_proj = cond_proj(condition)  →  [1, N, H]
        Tensor cProj = WhisperOps.ProjectLinear(backend, condition, _condProjW!, null, 1, n, _hiddenSize, _hiddenSize);

        // c = condition_proj + t  →  [1, N, H]
        Tensor c = new(cProj.Shape, DType.F32);
        backend.Add(c, cProj, t);
        cProj.Dispose();
        t.Dispose();

        // 4× HeadLayer.
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = HeadLayerForward(backend, x, c, _layers[i], n);
            x.Dispose();
            x = next;
        }

        // Final layer → [1, N, latent_size].
        Tensor result = FinalLayerForward(backend, x, c, n);
        x.Dispose();
        c.Dispose();
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
    }

    // ---- Internals --------------------------------------------------------

    private Tensor HeadLayerForward(IBackend backend, Tensor x, Tensor c, HeadLayerWeights l, int n)
    {
        // mod_out = adaLN(c) = Linear(silu(c))   →  [1, N, 3H]
        Tensor cAct = new(c.Shape, DType.F32);
        backend.Silu(cAct, c);
        Tensor mod = WhisperOps.ProjectLinear(backend, cAct, l.AdaLnW!, null, 1, n, _hiddenSize, 3 * _hiddenSize);
        cAct.Dispose();
        // Split into shift_ffn, scale_ffn, gate_ffn — each [N, H] in row-major layout.
        Tensor shift = SliceAlongLastDim(mod, n, 3 * _hiddenSize, 0, _hiddenSize);
        Tensor scale = SliceAlongLastDim(mod, n, 3 * _hiddenSize, _hiddenSize, _hiddenSize);
        Tensor gate = SliceAlongLastDim(mod, n, 3 * _hiddenSize, 2 * _hiddenSize, _hiddenSize);
        mod.Dispose();

        // normed = RMSNorm(x) with elementwise affine.
        Tensor normed = new(x.Shape, DType.F32);
        backend.RmsNorm(normed, x, l.NormW!, _config.RmsNormEps);

        // normed = modulate(normed, shift, scale)  ↦  normed * (1 + scale) + shift, in place.
        VibeVoiceOps.AdaLnModulate(normed, shift, scale, n, _hiddenSize);
        shift.Dispose();
        scale.Dispose();

        // SwiGLU FFN.
        Tensor swi = SwiGluForward(backend, normed, l.GateW!, l.UpW!, l.DownW!, n);
        normed.Dispose();

        // result = x + gate * swi.
        Tensor result = new(x.Shape, DType.F32);
        VibeVoiceOps.AdaLnGatedAdd(result, x, gate, swi, n, _hiddenSize);
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

        Tensor shift = SliceAlongLastDim(mod, n, 2 * _hiddenSize, 0, _hiddenSize);
        Tensor scale = SliceAlongLastDim(mod, n, 2 * _hiddenSize, _hiddenSize, _hiddenSize);
        mod.Dispose();

        // norm_final has elementwise_affine=False — divide by RMS only, no weight.
        Tensor normed = new(x.Shape, DType.F32);
        RmsNormNoAffine(normed, x, n, _hiddenSize, _config.RmsNormEps);

        VibeVoiceOps.AdaLnModulate(normed, shift, scale, n, _hiddenSize);
        shift.Dispose();
        scale.Dispose();

        // Linear projection back to latent_size.
        Tensor result = WhisperOps.ProjectLinear(backend, normed, _finalLayer.LinW!, null, 1, n, _hiddenSize, _latentSize);
        normed.Dispose();
        return result;
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

    private Tensor BuildTimestepEmbedding(IBackend backend, ReadOnlySpan<float> timesteps, int n)
    {
        // 1) Sinusoidal embedding [N, 256] = [cos(t*freqs), sin(t*freqs)].
        const int freqDim = 256;
        const int half = freqDim / 2;
        const float maxPeriod = 10_000f;
        Tensor sinEmb = new(new TensorShape(1, n, freqDim), DType.F32);
        float* sp = (float*)sinEmb.DataPointer;
        for (int i = 0; i < n; i++)
        {
            float ts = timesteps[i];
            int rowBase = i * freqDim;
            for (int k = 0; k < half; k++)
            {
                float freq = MathF.Exp(-MathF.Log(maxPeriod) * k / half);
                float arg = ts * freq;
                sp[rowBase + k] = MathF.Cos(arg);
                sp[rowBase + half + k] = MathF.Sin(arg);
            }
        }
        // freqDim is even (256), so no extra zero-pad column needed.

        // 2) MLP: Linear(256, H) → SiLU → Linear(H, H).
        Tensor h1 = WhisperOps.ProjectLinear(backend, sinEmb, _tEmbMlp0W!, null, 1, n, freqDim, _hiddenSize);
        sinEmb.Dispose();
        Tensor act = new(h1.Shape, DType.F32);
        backend.Silu(act, h1);
        h1.Dispose();
        Tensor result = WhisperOps.ProjectLinear(backend, act, _tEmbMlp2W!, null, 1, n, _hiddenSize, _hiddenSize);
        act.Dispose();
        return result;
    }

    private static void RmsNormNoAffine(Tensor output, Tensor input, int n, int d, float eps)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int i = 0; i < n; i++)
        {
            int rowBase = i * d;
            double sumSq = 0d;
            for (int k = 0; k < d; k++)
            {
                float v = ip[rowBase + k];
                sumSq += (double)v * v;
            }
            float invRms = 1f / MathF.Sqrt((float)(sumSq / d) + eps);
            for (int k = 0; k < d; k++) op[rowBase + k] = ip[rowBase + k] * invRms;
        }
    }

    /// <summary>Allocates a fresh <c>[1, N, segDim]</c> tensor copying a contiguous slice
    /// along the last axis of <paramref name="src"/> (shape <c>[1, N, fullDim]</c>).
    /// Implements the AdaLN <c>.chunk(k, dim=-1)</c> idiom by repeated calls with disjoint
    /// <paramref name="startCol"/> offsets.</summary>
    private static Tensor SliceAlongLastDim(Tensor src, int n, int fullDim, int startCol, int segDim)
    {
        Tensor result = new(new TensorShape(1, n, segDim), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int i = 0; i < n; i++)
        {
            int srcBase = i * fullDim + startCol;
            int dstBase = i * segDim;
            for (int k = 0; k < segDim; k++) dp[dstBase + k] = sp[srcBase + k];
        }
        return result;
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
