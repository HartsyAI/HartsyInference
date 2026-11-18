using HartsyInference.Audio.Layers;
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
/// here for parity. The layers run on <see cref="AdaLnSwiGluStack"/> and the timestep embedding on
/// <see cref="ScalarEmbed"/>, which Haru's MeanFlow head shares.</para></remarks>
internal sealed class VibeVoiceDiffusionHead
{
    private readonly VibeVoiceDiffusionHeadConfig _config;
    private readonly string _prefix;
    private readonly int _hiddenSize;
    private readonly int _latentSize;
    private readonly int _ffnDim;
    private readonly ScalarEmbed _timestep;

    private Tensor? _noisyProjW;          // [H, latent]
    private Tensor? _condProjW;           // [H, H]
    private Tensor? _tEmbMlp0W;           // [H, 256]
    private Tensor? _tEmbMlp2W;           // [H, H]
    private AdaLnSwiGluStack? _stack;

    public VibeVoiceDiffusionHead(VibeVoiceDiffusionHeadConfig config, string prefix)
    {
        _config = config;
        _prefix = prefix;
        _hiddenSize = config.HiddenSize;
        _latentSize = config.LatentSize;
        _ffnDim = (int)(config.HiddenSize * config.HeadFfnRatio);
        _timestep = new ScalarEmbed(config.HiddenSize);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _noisyProjW = WhisperOps.EnsureF32(w[$"{_prefix}.noisy_images_proj.weight"]);
        _condProjW = WhisperOps.EnsureF32(w[$"{_prefix}.cond_proj.weight"]);
        _tEmbMlp0W = WhisperOps.EnsureF32(w[$"{_prefix}.t_embedder.mlp.0.weight"]);
        _tEmbMlp2W = WhisperOps.EnsureF32(w[$"{_prefix}.t_embedder.mlp.2.weight"]);
        _timestep.LoadWeights(_tEmbMlp0W, _tEmbMlp2W);

        List<AdaLnSwiGluStack.Layer> layers = [];
        for (int i = 0; i < _config.HeadLayers; i++)
        {
            string lp = $"{_prefix}.layers.{i}";
            layers.Add(new AdaLnSwiGluStack.Layer(
                Norm: WhisperOps.EnsureF32(w[$"{lp}.norm.weight"]),
                Ada: WhisperOps.EnsureF32(w[$"{lp}.adaLN_modulation.1.weight"]),
                Gate: WhisperOps.EnsureF32(w[$"{lp}.ffn.gate_proj.weight"]),
                Up: WhisperOps.EnsureF32(w[$"{lp}.ffn.up_proj.weight"]),
                Down: WhisperOps.EnsureF32(w[$"{lp}.ffn.down_proj.weight"])));
        }
        _stack = new AdaLnSwiGluStack(_hiddenSize, _ffnDim, _config.RmsNormEps, _latentSize, layers,
            WhisperOps.EnsureF32(w[$"{_prefix}.final_layer.adaLN_modulation.1.weight"]),
            WhisperOps.EnsureF32(w[$"{_prefix}.final_layer.linear.weight"]));
    }

    /// <summary>One denoising step over channels-last <c>[1, N, ·]</c> tensors, <paramref name="timesteps"/> being positions inside the 1000-step DDPM schedule.</summary>
    /// <returns>A fresh <c>[1, N, latent_size]</c> tensor with the predicted v-target; caller owns disposal.</returns>
    public Tensor Forward(IBackend backend, Tensor noisyLatents, ReadOnlySpan<float> timesteps, Tensor condition)
    {
        if (_noisyProjW is null || _stack is null) throw new InvalidOperationException($"VibeVoiceDiffusionHead '{_prefix}' weights not loaded.");
        if (noisyLatents.Shape.Rank != 3 || (int)noisyLatents.Shape[2] != _latentSize)
            throw new ArgumentException($"noisyLatents must be [1, N, {_latentSize}], got {noisyLatents.Shape}.", nameof(noisyLatents));
        if (condition.Shape.Rank != 3 || (int)condition.Shape[2] != _hiddenSize)
            throw new ArgumentException($"condition must be [1, N, {_hiddenSize}], got {condition.Shape}.", nameof(condition));

        int n = (int)noisyLatents.Shape[1];
        if (timesteps.Length != n)
            throw new ArgumentException($"timesteps length ({timesteps.Length}) must match N ({n}).", nameof(timesteps));

        // x = noisy_images_proj(noisy_latents), c = cond_proj(condition) + t_embedder(timesteps); the layers take silu(c).
        Tensor x = WhisperOps.ProjectLinear(backend, noisyLatents, _noisyProjW, null, 1, n, _latentSize, _hiddenSize);
        Tensor t = _timestep.Forward(backend, timesteps);
        Tensor cProj = WhisperOps.ProjectLinear(backend, condition, _condProjW!, null, 1, n, _hiddenSize, _hiddenSize);
        Tensor c = new(cProj.Shape, DType.F32);
        backend.Add(c, cProj, t);
        cProj.Dispose();
        t.Dispose();
        Tensor active = new(c.Shape, DType.F32);
        backend.Silu(active, c);
        c.Dispose();

        Tensor result = _stack.Forward(backend, x, active);
        x.Dispose();
        active.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_noisyProjW, _condProjW];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (Tensor t in _timestep.EnumerateWeights()) yield return t;
        if (_stack is not null)
        {
            foreach (Tensor t in _stack.EnumerateWeights()) yield return t;
        }
    }
}
