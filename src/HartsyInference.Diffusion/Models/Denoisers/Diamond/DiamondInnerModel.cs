using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.Diamond;

/// <summary>DIAMOND <c>InnerModel</c>: the conditioned U-Net that denoises the next frame. Fourier noise
/// embedding + action embedding → <c>cond_proj</c> MLP → per-block AdaGroupNorm conditioning; <c>conv_in</c>
/// over <c>cat(obs, noisy)</c> → U-Net → <c>silu(norm_out)</c> → <c>conv_out</c>. Mirrors
/// <c>eloialonso/diamond</c> <c>models/diffusion/inner_model.py</c> + <c>models/blocks.py</c> 1:1.</summary>
public sealed unsafe class DiamondInnerModel
{
    private readonly DiamondConfig _cfg;
    private Tensor? _noiseEmbW;                       // [1, cond/2] Fourier random weights
    private Tensor? _actEmbW;                         // [numActions, cond/numSteps] embedding table
    private Tensor? _condProj0W, _condProj0B, _condProj2W, _condProj2B;
    private Tensor? _convInW, _convInB, _convOutW, _convOutB, _normOutW, _normOutB;
    private readonly DiamondUNet _unet;

    public DiamondConfig Config => _cfg;

    public DiamondInnerModel(DiamondConfig cfg)
    {
        _cfg = cfg;
        _unet = new DiamondUNet(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _noiseEmbW = TensorCasts.EnsureF32(w[$"{p}noise_emb.weight"]);
        _actEmbW = TensorCasts.EnsureF32(w[$"{p}act_emb.0.weight"]);
        _condProj0W = TensorCasts.EnsureF32(w[$"{p}cond_proj.0.weight"]); _condProj0B = TensorCasts.EnsureF32(w[$"{p}cond_proj.0.bias"]);
        _condProj2W = TensorCasts.EnsureF32(w[$"{p}cond_proj.2.weight"]); _condProj2B = TensorCasts.EnsureF32(w[$"{p}cond_proj.2.bias"]);
        _convInW = TensorCasts.EnsureF32(w[$"{p}conv_in.weight"]); _convInB = TensorCasts.EnsureF32(w[$"{p}conv_in.bias"]);
        _unet.LoadWeights(w, $"{p}unet");
        _normOutW = TensorCasts.EnsureF32(w[$"{p}norm_out.norm.weight"]); _normOutB = TensorCasts.EnsureF32(w[$"{p}norm_out.norm.bias"]);
        _convOutW = TensorCasts.EnsureF32(w[$"{p}conv_out.weight"]); _convOutB = TensorCasts.EnsureF32(w[$"{p}conv_out.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_noiseEmbW, _actEmbW, _condProj0W, _condProj0B, _condProj2W, _condProj2B,
            _convInW, _convInB, _normOutW, _normOutB, _convOutW, _convOutB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (Tensor t in _unet.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the U-Net. <paramref name="noisy"/> is <c>[1,3,H,W]</c> (already ·c_in), <paramref name="cNoise"/>
    /// the scalar EDM noise embed input <c>[1]</c>, <paramref name="obs"/> the context frames <c>[1, 4·3, H, W]</c>
    /// (already /sigma_data), <paramref name="act"/> the context actions <c>[1, 4]</c> (ints as floats).</summary>
    public Tensor Forward(IBackend backend, Tensor noisy, float cNoise, Tensor obs, ReadOnlySpan<int> act)
    {
        int cond = _cfg.CondChannels, h = (int)noisy.Shape[2], w = (int)noisy.Shape[3];

        // cond = cond_proj(noise_emb(c_noise) + act_emb(act))
        Tensor c0 = new(new TensorShape(1, cond), DType.F32);
        BuildCondInput(c0, cNoise, act);
        Tensor c1 = new(new TensorShape(1, cond), DType.F32); backend.Linear(c1, c0, _condProj0W!, _condProj0B!); c0.Dispose();
        Tensor c1a = new(c1.Shape, DType.F32); backend.Silu(c1a, c1); c1.Dispose();
        Tensor condVec = new(new TensorShape(1, cond), DType.F32); backend.Linear(condVec, c1a, _condProj2W!, _condProj2B!); c1a.Dispose();

        // x = conv_in(cat(obs, noisy)) — device channel-concat keeps the input resident (no host round-trip)
        int inCh = _cfg.ImgChannels * (_cfg.NumStepsConditioning + 1);
        Tensor cat = new(new TensorShape(1, inCh, h, w), DType.F32);
        backend.Concat(cat, [obs, noisy], 1);
        Tensor x = DiamondOps.Conv(backend, cat, _convInW!, _convInB, 1, 1, _cfg.Channels[0]); cat.Dispose();

        // The U-Net owns `x` (it stores it as the level-0 skip and disposes it internally) — don't dispose here.
        Tensor uOut = _unet.Forward(backend, x, condVec);
        condVec.Dispose();

        // conv_out(silu(norm_out(x)))
        Tensor normed = new(uOut.Shape, DType.F32);
        backend.GroupNorm(normed, uOut, _normOutW!, _normOutB!, DiamondOps.Groups(_cfg.Channels[0], _cfg), _cfg.GnEps);
        uOut.Dispose();
        Tensor act2 = new(normed.Shape, DType.F32); backend.Silu(act2, normed); normed.Dispose();
        Tensor outp = DiamondOps.Conv(backend, act2, _convOutW!, _convOutB, 1, 1, _cfg.ImgChannels); act2.Dispose();
        return outp;
    }

    /// <summary><c>noise_emb(c_noise) + act_emb(act)</c> → <c>[1, cond]</c>. Fourier: <c>cat(cos, sin)</c> of
    /// <c>2π·c_noise·weight</c>; action embedding: table lookup per context step, concatenated.</summary>
    private void BuildCondInput(Tensor dst, float cNoise, ReadOnlySpan<int> act)
    {
        int cond = _cfg.CondChannels, half = cond / 2, perAct = cond / _cfg.NumStepsConditioning;
        float* dp = (float*)dst.DataPointer;
        float* nw = (float*)_noiseEmbW!.DataPointer;   // [1, half]
        for (int i = 0; i < half; i++)
        {
            float f = 2f * MathF.PI * cNoise * nw[i];
            dp[i] = MathF.Cos(f);
            dp[half + i] = MathF.Sin(f);
        }
        float* aw = (float*)_actEmbW!.DataPointer;      // [numActions, perAct]
        for (int t = 0; t < _cfg.NumStepsConditioning; t++)
        {
            int a = act[t];
            for (int e = 0; e < perAct; e++) dp[t * perAct + e] += aw[(long)a * perAct + e];
        }
    }
}
