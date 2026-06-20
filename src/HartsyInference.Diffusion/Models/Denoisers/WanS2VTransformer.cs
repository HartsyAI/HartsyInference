using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Wan2.2-S2V DiT (speech-to-video), reconstructed from the original Wan repo (<c>wan/modules/s2v/model_s2v.py</c>
/// — <b>no diffusers reference</b>). The base Wan2.1 video transformer plus an <b>audio injector</b>: extra
/// temporally-aligned cross-attention blocks (the <see cref="WanAnimateFaceBlock"/> pattern) inserted at the
/// <c>AudioInjectLayers</c> block indices, where the latent stream cross-attends to the per-frame audio tokens
/// (<see cref="WanS2VAudioEncoder"/>) and the result is added back residually. Reference/motion-frame identity
/// conditioning is supplied by the pipeline via a latent-channel concat (config <c>InChannels</c>). Reuses
/// <see cref="WanDitOps"/> + <see cref="WanVideoBlock"/>; B=1; numerics + structure validation-pending.</summary>
public sealed unsafe class WanS2VTransformer : IDisposable
{
    private readonly WanVideoConfig _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly WanAnimateFaceBlock[] _audioInjector;
    private readonly int[] _injectLayers;
    private readonly WanRope _rope;
    private readonly int _patchVec;
    private int _disposed;

    private Tensor? _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB;
    private Tensor? _textW1, _textB1, _textW2, _textB2;

    public WanS2VTransformer(WanVideoConfig config, int[] audioInjectLayers)
    {
        _config = config;
        _injectLayers = audioInjectLayers;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        _audioInjector = new WanAnimateFaceBlock[_injectLayers.Length];
        for (int i = 0; i < _injectLayers.Length; i++) _audioInjector[i] = new WanAnimateFaceBlock(config.InnerDim, config.NumHeads, config.HeadDim, config.Eps);
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public WanVideoConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _patchW2d = WanDitOps.Reshape2d(w["patch_embedding.weight"], _config.InnerDim, _patchVec);
        w.TryGetValue("patch_embedding.bias", out _patchB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _finalScaleShift = LoadF32(w, "scale_shift_table");
        _timeEmb1W = w["condition_embedder.time_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["condition_embedder.time_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_2.bias", out _timeEmb2B);
        _timeProjW = w["condition_embedder.time_proj.weight"]; w.TryGetValue("condition_embedder.time_proj.bias", out _timeProjB);
        _textW1 = w["condition_embedder.text_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_1.bias", out _textB1);
        _textW2 = w["condition_embedder.text_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_2.bias", out _textB2);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
        for (int i = 0; i < _audioInjector.Length; i++) _audioInjector[i].LoadWeights(w, $"audio_injector.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2 })
            if (t is not null) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
        for (int i = 0; i < _audioInjector.Length; i++) foreach (Tensor t in _audioInjector[i].EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction. <paramref name="latent"/> is <c>[1, inChannels, T, H, W]</c> (inChannels includes
    /// any reference/motion concat); <paramref name="audioTokens"/> is <c>[gt, tokensPerFrame, dim]</c> (gt = latent
    /// frames, so it divides the post-patch token count); <paramref name="encoder"/> is umT5 features <c>[L, textDim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor audioTokens, Tensor encoder, float timestep)
    {
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int s = gt * gh * gw, dim = _config.InnerDim;

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(gt, gh, gw);
        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);
        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, [timestep], _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        Tensor encoderProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);

        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, s);
            cur.Dispose();
            cur = next;
            int injIdx = Array.IndexOf(_injectLayers, i);
            if (injIdx >= 0)
            {
                Tensor injected = _audioInjector[injIdx].Forward(backend, cur, audioTokens);   // latent ⨯ audio cross-attn
                AddInPlace(cur, injected);
                injected.Dispose();
            }
        }
        cos.Dispose(); sin.Dispose(); timestepProj.Dispose(); encoderProj.Dispose();

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, s, dim, _config.Eps, s);
        cur.Dispose();
        temb.Dispose();
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        return outVel;
    }

    private static void AddInPlace(Tensor acc, Tensor add)
    {
        long n = acc.Shape.ElementCount;
        float* ap = (float*)acc.DataPointer, dp = (float*)add.DataPointer;
        for (long i = 0; i < n; i++) ap[i] += dp[i];
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
        }
        GC.SuppressFinalize(this);
    }
}
