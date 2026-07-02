using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Wan2.2-S2V DiT (speech-to-video), a faithful port of ComfyUI's <c>WanModel_S2V.forward_orig</c>
/// (<c>comfy/ldm/wan/model.py</c>). The Wan2.1-14B T2V backbone plus:
/// <list type="bullet">
/// <item>a <c>trainable_cond_mask</c> Embedding(3, dim) — row 0 added to the main video tokens, row 1 to reference
/// tokens, row 2 to (unimplemented) motion tokens;</item>
/// <item>per-frame timesteps — the reference latent is appended as EXTRA TOKENS (patch-embedded with the shared
/// <c>patch_embedding</c>, RoPE at <c>t_start = max(30, T+9)</c>, timestep 0) while noisy frames run at the sampler
/// timestep, so the block/head AdaLN modulation is per latent frame (the TI2V multi-group path);</item>
/// <item>the audio injector (<see cref="WanS2VAudioInjector"/>) after each block in
/// <see cref="WanVideoConfig.AudioInjectLayers"/>, fed by <see cref="WanS2VAudioEncoder"/> tokens;</item>
/// <item>an optional <c>cond_encoder</c> Conv3d for the pose-control video — ComfyUI always feeds it (zeros when no
/// control), so with no control input its bias is still added to every main token.</item>
/// </list>
/// <b>TODO (FramePackMotioner):</b> the <c>frame_packer.*</c> weights (multi-clip motion-frame continuation) are
/// deliberately not consumed — single-clip generation never exercises them; port <c>FramePackMotioner</c> +
/// its negative-t_start RoPE when the autoregressive extend path is built. B=1.</summary>
public sealed unsafe class WanS2VTransformer : IDisposable
{
    private readonly WanVideoConfig _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly WanS2VAudioInjector _audioInjector;
    private readonly int[] _injectLayers;
    private readonly WanRope _rope;
    private readonly int _patchVec;
    private int _disposed;

    private Tensor? _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB;
    private Tensor? _textW1, _textB1, _textW2, _textB2;
    private Tensor? _condMask0, _condMask1;        // trainable_cond_mask.weight rows (main / reference tokens)
    private Tensor? _onesDim;                      // [1, dim] of 1s — the scale operand for GPU broadcast-adds
    private Tensor? _condEncW2d, _condEncB;        // cond_encoder Conv3d as linear (pose control)

    public WanS2VTransformer(WanVideoConfig config)
    {
        _config = config;
        _injectLayers = config.AudioInjectLayers;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        _audioInjector = new WanS2VAudioInjector(_injectLayers.Length, config.InnerDim, config.NumHeads, config.HeadDim, config.Eps);
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
        Tensor condMask = LoadF32(w, "trainable_cond_mask.weight");   // [3, dim]
        _condMask0 = CopyRow(condMask, 0, _config.InnerDim);
        _condMask1 = CopyRow(condMask, 1, _config.InnerDim);
        _onesDim = new Tensor(new TensorShape(1, _config.InnerDim), DType.F32);
        new Span<float>((float*)_onesDim.DataPointer, _config.InnerDim).Fill(1f);
        if (w.TryGetValue("cond_encoder.weight", out Tensor? condW))
        {
            _condEncW2d = WanDitOps.Reshape2d(condW, _config.InnerDim,
                _config.VaeLatentChannels * _config.PatchSize.T * _config.PatchSize.H * _config.PatchSize.W);
            _condEncB = LoadF32Opt(w, "cond_encoder.bias");
        }
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
        _audioInjector.LoadWeights(w);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2,
            _condMask0, _condMask1, _onesDim, _condEncW2d, _condEncB })
            if (t is not null) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
        foreach (Tensor t in _audioInjector.EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction. <paramref name="latent"/> is the noisy latent <c>[1, z, T, H, W]</c>;
    /// <paramref name="encoder"/> is umT5 features <c>[L, textDim]</c>; <paramref name="audioLocal"/> /
    /// <paramref name="audioGlobal"/> are the <see cref="WanS2VAudioEncoder"/> outputs (<c>[T, nTok, dim]</c> /
    /// <c>[T, 1, dim]</c>, pass null for no audio); <paramref name="referenceLatent"/> is an optional VAE-encoded
    /// reference image <c>[1, z, refT, H, W]</c> appended as extra tokens; <paramref name="controlLatent"/> is the
    /// optional pose-control latent <c>[1, z, T, H, W]</c>. Returns <c>[1, z, T, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float timestep,
        Tensor? audioLocal = null, Tensor? audioGlobal = null, Tensor? referenceLatent = null, Tensor? controlLatent = null)
    {
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int s = gt * gh * gw, dim = _config.InnerDim;
        int frameTokens = gh * gw;
        if (audioLocal is not null && (int)audioLocal.Shape[0] != gt)
            throw new ArgumentException($"audio tokens must have {gt} latent frames; got {audioLocal.Shape[0]}.", nameof(audioLocal));

        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);

        // Pose control: ComfyUI always feeds cond_encoder (zeros when absent) — conv over zeros = its bias everywhere.
        // All adds are backend ops into fresh tensors: host-mutating a cached op output would force a stream-sync +
        // D2H eviction per touch (the S2V host-glue slowdown), and the mutation itself never reaches the GPU copy.
        if (_condEncW2d is not null)
        {
            if (controlLatent is not null)
            {
                Tensor ctrl = WanDitOps.Patchify(backend, controlLatent, _config.VaeLatentChannels, dim, _config.PatchSize, _condEncW2d, _condEncB);
                Tensor summed = new Tensor(hidden.Shape, DType.F32);
                backend.Add(summed, hidden, ctrl);
                ctrl.Dispose(); hidden.Dispose();
                hidden = summed;
            }
            else if (_condEncB is not null)
            {
                hidden = AddRowGpu(backend, hidden, _condEncB);
            }
        }

        hidden = AddRowGpu(backend, hidden, _condMask0!);

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(gt, gh, gw);
        int totalS = s, groups = 1;
        float[] timesteps = [timestep];
        Tensor cur = hidden;

        if (referenceLatent is not null)
        {
            int refT = (int)referenceLatent.Shape[2] / pt;
            if ((int)referenceLatent.Shape[3] != hh || (int)referenceLatent.Shape[4] != ww)
                throw new ArgumentException($"reference latent spatial size must match the video latent ({hh}x{ww}); got {referenceLatent.Shape}.", nameof(referenceLatent));
            Tensor refTokens = WanDitOps.Patchify(backend, referenceLatent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);
            int refS = refT * frameTokens;
            refTokens = AddRowGpu(backend, refTokens, _condMask1!);

            // Reference RoPE sits far past the video frames on the temporal axis: t_start = max(30, T + 9).
            int tStart = Math.Max(30, gt + 9);
            int[] refFrames = new int[refT];
            for (int i = 0; i < refT; i++) refFrames[i] = tStart + i;
            (Tensor refCos, Tensor refSin) = _rope.BuildCosSin(refFrames, gh, gw);
            Tensor cos2 = WanDitOps.ConcatRows(cos, refCos, _config.HeadDim); cos.Dispose(); refCos.Dispose(); cos = cos2;
            Tensor sin2 = WanDitOps.ConcatRows(sin, refSin, _config.HeadDim); sin.Dispose(); refSin.Dispose(); sin = sin2;

            Tensor joined = new Tensor(new TensorShape(s + refS, dim), DType.F32);
            backend.Concat(joined, [cur, refTokens], 0);
            cur.Dispose(); refTokens.Dispose();
            cur = joined;
            totalS = s + refS;

            // Per-frame timesteps: the sampler timestep for noisy frames, 0 for the reference frame(s).
            groups = gt + refT;
            timesteps = new float[groups];
            for (int i = 0; i < gt; i++) timesteps[i] = timestep;
        }

        int tokensPerGroup = totalS / groups;
        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, timesteps, _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        Tensor encoderProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);

        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, tokensPerGroup);
            cur.Dispose();
            cur = next;
            int injIdx = Array.IndexOf(_injectLayers, i);
            if (injIdx >= 0 && audioLocal is not null && audioGlobal is not null)
            {
                Tensor injected = _audioInjector.Forward(backend, cur, injIdx, audioLocal, audioGlobal, s);
                cur.Dispose();
                cur = injected;
            }
        }
        cos.Dispose(); sin.Dispose(); timestepProj.Dispose(); encoderProj.Dispose();

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB,
            totalS, dim, _config.Eps, tokensPerGroup);
        cur.Dispose();
        temb.Dispose();
        // Unpatchify reads only the first gt·gh·gw rows — the appended reference tokens are dropped, as in the reference.
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        return outVel;
    }

    /// <summary>GPU broadcast row-add: consumes <paramref name="tokens"/> and returns a fresh tensor
    /// <c>tokens + row</c> (row <c>[1, dim]</c> broadcast over every token) via <c>AffineBroadcastLastDim</c>
    /// with a ones scale — keeps the stream GPU-resident instead of a host loop + cache eviction.</summary>
    private Tensor AddRowGpu(IBackend backend, Tensor tokens, Tensor row)
    {
        Tensor o = new Tensor(tokens.Shape, DType.F32);
        backend.AffineBroadcastLastDim(o, tokens, _onesDim!, row);
        tokens.Dispose();
        return o;
    }

    /// <summary>Copies row <paramref name="row"/> of a <c>[rows, dim]</c> F32 tensor into a standalone <c>[1, dim]</c>.</summary>
    private static Tensor CopyRow(Tensor x, int row, int dim)
    {
        Tensor o = new Tensor(new TensorShape(1, dim), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer + (long)row * dim, (float*)o.DataPointer, (long)dim * 4, (long)dim * 4);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
    private static Tensor? LoadF32Opt(IReadOnlyDictionary<string, Tensor> w, string key) => w.TryGetValue(key, out Tensor? t) ? (t.DType == DType.F32 ? t : t.CastTo(DType.F32)) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
            _condMask0?.Dispose(); _condMask1?.Dispose(); _onesDim?.Dispose();
            _condMask0 = _condMask1 = _onesDim = _condEncW2d = _condEncB = null;
        }
        GC.SuppressFinalize(this);
    }
}
