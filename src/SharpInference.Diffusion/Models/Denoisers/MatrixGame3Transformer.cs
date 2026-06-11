using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Matrix-Game 3.0 DiT — the Wan2.2 transformer with three additions (per <c>wan/modules/model.py</c>):
/// (1) a <b>memory-augmented sequence</b>: FOV-retrieved memory latents and clean past-overlap latents are patchified
/// by the same Conv3d as the noisy segment and concatenated on the temporal axis, with per-frame timesteps (clean
/// frames at 0) and RoPE t-indices preserving the memory frames' historical positions; (2) a per-block
/// <see cref="MatrixGame3ActionModule"/> (mouse self-attn / keyboard cross-attn) inserted between cross-attention and
/// FFN on the configured block subset; (3) a <b>Plücker camera projection</b> added to the patch embeddings.
/// Reuses <see cref="WanVideoBlock"/>/<see cref="WanRope"/>/<see cref="WanDitOps"/> directly — only the sequence
/// assembly and readout differ from <see cref="WanVideoTransformer"/>. Numerics are validation-pending.
/// See <c>docs/Research/MATRIX_GAME_3_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class MatrixGame3Transformer : IDisposable
{
    private readonly MatrixGame3Config _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly MatrixGame3ActionModule?[] _actionModules;
    private readonly WanRope _rope;
    private readonly int _patchVec;
    private int _disposed;

    private Tensor? _patchW2d, _patchB;
    private Tensor? _projOutW, _projOutB;
    private Tensor? _finalScaleShift;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B;
    private Tensor? _timeProjW, _timeProjB;
    private Tensor? _textW1, _textB1, _textW2, _textB2;
    private Tensor? _pluckerProjW, _pluckerProjB;          // patch_embedding_wancamctrl
    private Tensor? _pluckerH1W, _pluckerH1B, _pluckerH2W, _pluckerH2B;   // c2ws_hidden_states_layer1/2

    public MatrixGame3Transformer(MatrixGame3Config config)
    {
        _config = config;
        WanVideoConfig wan = config.ToWanConfig();
        _blocks = new WanVideoBlock[config.NumLayers];
        _actionModules = new MatrixGame3ActionModule?[config.NumLayers];
        HashSet<int>? actionSet = config.ActionBlocks is null ? null : [.. config.ActionBlocks];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks[i] = new WanVideoBlock(wan, crossAttnNorm: true);
            if (actionSet is null || actionSet.Contains(i))
                _actionModules[i] = new MatrixGame3ActionModule(config.InnerDim,
                    hiddenSize: config.ActionHiddenSize, streamHiddenDim: config.ActionStreamDim,
                    headsNum: config.ActionHeads, vaeTimeCompressionRatio: config.VaeTemporalCompression,
                    windowsSize: config.ActionWindowSize, ropeTheta: config.ActionRopeTheta, eps: config.Eps);
        }
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public MatrixGame3Config Config => _config;

    /// <summary>Loads the core DiT (diffusers-renamed Wan keys, as produced by the Matrix-Game converter) plus the
    /// Matrix-Game-specific weights (original naming: <c>blocks.{i}.action_model.*</c>, the Plücker projection,
    /// <c>c2ws_hidden_states_layer1/2</c>). Action/Plücker weights are optional — a checkpoint slice without them
    /// loads as a plain Wan2.2 DiT (useful for shape bring-up).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        Tensor pw = w["patch_embedding.weight"];
        _patchW2d = WanDitOps.Reshape2d(pw, _config.InnerDim, _patchVec);
        w.TryGetValue("patch_embedding.bias", out _patchB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _finalScaleShift = LoadF32(w, "scale_shift_table");
        _timeEmb1W = w["condition_embedder.time_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["condition_embedder.time_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_2.bias", out _timeEmb2B);
        _timeProjW = w["condition_embedder.time_proj.weight"]; w.TryGetValue("condition_embedder.time_proj.bias", out _timeProjB);
        _textW1 = w["condition_embedder.text_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_1.bias", out _textB1);
        _textW2 = w["condition_embedder.text_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_2.bias", out _textB2);

        if (w.TryGetValue("patch_embedding_wancamctrl.weight", out _pluckerProjW))
        {
            w.TryGetValue("patch_embedding_wancamctrl.bias", out _pluckerProjB);
            w.TryGetValue("c2ws_hidden_states_layer1.weight", out _pluckerH1W); w.TryGetValue("c2ws_hidden_states_layer1.bias", out _pluckerH1B);
            w.TryGetValue("c2ws_hidden_states_layer2.weight", out _pluckerH2W); w.TryGetValue("c2ws_hidden_states_layer2.bias", out _pluckerH2B);
        }

        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(w, $"blocks.{i}");
            if (_actionModules[i] is not null && w.ContainsKey($"blocks.{i}.action_model.t_qkv.weight"))
                _actionModules[i]!.LoadWeights(w, $"blocks.{i}.action_model");
            else
                _actionModules[i] = null;   // checkpoint has no module on this block
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2,
            _pluckerProjW, _pluckerProjB, _pluckerH1W, _pluckerH1B, _pluckerH2W, _pluckerH2B })
            if (t is not null) yield return t;
        for (int i = 0; i < _blocks.Length; i++)
        {
            foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
            if (_actionModules[i] is not null)
                foreach (Tensor t in _actionModules[i]!.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Velocity prediction over the assembled memory-augmented sequence.
    /// <paramref name="latent"/> is <c>[1, 48, T_total, H, W]</c> = memory ‖ past ‖ noisy-current on the temporal axis;
    /// <paramref name="frameTimesteps"/> has one value per latent frame (clean frames 0, noisy frames t·1000);
    /// <paramref name="ropeFrameIndices"/> assigns each frame its RoPE t-position (memory keeps historical indices);
    /// <paramref name="memoryFrames"/> counts the leading conditioning-only frames; <paramref name="outputFrames"/> is
    /// the trailing frame count to read out. <paramref name="mouse"/>/<paramref name="keyboard"/> are RGB-rate action
    /// rows covering the non-memory frames; <paramref name="pluckerTokens"/> is the optional per-token camera-ray
    /// input <c>[S_total, PluckerPatchDim]</c>. Returns <c>[1, 48, outputFrames, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor encoder, float[] frameTimesteps,
        ReadOnlySpan<int> ropeFrameIndices, int memoryFrames, int outputFrames,
        Tensor? mouse, Tensor? keyboard, Tensor? pluckerTokens)
    {
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int s = gt * gh * gw;
        int dim = _config.InnerDim;
        if (frameTimesteps.Length != gt)
            throw new ArgumentException($"frameTimesteps must have {gt} entries.", nameof(frameTimesteps));
        if (ropeFrameIndices.Length != gt)
            throw new ArgumentException($"ropeFrameIndices must have {gt} entries.", nameof(ropeFrameIndices));
        if (outputFrames <= 0 || outputFrames > gt - memoryFrames)
            throw new ArgumentException($"outputFrames must be in (0, {gt - memoryFrames}].", nameof(outputFrames));

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(ropeFrameIndices, gh, gw);
        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);
        if (pluckerTokens is not null) AddPlucker(backend, hidden, pluckerTokens, s, dim);

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, frameTimesteps, _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        Tensor encoderProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);

        int tokensPerGroup = s / gt;
        int[] frameIdx = ropeFrameIndices.ToArray();
        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            MatrixGame3ActionModule? action = _actionModules[i];
            Action<Tensor>? hook = action is null || mouse is null || keyboard is null
                ? null
                : h => action.Forward(backend, h, (gt, gh, gw), frameIdx, memoryFrames, mouse, keyboard);
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, tokensPerGroup, hook);
            cur.Dispose();
            cur = next;
        }
        cos.Dispose(); sin.Dispose(); timestepProj.Dispose(); encoderProj.Dispose();

        // Read out only the trailing outputFrames frames (memory/past are conditioning, not output).
        int skipFrames = gt - outputFrames;
        Tensor outTokens = SliceFrames(cur, skipFrames, outputFrames, tokensPerGroup, dim);
        cur.Dispose();
        Tensor outTemb = SliceRows(temb, skipFrames, outputFrames, dim);
        temb.Dispose();

        Tensor projected = WanDitOps.FinalLayer(backend, outTokens, outTemb, _finalScaleShift!, _projOutW!, _projOutB,
            outputFrames * tokensPerGroup, dim, _config.Eps, tokensPerGroup);
        outTokens.Dispose();
        outTemb.Dispose();
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, outputFrames, gh, gw, _config.PatchSize);
        projected.Dispose();
        return outVel;
    }

    /// <summary>Projects per-token Plücker rays through <c>patch_embedding_wancamctrl</c> (+ the two hidden refinement
    /// layers, residual) and adds them to the patch embeddings in place. Injection point is validation-gated.</summary>
    private void AddPlucker(IBackend backend, Tensor hidden, Tensor pluckerTokens, int s, int dim)
    {
        if (_pluckerProjW is null) return;
        if ((int)pluckerTokens.Shape[0] != s)
            throw new ArgumentException($"pluckerTokens rows {pluckerTokens.Shape[0]} != token count {s}.", nameof(pluckerTokens));
        Tensor proj = new Tensor(new TensorShape(s, dim), DType.F32);
        backend.Linear(proj, pluckerTokens, _pluckerProjW, _pluckerProjB);
        if (_pluckerH1W is not null && _pluckerH2W is not null)
        {
            Tensor h1 = new Tensor(new TensorShape(s, dim), DType.F32);
            backend.Linear(h1, proj, _pluckerH1W, _pluckerH1B);
            Tensor a = new Tensor(h1.Shape, DType.F32); backend.Gelu(a, h1); h1.Dispose();
            Tensor h2 = new Tensor(new TensorShape(s, dim), DType.F32);
            backend.Linear(h2, a, _pluckerH2W, _pluckerH2B);
            a.Dispose();
            float* pp = (float*)proj.DataPointer;
            float* hp2 = (float*)h2.DataPointer;
            for (long i = 0; i < (long)s * dim; i++) pp[i] += hp2[i];
            h2.Dispose();
        }
        float* hd = (float*)hidden.DataPointer;
        float* ppf = (float*)proj.DataPointer;
        for (long i = 0; i < (long)s * dim; i++) hd[i] += ppf[i];
        proj.Dispose();
    }

    private static Tensor SliceFrames(Tensor tokens, int skipFrames, int frames, int tokensPerFrame, int dim)
    {
        Tensor o = new Tensor(new TensorShape(frames * tokensPerFrame, dim), DType.F32);
        long offset = (long)skipFrames * tokensPerFrame * dim;
        long count = (long)frames * tokensPerFrame * dim;
        Buffer.MemoryCopy((float*)tokens.DataPointer + offset, (float*)o.DataPointer, count * 4, count * 4);
        return o;
    }

    private static Tensor SliceRows(Tensor rows, int skip, int count, int dim)
    {
        Tensor o = new Tensor(new TensorShape(count, dim), DType.F32);
        Buffer.MemoryCopy((float*)rows.DataPointer + (long)skip * dim, (float*)o.DataPointer, (long)count * dim * 4, (long)count * dim * 4);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
            _pluckerProjW = _pluckerProjB = _pluckerH1W = _pluckerH1B = _pluckerH2W = _pluckerH2B = null;
        }
        GC.SuppressFinalize(this);
    }
}
