using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Matrix-Game 2.0 DiT (<c>CausalWanModel</c>) — the SkyReels-V2 / Wan2.1 1.3B I2V transformer with the text
/// branch removed: cross-attention consumes the CLIP-ViT-H visual context (projected by the I2V <c>img_emb</c> MLP),
/// per-block <see cref="MatrixGame3ActionModule"/>s (the shared GameFactory module) ride the same post-cross-attn
/// hook as Matrix-Game 3, self-attention is <b>block-causal with a sliding local window</b> (additive mask), and
/// timesteps are per-frame (Self-Forcing context frames condition at t=0). Reuses <see cref="WanVideoBlock"/> /
/// <see cref="WanRope"/> / <see cref="WanDitOps"/> verbatim. The KV-cache formulation is replaced by an equivalent
/// joint forward over [window context ‖ current block] — a documented recompute approximation until the rolling
/// cache lands. Numerics validation-pending. See <c>docs/Research/MATRIX_GAME_2_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class MatrixGame2Transformer : IDisposable
{
    private readonly MatrixGame2Config _config;
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
    private readonly WanImageEmbedder _imgEmbedder;   // img_emb MLPProj (shared with Wan I2V / Animate)

    public MatrixGame2Transformer(MatrixGame2Config config)
    {
        _config = config;
        WanVideoConfig wan = config.ToWanConfig();
        _blocks = new WanVideoBlock[config.NumLayers];
        _actionModules = new MatrixGame3ActionModule?[config.NumLayers];
        HashSet<int>? actionSet = config.ActionBlocks is null ? null : [.. config.ActionBlocks];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _blocks[i] = new WanVideoBlock(wan, crossAttnNorm: true) { FfnDtype = DiTBlocks.DitDtype.Act };
            if (actionSet is null || actionSet.Contains(i))
                _actionModules[i] = new MatrixGame3ActionModule(config.InnerDim,
                    keyboardDimIn: config.KeyboardDim, hiddenSize: config.ActionHiddenSize,
                    streamHiddenDim: config.ActionStreamDim, headsNum: config.ActionHeads,
                    vaeTimeCompressionRatio: config.VaeTemporalCompression, windowsSize: config.ActionWindowSize,
                    ropeTheta: config.ActionRopeTheta, eps: config.Eps,
                    enableMouse: config.EnableMouse);
        }
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _imgEmbedder = new WanImageEmbedder(config.Eps);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public MatrixGame2Config Config => _config;

    /// <summary>Loads the diffusers-renamed Wan core (the Matrix-Game converter handles the original naming) plus the
    /// I2V <c>img_emb</c> projection (<c>condition_embedder.image_embedder.*</c> after renames) and per-block
    /// <c>action_model</c> weights.</summary>
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

        // Shared Wan I2V MLPProj — shape-driven (intermediate width from ff.net.0.proj rows). The old duplicated
        // projection here hardcoded the intermediate buffer to [L, clipDim]; a checkpoint whose first Linear widens
        // (rows > clipDim) made backend.Linear write rows·L floats into that undersized buffer → heap corruption
        // (the MatrixGame2 testhost crash). backend.Linear N always comes from the WEIGHT, the buffer from the caller.
        if (!_imgEmbedder.TryLoadWeights(w))
            throw new ArgumentException("Matrix-Game 2 checkpoint is missing condition_embedder.image_embedder.* weights.", nameof(w));

        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(w, $"blocks.{i}");
            if (_actionModules[i] is not null && w.ContainsKey($"blocks.{i}.action_model.keyboard_embed.0.weight"))
                _actionModules[i]!.LoadWeights(w, $"blocks.{i}.action_model");
            else
                _actionModules[i] = null;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB })
            if (t is not null) yield return t;
        foreach (Tensor t in _imgEmbedder.EnumerateWeights()) yield return t;
        for (int i = 0; i < _blocks.Length; i++)
        {
            foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
            if (_actionModules[i] is not null)
                foreach (Tensor t in _actionModules[i]!.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Velocity prediction over [context ‖ current] frames. <paramref name="latent"/> is
    /// <c>[1, 36, T_total, H, W]</c> (noisy/clean 16ch + 4ch mask + 16ch image-condition, assembled by the pipeline);
    /// <paramref name="clipContext"/> is the raw CLIP penultimate hidden states <c>[257, 1280]</c>;
    /// <paramref name="frameTimesteps"/> has one value per latent frame (clean context at 0);
    /// <paramref name="ropeFrameIndices"/> are the global frame positions; <paramref name="outputFrames"/> is the
    /// trailing block to read out. Returns <c>[1, 16, outputFrames, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor clipContext, float[] frameTimesteps,
        ReadOnlySpan<int> ropeFrameIndices, int outputFrames, Tensor? mouse, Tensor? keyboard)
        => Forward(backend, latent, clipContext, frameTimesteps, ropeFrameIndices, outputFrames, mouse, keyboard, null);

    /// <summary>Forward with optional per-stage debug taps (parity bisection): <c>patch</c>, <c>ctx</c>,
    /// <c>block0</c>, <c>blockLast</c> captured as fresh CPU copies.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor clipContext, float[] frameTimesteps,
        ReadOnlySpan<int> ropeFrameIndices, int outputFrames, Tensor? mouse, Tensor? keyboard, Dictionary<string, Tensor>? taps)
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
        if (outputFrames <= 0 || outputFrames > gt)
            throw new ArgumentException($"outputFrames must be in (0, {gt}].", nameof(outputFrames));

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(ropeFrameIndices, gh, gw);
        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, frameTimesteps, _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        Tensor encoderProj = _imgEmbedder.Forward(backend, clipContext, dim);
        Tensor mask = BuildBlockCausalMask(ropeFrameIndices, gh * gw, _config.LocalAttnSize, _config.NumFramePerBlock);
        if (taps is not null) { taps["patch"] = CloneCpu(hidden); taps["ctx"] = CloneCpu(encoderProj); }

        int tokensPerGroup = s / gt;
        int[] frameIdx = ropeFrameIndices.ToArray();
        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            MatrixGame3ActionModule? action = _actionModules[i];
            Action<Tensor>? hook = action is null || keyboard is null
                ? null
                : h => action.Forward(backend, h, (gt, gh, gw), frameIdx, 0, mouse ?? keyboard, keyboard);
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, tokensPerGroup, hook, mask);
            cur.Dispose();
            cur = next;
            if (taps is not null && i == 0) taps["block0"] = CloneCpu(cur);
            if (taps is not null && i == _blocks.Length - 1) taps["blockLast"] = CloneCpu(cur);
        }
        cos.Dispose(); sin.Dispose(); timestepProj.Dispose(); encoderProj.Dispose(); mask.Dispose();

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

    private static Tensor CloneCpu(Tensor src)
    {
        Tensor dst = new Tensor(src.Shape, DType.F32);
        new ReadOnlySpan<float>((float*)src.DataPointer, (int)src.ElementCount)
            .CopyTo(new Span<float>((float*)dst.DataPointer, (int)dst.ElementCount));
        return dst;
    }

    /// <summary>Builds the additive block-causal + sliding-window self-attention mask <c>[1, 1, S, S]</c>: token i
    /// (global frame f_i) attends token j (frame f_j) iff <c>block(f_j) ≤ block(f_i)</c> and f_j is within the local
    /// window below the querying block's start (<c>localAttnSize ≤ 0</c> = full block-causal).</summary>
    public static Tensor BuildBlockCausalMask(ReadOnlySpan<int> frameIndices, int tokensPerFrame, int localAttnSize, int framesPerBlock)
    {
        int gt = frameIndices.Length;
        int s = gt * tokensPerFrame;
        Tensor mask = new Tensor(new TensorShape([1L, 1, s, s]), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int i = 0; i < s; i++)
        {
            int fi = frameIndices[i / tokensPerFrame];
            int bi = fi / framesPerBlock;
            for (int j = 0; j < s; j++)
            {
                int fj = frameIndices[j / tokensPerFrame];
                bool allowed = fj / framesPerBlock <= bi
                    && (localAttnSize <= 0 || fj >= bi * framesPerBlock - localAttnSize);
                mp[(long)i * s + j] = allowed ? 0f : -1e9f;
            }
        }
        return mask;
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
            _imgEmbedder.Clear();
        }
        GC.SuppressFinalize(this);
    }
}
