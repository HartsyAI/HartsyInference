using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Wan VACE DiT (<c>WanVACETransformer3DModel</c>), ported from diffusers <c>transformer_wan_vace.py</c>. The
/// base Wan video transformer plus a parallel control branch: a <c>vace_patch_embedding</c> (Conv3d
/// <c>VaceInChannels→inner</c>) + <see cref="WanVaceBlock"/>s (one per <see cref="WanVideoConfig.VaceLayers"/> entry)
/// that run on the VAE-encoded control video and emit per-layer hints, which are added into the main stream at the
/// matching block indices (scaled by the per-layer control scale). Reuses <see cref="WanDitOps"/> + the main
/// <see cref="WanVideoBlock"/> exactly as <see cref="WanVideoTransformer"/> does. B=1; numerics validation-pending.</summary>
public sealed unsafe class WanVaceTransformer : IDisposable
{
    private readonly WanVideoConfig _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly WanVaceBlock[] _vaceBlocks;
    private readonly int[] _vaceLayers;
    private readonly WanRope _rope;
    private readonly DiTBlocks.WanForwardCaches _caches = new();
    private readonly int _patchVec, _vacePatchVec;
    private int _disposed;

    private Tensor? _patchW2d, _patchB, _vacePatchW2d, _vacePatchB;
    private Tensor? _projOutW, _projOutB, _finalScaleShift;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB;
    private Tensor? _textW1, _textB1, _textW2, _textB2;

    public WanVaceTransformer(WanVideoConfig config)
    {
        _config = config;
        _vaceLayers = config.VaceLayers;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        _vaceBlocks = new WanVaceBlock[_vaceLayers.Length];
        for (int i = 0; i < _vaceLayers.Length; i++) _vaceBlocks[i] = new WanVaceBlock(config, applyInputProjection: i == 0);
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
        _vacePatchVec = config.VaceInChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public WanVideoConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _patchW2d = WanDitOps.Reshape2d(w["patch_embedding.weight"], _config.InnerDim, _patchVec);
        w.TryGetValue("patch_embedding.bias", out _patchB);
        _vacePatchW2d = WanDitOps.Reshape2d(w["vace_patch_embedding.weight"], _config.InnerDim, _vacePatchVec);
        w.TryGetValue("vace_patch_embedding.bias", out _vacePatchB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _finalScaleShift = LoadF32(w, "scale_shift_table");
        _timeEmb1W = w["condition_embedder.time_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["condition_embedder.time_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_2.bias", out _timeEmb2B);
        _timeProjW = w["condition_embedder.time_proj.weight"]; w.TryGetValue("condition_embedder.time_proj.bias", out _timeProjB);
        _textW1 = w["condition_embedder.text_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_1.bias", out _textB1);
        _textW2 = w["condition_embedder.text_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_2.bias", out _textB2);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
        for (int i = 0; i < _vaceBlocks.Length; i++) _vaceBlocks[i].LoadWeights(w, $"vace_blocks.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _vacePatchW2d, _vacePatchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2 })
            if (t is not null) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
        for (int i = 0; i < _vaceBlocks.Length; i++) foreach (Tensor t in _vaceBlocks[i].EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction with VACE control. <paramref name="latent"/> is <c>[1, inChannels, T, H, W]</c>;
    /// <paramref name="control"/> is the VAE-encoded control context <c>[1, VaceInChannels, T, H, W]</c>;
    /// <paramref name="encoder"/> is umT5 features <c>[L, textDim]</c>; <paramref name="controlScales"/> is one scale per
    /// VACE layer (null = all 1). Returns <c>[1, outChannels, T, H, W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor control, Tensor encoder, float timestep,
        float[]? controlScales = null)
    {
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int s = gt * gh * gw;
        int dim = _config.InnerDim;
        float[] scales = controlScales ?? System.Linq.Enumerable.Repeat(1f, _vaceLayers.Length).ToArray();
        if (scales.Length != _vaceLayers.Length)
            throw new ArgumentException($"controlScales must have {_vaceLayers.Length} entries; got {scales.Length}.", nameof(controlScales));

        (Tensor cos, Tensor sin) = _caches.RopeCosSin((gt, gh, gw, 0), () => _rope.BuildCosSin(gt, gh, gw));

        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);
        Tensor controlTokens = PatchifyControl(backend, control, s, dim);

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, [timestep], _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        Tensor encoderProj = _caches.TextProj(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);

        // VACE control branch: run all control blocks, collecting per-layer hints (reverse to pop in main-loop order).
        Tensor[] hints = new Tensor[_vaceBlocks.Length];
        Tensor ctrl = controlTokens;
        for (int i = 0; i < _vaceBlocks.Length; i++)
        {
            (Tensor hint, Tensor nextCtrl) = _vaceBlocks[i].Forward(backend, hidden, encoderProj, ctrl, timestepProj, _rope, cos, sin, s);
            ctrl.Dispose();
            ctrl = nextCtrl;
            hints[i] = hint;
        }
        ctrl.Dispose();

        // Main blocks; add the matching hint after each VACE-layer block.
        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, s);
            cur.Dispose();
            cur = next;
            int hintIdx = Array.IndexOf(_vaceLayers, i);
            if (hintIdx >= 0)
            {
                // GPU-resident hint add (Scale + Add into a fresh tensor) — a host loop here would stream-sync +
                // evict the block output from the activation cache on every VACE layer.
                Tensor scaled = new Tensor(hints[hintIdx].Shape, DType.F32);
                backend.Scale(scaled, hints[hintIdx], scales[hintIdx]);
                Tensor fused = new Tensor(cur.Shape, DType.F32);
                backend.Add(fused, cur, scaled);
                scaled.Dispose();
                cur.Dispose();
                cur = fused;
                hints[hintIdx].Dispose();
                hints[hintIdx] = null!;
            }
        }
        // Every VACE layer index is < NumLayers, so all hints were added + disposed above; guard just in case.
        foreach (Tensor? h in hints) h?.Dispose();
        timestepProj.Dispose();   // cos/sin/encoderProj are owned by the per-generation caches

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, s, dim, _config.Eps, s);
        cur.Dispose();
        temb.Dispose();
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        return outVel;
    }

    /// <summary>Patchifies the control video and zero-pads its token count up to the main stream length <paramref name="s"/>.</summary>
    private Tensor PatchifyControl(IBackend backend, Tensor control, int s, int dim)
    {
        Tensor tokens = WanDitOps.Patchify(backend, control, _config.VaceInChannels, dim, _config.PatchSize, _vacePatchW2d!, _vacePatchB);
        int sc = (int)tokens.Shape[0];
        if (sc == s) return tokens;
        if (sc > s)
            throw new ArgumentException($"control tokens {sc} exceed main tokens {s}.");
        Tensor padded = new Tensor(new TensorShape(s, dim), DType.F32);
        new Span<float>((float*)padded.DataPointer, s * dim).Clear();
        Buffer.MemoryCopy((float*)tokens.DataPointer, (float*)padded.DataPointer, (long)sc * dim * 4, (long)sc * dim * 4);
        tokens.Dispose();
        return padded;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _vacePatchW2d = _vacePatchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
        }
        GC.SuppressFinalize(this);
    }
}
