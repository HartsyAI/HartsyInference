using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Oasis-500m DiT-S/2 (Decart/Etched, MIT) — the action-conditioned spatio-temporal world-model backbone,
/// ported from <c>open-oasis/dit.py</c>. v-prediction over a window of 16-channel latent frames: patchify 2×2 →
/// 16 × <see cref="OasisSpatioTemporalBlock"/> (spatial bidirectional + temporal causal axial attention) → adaLN final
/// layer → unpatchify. Per-frame conditioning <c>c[t] = TimestepEmbed(noiseIdx[t]) + Linear(action[t])</c> — Diffusion
/// Forcing gives every frame its own integer noise index, and the 25-dim VPT action vector is added straight into the
/// timestep embedding (the <c>TimestepAddon</c> conditioning pattern, vs Matrix-Game's attention streams). Numerics
/// validation-pending. See <c>docs/Research/OASIS_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class OasisDit : IDisposable
{
    private readonly OasisDitConfig _config;
    private readonly OasisSpatioTemporalBlock[] _blocks;
    private readonly AxialRope2D _spatialRope;
    private readonly WanRope _temporalRope;
    private int _disposed;

    private Tensor? _patchW2d, _patchB;               // x_embedder Conv2d(16→dim, 2, stride 2) as linear
    private Tensor? _tEmb1W, _tEmb1B, _tEmb2W, _tEmb2B;
    private Tensor? _extCondW, _extCondB;             // Linear(25 → dim)
    private Tensor? _finalModW, _finalModB;           // FinalLayer adaLN Linear(dim → 2·dim)
    private Tensor? _finalW, _finalB;                 // Linear(dim → p²·C)

    // ── Step-graph state (HARTSY_DIT_GRAPH): the 16-block loop is fully device-resident, so it captures once and
    //    replays with a single cuGraphLaunch per DDIM step — the interactive AR loop has FIXED geometry, so the
    //    signature never flips and the graph is reused every forward. Host patchify/cond-build/final-layer stay
    //    OUTSIDE capture, refreshing the fixed input/cond buffers before each launch. Geometry-invariant RoPE
    //    tables + causal mask are cached per (t,gh,gw) sig so the captured region reads stable device addresses. ──
    private const int GraphCaptureCall = 3;
    private long _geomSig = long.MinValue;
    private Tensor? _sCos, _sSin, _tCos, _tSin, _mask;
    private Tensor? _xFixed, _condFixed, _blockOutFixed;
    private long _graphSig = long.MinValue;
    private int _graphSigCalls;
    private bool _graphDead;

    public OasisDit(OasisDitConfig config)
    {
        _config = config;
        _blocks = new OasisSpatioTemporalBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++) _blocks[i] = new OasisSpatioTemporalBlock(config);
        _spatialRope = new AxialRope2D(config.SpatialRopeDimPerAxis, config.SpatialRopeMaxFreq);
        // Temporal RoPE: standard 1-D over the full head dim, θ=10000 (lucidrains "lang" mode).
        _temporalRope = new WanRope(config.HiddenSize / config.NumHeads, 0, 0, 10000.0);
    }

    public OasisDitConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int patchVec = _config.InChannels * _config.PatchSize * _config.PatchSize;
        _patchW2d = WanDitOps.Reshape2d(w["x_embedder.proj.weight"], _config.HiddenSize, patchVec);
        w.TryGetValue("x_embedder.proj.bias", out _patchB);
        _tEmb1W = w["t_embedder.mlp.0.weight"]; w.TryGetValue("t_embedder.mlp.0.bias", out _tEmb1B);
        _tEmb2W = w["t_embedder.mlp.2.weight"]; w.TryGetValue("t_embedder.mlp.2.bias", out _tEmb2B);
        _extCondW = w["external_cond.weight"]; w.TryGetValue("external_cond.bias", out _extCondB);
        _finalModW = w["final_layer.adaLN_modulation.1.weight"]; w.TryGetValue("final_layer.adaLN_modulation.1.bias", out _finalModB);
        _finalW = w["final_layer.linear.weight"]; w.TryGetValue("final_layer.linear.bias", out _finalB);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _tEmb1W, _tEmb1B, _tEmb2W, _tEmb2B,
            _extCondW, _extCondB, _finalModW, _finalModB, _finalW, _finalB })
            if (t is not null) yield return t;
        foreach (OasisSpatioTemporalBlock b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>v-prediction over the window. <paramref name="latents"/> is <c>[T, C, gridH, gridW]</c> (scaled latent
    /// space); <paramref name="noiseIndices"/> one integer per frame (Diffusion Forcing — context frames at the
    /// stabilization level, the target at its DDIM index); <paramref name="actions"/> is <c>[T, 25]</c>.
    /// Returns <c>[T, C, gridH, gridW]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latents, ReadOnlySpan<int> noiseIndices, Tensor actions)
        => Forward(backend, latents, noiseIndices, actions, null);

    /// <summary>Forward with optional per-stage debug taps (parity bisection): <c>c</c>, <c>xembed</c>
    /// (post patch-embed), <c>block0</c> (after the first block) captured as fresh copies.</summary>
    public Tensor Forward(IBackend backend, Tensor latents, ReadOnlySpan<int> noiseIndices, Tensor actions, Dictionary<string, Tensor>? taps)
    {
        int t = (int)latents.Shape[0];
        int gh = (int)latents.Shape[2] / _config.PatchSize;
        int gw = (int)latents.Shape[3] / _config.PatchSize;
        int sp = gh * gw;
        int dim = _config.HiddenSize;
        if (noiseIndices.Length != t) throw new ArgumentException($"need {t} noise indices.", nameof(noiseIndices));
        if ((int)actions.Shape[0] != t || (int)actions.Shape[1] != _config.ExternalCondDim)
            throw new ArgumentException($"actions must be [{t}, {_config.ExternalCondDim}].", nameof(actions));
        if (t > _config.MaxFrames) throw new ArgumentException($"window {t} exceeds max_frames {_config.MaxFrames}.", nameof(latents));

        bool prof = taps is null && Environment.GetEnvironmentVariable("HARTSY_OASIS_PHASE") == "1";
        long tsEntry = System.Diagnostics.Stopwatch.GetTimestamp();

        Tensor cond = BuildCondition(backend, noiseIndices, actions, t, dim);
        Tensor x = Patchify(backend, latents, t, gh, gw);
        if (taps is not null) { taps["c"] = CloneCpu(cond); taps["xembed"] = CloneCpu(x); }

        EnsureGeometry(backend, t, gh, gw);   // cache RoPE tables + causal mask per geometry (stable device buffers)

        // ── Step-graph path: capture the device-resident 16-block loop once, replay it every forward. Requires
        //    the plain (no-taps) path; falls back to eager cleanly on any capture failure. ──
        bool graphMode = DiTBlocks.DitStepGraph.Enabled && backend.StepGraphSupported && !_graphDead && taps is null;
        if (graphMode)
        {
            int total = t * sp;
            _xFixed ??= new Tensor(new TensorShape(total, dim), DType.F32);
            _condFixed ??= new Tensor(new TensorShape(t, dim), DType.F32);
            _blockOutFixed ??= new Tensor(new TensorShape(total, dim), DType.F32);
            backend.CopyInto(_xFixed, x); x.Dispose();
            backend.CopyInto(_condFixed, cond); cond.Dispose();

            long sig = ((long)t << 40) ^ ((long)gh << 20) ^ gw;
            bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
            if (sig != _graphSig || ownerLost)
            {
                backend.StepGraphReset();
                _graphSig = sig; _graphSigCalls = 0; backend.StepGraphOwner = this;
            }
            _graphSigCalls++;

            if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
            {
                if (prof)
                {
                    backend.Sync(); double setupMs = ElapsedMs(tsEntry); long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
                    backend.StepGraphLaunch(); backend.Sync(); double replayMs = ElapsedMs(t1); long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
                    Tensor ov = FinalLayer(backend, _blockOutFixed, _condFixed, t, sp, gh, gw); backend.Sync(); double finalMs = ElapsedMs(t2);
                    HartsyInference.Core.Logging.Logs.Info($"[oasis-phase] setup(patchify+cond+copyfixed)={setupMs:F2} graph-replay={replayMs:F2} final(mod+unpatch)={finalMs:F2} ms");
                    return ov;
                }
                backend.StepGraphLaunch();
                return FinalLayer(backend, _blockOutFixed, _condFixed, t, sp, gh, gw);
            }

            bool capture = _graphSigCalls == GraphCaptureCall;
            if (capture) backend.StepGraphBegin();
            try
            {
                RunBlocksIntoFixed(backend, t, sp);
            }
            catch (Exception ex) when (capture)
            {
                backend.StepGraphReset(); _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[Oasis graph] capture invalidated — eager fallback: {ex.Message}");
                RunBlocksIntoFixed(backend, t, sp);
                return FinalLayer(backend, _blockOutFixed, _condFixed, t, sp, gh, gw);
            }
            if (capture)
            {
                try { backend.StepGraphEndAndLaunch(); HartsyInference.Core.Logging.Logs.Info("[Oasis graph] block loop captured; replaying via cuGraphLaunch."); }
                catch (Exception ex)
                {
                    backend.StepGraphReset(); _graphDead = true;
                    HartsyInference.Core.Logging.Logs.Warning($"[Oasis graph] capture failed — eager fallback: {ex.Message}");
                    RunBlocksIntoFixed(backend, t, sp);
                }
            }
            return FinalLayer(backend, _blockOutFixed, _condFixed, t, sp, gh, gw);
        }

        Tensor cur = x;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, cond, t, sp, _sCos!, _sSin!, _tCos!, _tSin!, _mask!);
            cur.Dispose();
            cur = next;
            if (taps is not null && i == 0) taps["block0"] = CloneCpu(cur);
            if (taps is not null && i == _blocks.Length - 1) taps["blockLast"] = CloneCpu(cur);
        }
        Tensor outV = FinalLayer(backend, cur, cond, t, sp, gh, gw);
        cur.Dispose();
        cond.Dispose();
        return outV;
    }

    /// <summary>Runs the 16-block loop over the fixed input/cond buffers and lands the result in
    /// <see cref="_blockOutFixed"/> (a stable, non-graph-owned buffer — the last op the captured graph records,
    /// so replay's output survives the graph-memory auto-free).</summary>
    private void RunBlocksIntoFixed(IBackend backend, int t, int sp)
    {
        Tensor cur = _xFixed!;
        bool ownCur = false;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, _condFixed!, t, sp, _sCos!, _sSin!, _tCos!, _tSin!, _mask!);
            if (ownCur) cur.Dispose();
            cur = next; ownCur = true;
        }
        backend.CopyInto(_blockOutFixed!, cur);
        cur.Dispose();
    }

    /// <summary>Caches the geometry-invariant RoPE tables + causal mask per (t,gh,gw); rebuilds only on a
    /// geometry change so the captured graph reads stable device addresses across forwards.</summary>
    private void EnsureGeometry(IBackend backend, int t, int gh, int gw)
    {
        long sig = ((long)t << 40) ^ ((long)gh << 20) ^ gw;
        if (sig == _geomSig && _sCos is not null) return;
        _sCos?.Dispose(); _sSin?.Dispose(); _tCos?.Dispose(); _tSin?.Dispose(); _mask?.Dispose();
        (_sCos, _sSin) = _spatialRope.Build(gh, gw);
        (_tCos, _tSin) = _temporalRope.BuildCosSin(t, 1, 1);
        _mask = BuildCausalMask(t);
        _geomSig = sig;
    }

    private static double ElapsedMs(long ts) => (System.Diagnostics.Stopwatch.GetTimestamp() - ts) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    private static Tensor CloneCpu(Tensor src)
    {
        Tensor dst = new Tensor(src.Shape, DType.F32);
        new ReadOnlySpan<float>((float*)src.DataPointer, (int)src.ElementCount)
            .CopyTo(new Span<float>((float*)dst.DataPointer, (int)dst.ElementCount));
        return dst;
    }

    /// <summary><c>c[t] = mlp(sinusoidal(noiseIdx[t])) + external_cond(action[t])</c> → <c>[T, dim]</c>.</summary>
    private Tensor BuildCondition(IBackend backend, ReadOnlySpan<int> noiseIndices, Tensor actions, int t, int dim)
    {
        Tensor sinEmb = new Tensor(new TensorShape(t, _config.FreqDim), DType.F32);
        for (int f = 0; f < t; f++)
        {
            Tensor row = new Tensor(new TensorShape(1, _config.FreqDim), DType.F32);
            DiTUtils.SinusoidalTimestepEmbedding(row, noiseIndices[f], 1, _config.FreqDim, 10000f);
            Buffer.MemoryCopy((float*)row.DataPointer, (float*)sinEmb.DataPointer + (long)f * _config.FreqDim,
                (long)_config.FreqDim * 4, (long)_config.FreqDim * 4);
            row.Dispose();
        }
        Tensor e1 = new Tensor(new TensorShape(t, dim), DType.F32);
        backend.Linear(e1, sinEmb, _tEmb1W!, _tEmb1B);
        sinEmb.Dispose();
        Tensor act = new Tensor(e1.Shape, DType.F32);
        backend.Silu(act, e1);
        e1.Dispose();
        Tensor cond = new Tensor(new TensorShape(t, dim), DType.F32);
        backend.Linear(cond, act, _tEmb2W!, _tEmb2B);
        act.Dispose();

        Tensor actionEmb = new Tensor(new TensorShape(t, dim), DType.F32);
        backend.Linear(actionEmb, actions, _extCondW!, _extCondB);
        float* cp = (float*)cond.DataPointer;
        float* ap = (float*)actionEmb.DataPointer;
        for (long i = 0; i < (long)t * dim; i++) cp[i] += ap[i];
        actionEmb.Dispose();
        return cond;
    }

    private Tensor Patchify(IBackend backend, Tensor latents, int t, int gh, int gw)
    {
        int p = _config.PatchSize, c = _config.InChannels;
        int h = gh * p, w = gw * p, sp = gh * gw, patchVec = c * p * p;
        Tensor patches = new Tensor(new TensorShape(t * sp, patchVec), DType.F32);
        float* xp = (float*)latents.DataPointer;
        float* pp = (float*)patches.DataPointer;
        for (int f = 0; f < t; f++)
            for (int ty = 0; ty < gh; ty++)
                for (int tx = 0; tx < gw; tx++)
                {
                    long baseIdx = (((long)f * gh + ty) * gw + tx) * patchVec;
                    int d = 0;
                    for (int ci = 0; ci < c; ci++)
                        for (int py = 0; py < p; py++)
                            for (int px = 0; px < p; px++)
                                pp[baseIdx + d++] = xp[(((long)f * c + ci) * h + ty * p + py) * w + tx * p + px];
                }
        Tensor tokens = new Tensor(new TensorShape(t * sp, _config.HiddenSize), DType.F32);
        backend.Linear(tokens, patches, _patchW2d!, _patchB);
        patches.Dispose();
        return tokens;
    }

    private Tensor FinalLayer(IBackend backend, Tensor x, Tensor cond, int t, int sp, int gh, int gw)
    {
        int dim = _config.HiddenSize, p = _config.PatchSize, c = _config.InChannels;
        Tensor silu = new Tensor(new TensorShape(t, dim), DType.F32);
        backend.Silu(silu, cond);
        Tensor mod = new Tensor(new TensorShape(t, 2 * dim), DType.F32);
        backend.Linear(mod, silu, _finalModW!, _finalModB);
        silu.Dispose();

        Tensor normed = new Tensor(new TensorShape(t * sp, dim), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, 1, t * sp, dim, 1e-6f);
        float* np = (float*)normed.DataPointer;
        float* mp = (float*)mod.DataPointer;
        for (int f = 0; f < t; f++)
        {
            float* shift = mp + (long)f * 2 * dim;
            float* scale = shift + dim;
            for (int i = 0; i < sp; i++)
            {
                long off = ((long)f * sp + i) * dim;
                for (int d = 0; d < dim; d++) np[off + d] = np[off + d] * (1f + scale[d]) + shift[d];
            }
        }
        mod.Dispose();

        int outVec = c * p * p;
        Tensor proj = new Tensor(new TensorShape(t * sp, outVec), DType.F32);
        backend.Linear(proj, normed, _finalW!, _finalB);
        normed.Dispose();

        // Unpatchify back to [T, C, gridH·p, gridW·p].
        int h = gh * p, w = gw * p;
        Tensor outT = new Tensor(new TensorShape([(long)t, c, h, w]), DType.F32);
        float* sp2 = (float*)proj.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int f = 0; f < t; f++)
            for (int ty = 0; ty < gh; ty++)
                for (int tx = 0; tx < gw; tx++)
                {
                    long baseIdx = (((long)f * gh + ty) * gw + tx) * outVec;
                    // Reference unpatchify: reshape(...,p,p,c) then einsum nhwpqc->nchpwq — the out-vector is
                    // laid out [py, px, ci] (channel innermost), NOT [ci, py, px]. Read with that stride.
                    for (int ci = 0; ci < c; ci++)
                        for (int py = 0; py < p; py++)
                            for (int px = 0; px < p; px++)
                                op[(((long)f * c + ci) * h + ty * p + py) * w + tx * p + px]
                                    = sp2[baseIdx + (py * p + px) * c + ci];
                }
        proj.Dispose();
        return outT;
    }

    private static Tensor BuildCausalMask(int t)
    {
        Tensor mask = new Tensor(new TensorShape([1L, 1, t, t]), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int i = 0; i < t; i++)
            for (int j = 0; j < t; j++)
                mp[(long)i * t + j] = j <= i ? 0f : -1e9f;
        return mask;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _tEmb1W = _tEmb1B = _tEmb2W = _tEmb2B = null;
            _extCondW = _extCondB = _finalModW = _finalModB = _finalW = _finalB = null;
            // Null (do not device-dispose) the graph fixed buffers + geometry caches — the context may already be
            // torn down; the captured graph itself is owned by the backend's single slot and reset on model switch.
            _sCos = _sSin = _tCos = _tSin = _mask = null;
            _xFixed = _condFixed = _blockOutFixed = null;
        }
        GC.SuppressFinalize(this);
    }
}
