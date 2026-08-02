using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>SeedVR2 NaDiT — one-step video-restoration transformer. Consumes a channels-LAST 33-channel
/// latent grid (16 noisy + 16 conditioning + 1 mask), patchifies 1×2×2 in <c>(t h w c)</c> order, runs 32
/// windowed MM-DiT blocks (alternating regular/shifted partitions), and emits the 16-channel v-prediction.
/// Tail modulation implements the verified cache-collision semantics: the ATTN slice of the shared emb
/// combines with <c>vid_out_ada.out_shift/out_scale</c> (see SEEDVR2_ARCHITECTURE.md §2.5 — porting the
/// code as written, without the reference's cache hit, is dimensionally impossible AND wrong).
/// Single-clip (B=1). Tokens are device-resident across all blocks (<see cref="SeedVr2DitBlock"/>); the
/// only host work per forward is patchify/unpatchify and the scalar time-embed. Geometry constants live in
/// a cached <see cref="SeedVr2DevicePlan"/> — chunks of one clip share it.</summary>
public sealed class SeedVr2Dit : IDisposable
{
    private readonly SeedVr2Config _config;
    private readonly SeedVr2DitBlock[] _blocks;
    private Tensor _vidInW = null!, _vidInB = null!, _txtInW = null!, _txtInB = null!;
    private Tensor _embInW = null!, _embInB = null!, _embHidW = null!, _embHidB = null!;
    private Tensor _embOutW = null!, _embOutB = null!;
    private Tensor _vidOutW = null!, _vidOutB = null!;
    private float[]? _vidOutNormW, _outShift, _outScale;

    private IBackend? _planBackend;
    private SeedVr2DevicePlan? _plan;
    private Tensor? _ones;
    private Tensor? _tailNormW, _tailScale, _tailShift;
    private float[]? _emb;          // cached time embedding (timestep is constant across a clip's chunks)
    private float _embTimestep = float.NaN;
    private bool _disposed;

    /// <summary>Per-block tap for parity bisection: (blockIndex, vidTokens, txtTokens); −1 = post-embed.</summary>
    public Action<int, Tensor, Tensor>? OnBlockOutput { get; set; }

    /// <summary>Creates the graph for <paramref name="config"/>; weights via <see cref="LoadWeights"/>.</summary>
    public SeedVr2Dit(SeedVr2Config config)
    {
        _config = config;
        _blocks = new SeedVr2DitBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new SeedVr2DitBlock(config, i);
    }

    /// <summary>Config this model was built for.</summary>
    public SeedVr2Config Config => _config;

    /// <summary>Binds converter output (verbatim reference names, RoPE buffers already dropped).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _vidInW = weights["vid_in.proj.weight"];
        _vidInB = weights["vid_in.proj.bias"];
        _txtInW = weights["txt_in.weight"];
        _txtInB = weights["txt_in.bias"];
        _embInW = weights["emb_in.proj_in.weight"];
        _embInB = weights["emb_in.proj_in.bias"];
        _embHidW = weights["emb_in.proj_hid.weight"];
        _embHidB = weights["emb_in.proj_hid.bias"];
        _embOutW = weights["emb_in.proj_out.weight"];
        _embOutB = weights["emb_in.proj_out.bias"];
        if (_config.HasTailNorm)
        {
            _vidOutNormW = ToHostF32(weights["vid_out_norm.weight"]);
            _outShift = ToHostF32(weights["vid_out_ada.out_shift"]);
            _outScale = ToHostF32(weights["vid_out_ada.out_scale"]);
        }
        _vidOutW = weights["vid_out.proj.weight"];
        _vidOutB = weights["vid_out.proj.bias"];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, i);
    }

    /// <summary>Runs the transformer. <paramref name="latent"/> is channels-last <c>[T,H,W,33]</c> (grid in
    /// LATENT cells, pre-patchify); <paramref name="txtEmbed"/> is the frozen positive embedding
    /// <c>[Lt,5120]</c>; <paramref name="timestep"/> in [0,1000]. Returns <c>[T,H,W,16]</c> channels-last.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor txtEmbed, float timestep)
    {
        if (latent.Shape.Rank != 4 || latent.Shape[3] != _config.InChannels)
            throw new HartsyInferenceException($"Expected [T,H,W,{_config.InChannels}] latent, got {latent.Shape}.");
        int t = (int)latent.Shape[0], h = (int)latent.Shape[1], w = (int)latent.Shape[2];
        (int pT, int pH, int pW) = _config.PatchSize;
        if (h % pH != 0 || w % pW != 0)
            throw new HartsyInferenceException($"Latent grid {h}x{w} not divisible by patch {pH}x{pW}.");
        int gridT = t / pT, gridH = h / pH, gridW = w / pW;
        int lv = gridT * gridH * gridW;
        int lt = (int)txtEmbed.Shape[0];

        EnsurePlan(backend, gridT, gridH, gridW, lt);
        float[] emb = EnsureEmb(backend, timestep);

        // Patchify (t h w c) then vid_in projection.
        Tensor patches = Patchify(latent, t, h, w, gridH, gridW);
        Tensor vid = new Tensor(new TensorShape(lv, _config.VidDim), DType.F32);
        backend.Linear(vid, patches, _vidInW, _vidInB);
        patches.Dispose();

        // txt_in projection of the frozen embedding.
        Tensor txt = new Tensor(new TensorShape(lt, _config.VidDim), DType.F32);
        backend.Linear(txt, txtEmbed, _txtInW, _txtInB);

        OnBlockOutput?.Invoke(-1, vid, txt);

        for (int i = 0; i < _blocks.Length; i++)
        {
            (vid, txt) = _blocks[i].Forward(backend, vid, txt, _plan!, _ones!);
            OnBlockOutput?.Invoke(i, vid, txt);
        }

        // Tail: affine RMS norm + attn-slice modulation (cache-collision semantics) — 3B only; the 7B has a
        // bare vid_out (no vid_out_norm in its config or checkpoint).
        if (_config.HasTailNorm)
        {
            Tensor normed = new Tensor(vid.Shape, DType.F32);
            backend.RmsNorm(normed, vid, _tailNormW!, 1e-5f);
            Tensor modded = new Tensor(vid.Shape, DType.F32);
            backend.AffineBroadcastLastDim(modded, normed, _tailScale!, _tailShift!);
            normed.Dispose();
            vid.Dispose();
            vid = modded;
        }
        Tensor projected = new Tensor(new TensorShape(lv, _config.OutChannels * pT * pH * pW), DType.F32);
        backend.Linear(projected, vid, _vidOutW, _vidOutB);
        backend.Sync();
        vid.Dispose();
        txt.Dispose();

        Tensor output = Unpatchify(projected, t, h, w, gridH, gridW);
        projected.Dispose();
        return output;
    }

    private void EnsurePlan(IBackend backend, int gridT, int gridH, int gridW, int lt)
    {
        if (_plan is not null && _plan.Key == (gridT, gridH, gridW, lt) && ReferenceEquals(_planBackend, backend))
            return;
        _plan?.Release(_planBackend ?? backend);
        _plan = new SeedVr2DevicePlan(backend, _config, gridT, gridH, gridW, lt);
        _planBackend = backend;
        if (_ones is null)
        {
            _ones = new Tensor(new TensorShape(_config.VidDim), DType.F32);
            _ones.AsSpan<float>().Fill(1f);
            backend.PreloadWeights([_ones]);
        }
    }

    private float[] EnsureEmb(IBackend backend, float timestep)
    {
        if (_emb is not null && _embTimestep == timestep)
        {
            foreach (SeedVr2DitBlock block in _blocks)
                block.PrepareMod(backend, _emb);
            return _emb;
        }
        _emb = TimeEmbed(backend, timestep);
        _embTimestep = timestep;
        foreach (SeedVr2DitBlock block in _blocks)
            block.PrepareMod(backend, _emb);
        if (_config.HasTailNorm)
        {
            int c = _config.VidDim;
            ReleaseTail(backend);
            _tailNormW = FromHost(_vidOutNormW!);
            Tensor scale = new Tensor(new TensorShape(c), DType.F32);
            Tensor shift = new Tensor(new TensorShape(c), DType.F32);
            Span<float> sc = scale.AsSpan<float>(), sh = shift.AsSpan<float>();
            for (int i = 0; i < c; i++)
            {
                sc[i] = _emb[i * 6 + 1] + _outScale![i];
                sh[i] = _emb[i * 6 + 0] + _outShift![i];
            }
            _tailScale = scale;
            _tailShift = shift;
            backend.PreloadWeights([_tailNormW, _tailScale, _tailShift]);
        }
        return _emb;
    }

    private static Tensor FromHost(float[] values)
    {
        Tensor t = new Tensor(new TensorShape(values.Length), DType.F32);
        values.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }

    private void ReleaseTail(IBackend backend)
    {
        if (_tailNormW is null)
            return;
        backend.FreeWeights([_tailNormW, _tailScale!, _tailShift!]);
        _tailNormW.Dispose();
        _tailScale!.Dispose();
        _tailShift!.Dispose();
        _tailNormW = _tailScale = _tailShift = null;
    }

    /// <summary>Frees the geometry plan, modulation constants, and other DiT-owned device tensors.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        IBackend? backend = _planBackend;
        if (backend is not null)
        {
            _plan?.Release(backend);
            foreach (SeedVr2DitBlock block in _blocks)
                block.ReleaseOwned(backend);
            ReleaseTail(backend);
            if (_ones is not null)
                backend.FreeWeights([_ones]);
        }
        else
        {
            _plan?.Dispose();
        }
        _plan = null;
        _ones?.Dispose();
        _ones = null;
    }

    // Host-math vectors must be F32 regardless of checkpoint dtype (the 7B ships F16-converted for disk).
    private static float[] ToHostF32(Tensor t)
    {
        if (t.DType == Core.Tensors.DType.F32)
            return t.AsSpan<float>().ToArray();
        Tensor cast = t.CastTo(Core.Tensors.DType.F32);
        float[] result = cast.AsSpan<float>().ToArray();
        cast.Dispose();
        return result;
    }

    /// <summary>GEMM weights for backend preload/free at pipeline phase transitions.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _vidInW;
        yield return _vidInB;
        yield return _txtInW;
        yield return _txtInB;
        yield return _embInW;
        yield return _embInB;
        yield return _embHidW;
        yield return _embHidB;
        yield return _embOutW;
        yield return _embOutB;
        yield return _vidOutW;
        yield return _vidOutB;
        foreach (SeedVr2DitBlock block in _blocks)
            foreach (Tensor t in block.EnumerateWeights())
                yield return t;
    }

    private Tensor Patchify(Tensor latent, int t, int h, int w, int gridH, int gridW)
    {
        (int pT, int pH, int pW) = _config.PatchSize;
        int cIn = _config.InChannels;
        int tokenDim = cIn * pT * pH * pW;
        Tensor patches = new Tensor(new TensorShape((long)t * gridH * gridW, tokenDim), DType.F32);
        ReadOnlySpan<float> src = latent.AsSpan<float>();
        Span<float> dst = patches.AsSpan<float>();
        for (int ti = 0; ti < t; ti++)
        for (int gh = 0; gh < gridH; gh++)
        for (int gw = 0; gw < gridW; gw++)
        {
            int token = (ti * gridH + gh) * gridW + gw;
            int col = 0;
            for (int hh = 0; hh < pH; hh++)
            for (int ww = 0; ww < pW; ww++)
            {
                int srcIdx = ((ti * h + gh * pH + hh) * w + gw * pW + ww) * cIn;
                src.Slice(srcIdx, cIn).CopyTo(dst.Slice(token * tokenDim + col, cIn));
                col += cIn;
            }
        }
        return patches;
    }

    private Tensor Unpatchify(Tensor projected, int t, int h, int w, int gridH, int gridW)
    {
        (int pT, int pH, int pW) = _config.PatchSize;
        int cOut = _config.OutChannels;
        Tensor output = new Tensor(new TensorShape((long)t, h, w, cOut), DType.F32);
        ReadOnlySpan<float> src = projected.AsSpan<float>();
        Span<float> dst = output.AsSpan<float>();
        int tokenDim = cOut * pT * pH * pW;
        for (int ti = 0; ti < t; ti++)
        for (int gh = 0; gh < gridH; gh++)
        for (int gw = 0; gw < gridW; gw++)
        {
            int token = (ti * gridH + gh) * gridW + gw;
            int col = 0;
            for (int hh = 0; hh < pH; hh++)
            for (int ww = 0; ww < pW; ww++)
            {
                int dstIdx = ((ti * h + gh * pH + hh) * w + gw * pW + ww) * cOut;
                src.Slice(token * tokenDim + col, cOut).CopyTo(dst.Slice(dstIdx, cOut));
                col += cOut;
            }
        }
        return output;
    }

    private float[] TimeEmbed(IBackend backend, float timestep)
    {
        // diffusers get_timestep_embedding: half=128, exponent = −ln(10000)·i/half, sin block then cos.
        const int sinusoidalDim = 256;
        int half = sinusoidalDim / 2;
        Tensor sinusoid = new Tensor(new TensorShape(1, sinusoidalDim), DType.F32);
        Span<float> s = sinusoid.AsSpan<float>();
        for (int i = 0; i < half; i++)
        {
            double angle = timestep * Math.Exp(-Math.Log(10000.0) * i / half);
            s[i] = (float)Math.Sin(angle);
            s[half + i] = (float)Math.Cos(angle);
        }
        Tensor h1 = new Tensor(new TensorShape(1, _config.VidDim), DType.F32);
        backend.Linear(h1, sinusoid, _embInW, _embInB);
        backend.Sync();
        Silu(h1);
        Tensor h2 = new Tensor(new TensorShape(1, _config.VidDim), DType.F32);
        backend.Linear(h2, h1, _embHidW, _embHidB);
        backend.Sync();
        Silu(h2);
        Tensor outT = new Tensor(new TensorShape(1, _config.EmbDim), DType.F32);
        backend.Linear(outT, h2, _embOutW, _embOutB);
        backend.Sync();
        float[] emb = outT.AsSpan<float>().ToArray();
        sinusoid.Dispose(); h1.Dispose(); h2.Dispose(); outT.Dispose();
        return emb;
    }

    private static void Silu(Tensor t)
    {
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++)
            s[i] = s[i] / (1f + MathF.Exp(-s[i]));
    }
}
