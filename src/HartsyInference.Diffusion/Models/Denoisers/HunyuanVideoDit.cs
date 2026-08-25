using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>HunyuanVideo MM-DiT — the base offline-video transformer and the backbone the Hunyuan-GameCraft world model finetunes. Reuses <see cref="HunyuanImageBlock"/> (dual-stream) and <see cref="HunyuanImageSingleBlock"/> (single-stream) with a <b>3-axis (T,H,W) RoPE</b> via the generalized <see cref="HunyuanImageRope"/>. Predicts rectified-flow velocity over the 16-channel latent from a patchified input (16-ch plain, or GameCraft's 33-ch noisy+history+mask), Llava text tokens refined by the <see cref="HunyuanVideoTokenRefiner"/>, a pooled CLIP vector, a timestep, and — for plain HunyuanVideo (<c>guidance_embeds=True</c>) — an embedded guidance scalar. Optional <c>cameraTokens</c> are token-added to the image stream (GameCraft CameraNet). <para>Faithful to diffusers <c>transformer_hunyuan_video.py</c>: <c>temb = timestep_emb + guidance_emb + pooled_vec_emb</c>, the 2-layer token refiner replaces the plain <c>txt_in</c> projection, and the final AdaLN-continuous chunks <c>[shift, scale]</c> (Tencent order). The 60 double+single blocks are streamable via <see cref="GetBlock"/>/<see cref="BeforeBlockForward"/> for the 24 GB-class bf16 checkpoint.</para></summary>
public sealed unsafe class HunyuanVideoDit : IDisposable, IStreamableDenoiser
{
    private readonly HunyuanVideoConfig _cfg;
    private readonly HunyuanImageBlock[] _double;
    private readonly HunyuanImageSingleBlock[] _single;
    private readonly HunyuanImageRope _rope;
    private readonly HunyuanVideoTokenRefiner _refiner;

    private Tensor? _imgInW, _imgInB;       // [hidden, inCh*pT*pH*pW]
    private Tensor? _timeW0, _timeB0, _timeW1, _timeB1;
    private Tensor? _vecW0, _vecB0, _vecW1, _vecB1;
    private Tensor? _guidW0, _guidB0, _guidW1, _guidB1;   // guidance_in (plain HunyuanVideo only)
    private Tensor? _finalModW, _finalModB; // [2*hidden, hidden]
    private Tensor? _outW, _outB;           // [outCh*pT*pH*pW, hidden]

    // ── Step-graph capture (HARTSY_DIT_GRAPH, opt-in) ─────────────────────────────────────────────────────────
    // HunyuanVideo is single-forward (embedded guidance, no CFG) → the Krea2 single-forward template. The captured
    // body is img_in → double+single blocks → final AdaLN → proj_out, reading ONLY the persistent fixed buffers
    // below; the host boundary work (Patchify, the token refiner, BuildTemb, Unpatchify, and the pipeline's host
    // Euler) runs OUTSIDE the capture and refreshes the fixed input buffers via CopyInto — so no space-to-depth
    // kernel is needed. Restricted to b==1 (the block RoPE host fallback is batch>1-only) with no camera tokens.
    // On the graph path the pipeline must NOT run its per-step FreeActivations (it would free the addresses the
    // capture bakes; the graph balances its own allocations) — see StepGraphActive.
    private Tensor? _patchesFixed;    // [1, sImg, patchVec]  img_in input — refreshed each step (Patchify → CopyInto)
    private Tensor? _txtTokFixed;     // [1, sTxt, hidden]    refined text — refreshed each step (refiner → CopyInto)
    private Tensor? _tembFixed;       // [1, hidden]          time+guidance+vector embedding — refreshed each step
    private Tensor? _graphProjected;  // [1, sImg, outVec]    captured proj_out landing buffer (read back to Unpatchify)
    private long _graphSig = long.MinValue;
    private int _graphSigCalls, _graphSigFlips;
    private bool _graphDead;
    private const int GraphCaptureCall = 3;

    /// <summary>True while the step graph is engaged for this session (opt-in flag on, the graph path has run at least once, not self-disabled). The pipeline suppresses its per-step <c>FreeActivations</c> while this holds: the captured graph balances its own allocations, and freeing activations would release the fixed buffers the capture bakes. Re-check AFTER each <see cref="Forward"/> so a mid-gen self-disable re-enables the sweep.</summary>
    public bool StepGraphActive => DitStepGraph.Enabled && !_graphDead && _graphSig != long.MinValue;

    /// <summary>Invalidates the captured step graph — MUST be called whenever the DiT weights are freed (the pipeline's post-gen <c>FreeWeights</c>): the captured graph bakes the weight device pointers, so a free + next-gen re-upload would leave it replaying against freed memory (CUDA 700). The next generation re-warms and re-captures.</summary>
    public void InvalidateStepGraph(IBackend backend)
    {
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, this)) backend.StepGraphOwner = null;
        _graphSig = long.MinValue;   // MinValue = "no sig": the next call resets WITHOUT counting a flip
        _graphSigCalls = 0;
    }

    public HunyuanVideoConfig Config => _cfg;

    /// <summary>Optional hook invoked immediately before each transformer block's forward pass (global index over double-then-single blocks). Pipelines plug a <c>BlockStreamingController</c> here to drive prefetch/eviction so the 24 GB bf16 DiT fits in 24 GB. Null = all resident.</summary>
    public Action<int>? BeforeBlockForward { get; set; }

    public HunyuanVideoDit(HunyuanVideoConfig cfg)
    {
        _cfg = cfg;
        _double = new HunyuanImageBlock[cfg.NumDoubleBlocks];
        for (int i = 0; i < cfg.NumDoubleBlocks; i++)
            _double[i] = new HunyuanImageBlock(cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim, cfg.MlpDim);
        _single = new HunyuanImageSingleBlock[cfg.NumSingleBlocks];
        for (int i = 0; i < cfg.NumSingleBlocks; i++)
            _single[i] = new HunyuanImageSingleBlock(cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim, cfg.MlpDim);
        _rope = new HunyuanImageRope(cfg.RopeAxesDim, cfg.RopeTheta);
        _refiner = new HunyuanVideoTokenRefiner(cfg.TextEmbedDim, cfg.HiddenSize, cfg.NumHeads, numBlocks: 2);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _imgInW = TensorCasts.EnsureF32(w[$"{p}img_in.weight"]); _imgInB = TensorCasts.EnsureF32(w[$"{p}img_in.bias"]);
        _timeW0 = TensorCasts.EnsureF32(w[$"{p}time_in.0.weight"]); _timeB0 = TensorCasts.EnsureF32(w[$"{p}time_in.0.bias"]);
        _timeW1 = TensorCasts.EnsureF32(w[$"{p}time_in.2.weight"]); _timeB1 = TensorCasts.EnsureF32(w[$"{p}time_in.2.bias"]);
        _vecW0 = TensorCasts.EnsureF32(w[$"{p}vector_in.0.weight"]); _vecB0 = TensorCasts.EnsureF32(w[$"{p}vector_in.0.bias"]);
        _vecW1 = TensorCasts.EnsureF32(w[$"{p}vector_in.2.weight"]); _vecB1 = TensorCasts.EnsureF32(w[$"{p}vector_in.2.bias"]);
        if (_cfg.GuidanceEmbed)
        {
            _guidW0 = TensorCasts.EnsureF32(w[$"{p}guidance_in.0.weight"]); _guidB0 = TensorCasts.EnsureF32(w[$"{p}guidance_in.0.bias"]);
            _guidW1 = TensorCasts.EnsureF32(w[$"{p}guidance_in.2.weight"]); _guidB1 = TensorCasts.EnsureF32(w[$"{p}guidance_in.2.bias"]);
        }
        _refiner.LoadWeights(w, $"{p}txt_in");
        for (int i = 0; i < _double.Length; i++) _double[i].LoadWeights(w, $"{p}double_blocks.{i}");
        for (int i = 0; i < _single.Length; i++) _single[i].LoadWeights(w, $"{p}single_blocks.{i}");
        _finalModW = TensorCasts.EnsureF32(w[$"{p}final_layer.mod.weight"]); _finalModB = TensorCasts.EnsureF32(w[$"{p}final_layer.mod.bias"]);
        _outW = TensorCasts.EnsureF32(w[$"{p}final_layer.proj.weight"]); _outB = TensorCasts.EnsureF32(w[$"{p}final_layer.proj.bias"]);
    }

    /// <summary>Always-resident weights: patchify/text-refiner/time/vector/guidance embedders + final layer. Touched every step regardless of the executing block, so the streaming controller doesn't manage them.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        Tensor?[] head = [_imgInW, _imgInB, _timeW0, _timeB0, _timeW1, _timeB1,
            _vecW0, _vecB0, _vecW1, _vecB1, _guidW0, _guidB0, _guidW1, _guidB1, _finalModW, _finalModB, _outW, _outB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (Tensor t in _refiner.EnumerateWeights()) yield return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in EnumerateSharedWeights()) yield return t;
        foreach (HunyuanImageBlock b in _double) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (HunyuanImageSingleBlock b in _single) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Number of streamable transformer blocks (double + single, in that order).</summary>
    public int BlockCount => _double.Length + _single.Length;

    /// <summary>Streamable block at global index <paramref name="idx"/> (double blocks first, then single).</summary>
    public IStreamingBlock GetBlock(int idx) => idx < _double.Length ? _double[idx] : _single[idx - _double.Length];

    /// <summary>Predicts velocity <c>[B, OutChannels, T, H, W]</c> for the latent <c>[B, InChannels, T, H, W]</c>, conditioned on the raw Llava text hidden states <c>[B, L, TextEmbedDim]</c> (refined internally), a pooled CLIP vector <c>[B, PooledEmbedDim]</c>, <paramref name="timestep"/> (≈0..1000), and — for plain HunyuanVideo — <paramref name="guidance"/> (the embedded-guidance scalar, typically <c>EmbeddedGuidanceScale·1000</c>). <paramref name="cameraTokens"/> (<c>[B, S_img, HiddenSize]</c>), when supplied, is added to the image tokens after patchify (GameCraft CameraNet fusion).</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor txt, Tensor pooled, float timestep,
        float guidance = 0f, Tensor? cameraTokens = null)
    {
        int b = (int)latent.Shape[0];
        int hidden = _cfg.HiddenSize;
        (int pT, int pH, int pW) = _cfg.PatchSize;
        int T = (int)latent.Shape[2], H = (int)latent.Shape[3], W = (int)latent.Shape[4];
        int tOut = T / pT, hOut = H / pH, wOut = W / pW;
        int sImg = tOut * hOut * wOut;
        int patchVec = _cfg.InChannels * pT * pH * pW;

        // Step-graph fast path (opt-in): capture img_in → blocks → final → proj_out once and replay it. Only the
        // single-forward b==1 path with no camera tokens qualifies (the block RoPE host fallback is batch>1-only).
        // Requires the DiT to be RESIDENT (BeforeBlockForward null): a captured graph bakes weight device pointers,
        // so block streaming — which re-points weights per block and runs a host prefetch hook — is incompatible.
        // Self-disables to eager.
        if (DitStepGraph.Enabled && backend.StepGraphSupported && !_graphDead && b == 1 && cameraTokens is null
            && BeforeBlockForward is null)
        {
            return ForwardGraph(backend, latent, txt, pooled, timestep, guidance,
                hidden, pT, pH, pW, T, H, W, tOut, hOut, wOut, sImg, patchVec);
        }

        // 1. Patchify → img tokens.
        Tensor patches = new(new TensorShape(b, sImg, patchVec), DType.F32);
        Patchify(latent, patches, b, _cfg.InChannels, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        Tensor img = new(new TensorShape(b, sImg, hidden), DType.F32);
        backend.Linear(img, patches, _imgInW!, _imgInB!); patches.Dispose();
        if (cameraTokens is not null) { Tensor sum = new(img.Shape, DType.F32); backend.Add(sum, img, cameraTokens); img.Dispose(); img = sum; }

        // 2. Text tokens via the 2-layer token refiner.
        Tensor txtTok = _refiner.Forward(backend, txt, timestep);

        // 3. temb = time(timestep) + guidance(guidance) + vector(pooled).
        Tensor temb = BuildTemb(backend, b, hidden, timestep, guidance, pooled);

        // 4. Blocks (3D rope, packed (tOut,hOut,wOut)).
        for (int i = 0; i < _double.Length; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (Tensor ni, Tensor nt) = _double[i].Forward(backend, img, txtTok, temb, _rope, hOut, wOut, tOut);
            img.Dispose(); txtTok.Dispose(); img = ni; txtTok = nt;
        }
        for (int i = 0; i < _single.Length; i++)
        {
            BeforeBlockForward?.Invoke(_double.Length + i);
            (Tensor ni, Tensor nt) = _single[i].Forward(backend, img, txtTok, temb, _rope, hOut, wOut, tOut);
            img.Dispose(); txtTok.Dispose(); img = ni; txtTok = nt;
        }
        txtTok.Dispose();

        // 5. Final AdaLN-continuous → proj → unpatchify.
        Tensor mod = new(new TensorShape(b, 2 * hidden), DType.F32);
        Tensor tAct = new(temb.Shape, DType.F32); backend.Silu(tAct, temb); temb.Dispose();
        backend.Linear(mod, tAct, _finalModW!, _finalModB!); tAct.Dispose();
        Tensor normed = new(img.Shape, DType.F32);
        if (b == 1)
        {
            // Device-resident final layer (B=1): the host path below drains the FULL last hidden state D2H,
            // modulates on the CPU, and re-uploads it for proj — one full-hidden round-trip per forward.
            // mod is packed [shift, scale] (Tencent adaLN_modulation order) — sliced in that order.
            backend.LayerNormNoAffine(normed, img, 1e-6f); img.Dispose();
            Tensor fShift = new(new TensorShape(hidden), DType.F32);
            backend.SliceRows(fShift, mod, 0);
            Tensor fScale = new(new TensorShape(hidden), DType.F32);
            backend.SliceRows(fScale, mod, 1);
            mod.Dispose();
            Tensor fScale1 = new(new TensorShape(hidden), DType.F32);
            backend.AddScalar(fScale1, fScale, 1f); fScale.Dispose();
            Tensor modded = new(normed.Shape, DType.F32);
            backend.AffineBroadcastLastDim(modded, normed, fScale1, fShift);
            normed.Dispose(); fScale1.Dispose(); fShift.Dispose();
            normed = modded;
        }
        else
        {
            DiTUtils.LayerNormNoAffine(normed, img, b, sImg, hidden); img.Dispose();
            Modulate(normed, mod, b, sImg, hidden); mod.Dispose();
        }

        int outVec = _cfg.OutChannels * pT * pH * pW;
        Tensor projected = new(new TensorShape(b, sImg, outVec), DType.F32);
        backend.Linear(projected, normed, _outW!, _outB!); normed.Dispose();

        Tensor velocity = new(new TensorShape([(long)b, _cfg.OutChannels, T, H, W]), DType.F32);
        Unpatchify(projected, velocity, b, _cfg.OutChannels, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        projected.Dispose();
        return velocity;
    }

    /// <summary>Step-graph forward for the single-forward b==1 path. Refreshes the fixed input buffers OUTSIDE the capture (host Patchify / token refiner / BuildTemb → CopyInto), then captures (call <see cref="GraphCaptureCall"/>) or replays (later calls) the img_in → blocks → final → proj_out body, and unpatchifies the fixed proj_out buffer into a fresh caller-owned velocity. Warm calls 1..2 run the body eagerly through the fixed buffers so nothing inside the capture does a first-touch allocation (RoPE tables, weight promotions). Self-disables to the eager body on any capture failure so the generation stays correct.</summary>
    private Tensor ForwardGraph(IBackend backend, Tensor latent, Tensor txt, Tensor pooled, float timestep,
        float guidance, int hidden, int pT, int pH, int pW, int T, int H, int W, int tOut, int hOut, int wOut,
        int sImg, int patchVec)
    {
        int outVec = _cfg.OutChannels * pT * pH * pW;

        // ── Refresh the fixed inputs OUTSIDE the capture (all the per-step host excursions live here) ──
        Tensor patches = new(new TensorShape(1, sImg, patchVec), DType.F32);
        Patchify(latent, patches, 1, _cfg.InChannels, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        _patchesFixed ??= new Tensor(new TensorShape(1, sImg, patchVec), DType.F32);
        backend.CopyInto(_patchesFixed, patches); patches.Dispose();

        Tensor txtTok = _refiner.Forward(backend, txt, timestep);
        _txtTokFixed ??= new Tensor(txtTok.Shape, DType.F32);
        backend.CopyInto(_txtTokFixed, txtTok); txtTok.Dispose();

        Tensor temb = BuildTemb(backend, 1, hidden, timestep, guidance, pooled);
        _tembFixed ??= new Tensor(temb.Shape, DType.F32);
        backend.CopyInto(_tembFixed, temb); temb.Dispose();

        _graphProjected ??= new Tensor(new TensorShape(1, sImg, outVec), DType.F32);

        // ── Signature / owner management (mirror Krea2ForwardPatched). The capture depends only on the token
        // geometry; the shared backend graph slot may have been captured by another resident model. ──
        long sig = ((long)tOut * 73856093L) ^ ((long)hOut * 19349663L) ^ ((long)wOut * 83492791L)
            ^ ((long)_txtTokFixed!.Shape[1] * 2654435761L);
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning("[HunyuanVideo graph] signature flip storm — step-graph disabled for this session.");
                RunGraphBody(backend, hidden, hOut, wOut, tOut, sImg, outVec);
                return UnpatchifyToVelocity(backend, T, H, W, pT, pH, pW, tOut, hOut, wOut);
            }
            _graphSig = sig;
            _graphSigCalls = 0;
            backend.StepGraphOwner = this;
        }
        _graphSigCalls++;

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return UnpatchifyToVelocity(backend, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        }

        // Calls 1..GraphCaptureCall-1 run the body eagerly THROUGH the fixed buffers (warm caches/promotions);
        // call GraphCaptureCall records the same op sequence.
        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture) backend.StepGraphBegin();
        try
        {
            RunGraphBody(backend, hidden, hOut, wOut, tOut, sImg, outVec);
        }
        catch (Exception ex) when (capture)
        {
            backend.StepGraphReset();
            _graphDead = true;
            HartsyInference.Core.Logging.Logs.Warning($"[HunyuanVideo graph] capture invalidated — falling back to eager: {ex.Message}");
            RunGraphBody(backend, hidden, hOut, wOut, tOut, sImg, outVec);
            return UnpatchifyToVelocity(backend, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();
                HartsyInference.Core.Logging.Logs.Info("[HunyuanVideo graph] denoise step captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                backend.StepGraphReset();
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[HunyuanVideo graph] capture failed — falling back to eager: {ex.Message}");
                RunGraphBody(backend, hidden, hOut, wOut, tOut, sImg, outVec);
            }
        }
        return UnpatchifyToVelocity(backend, T, H, W, pT, pH, pW, tOut, hOut, wOut);
    }

    /// <summary>The captured (or capture-warming) body: img_in on the fixed patchified latent → double+single blocks (a disposable working copy of the fixed text so the block loop can free it) → device final AdaLN → proj_out, landing in the fixed <see cref="_graphProjected"/> buffer. All device ops — no host reads (b==1 RoPE is the GPU path, the diagnostic block hooks are gated off on this path).</summary>
    private void RunGraphBody(IBackend backend, int hidden, int hOut, int wOut, int tOut, int sImg, int outVec)
    {
        Tensor img = new(new TensorShape(1, sImg, hidden), DType.F32);
        backend.Linear(img, _patchesFixed!, _imgInW!, _imgInB!);
        Tensor txtTok = new(_txtTokFixed!.Shape, DType.F32);
        backend.CopyInto(txtTok, _txtTokFixed!);

        for (int i = 0; i < _double.Length; i++)
        {
            (Tensor ni, Tensor nt) = _double[i].Forward(backend, img, txtTok, _tembFixed!, _rope, hOut, wOut, tOut);
            img.Dispose(); txtTok.Dispose(); img = ni; txtTok = nt;
        }
        for (int i = 0; i < _single.Length; i++)
        {
            (Tensor ni, Tensor nt) = _single[i].Forward(backend, img, txtTok, _tembFixed!, _rope, hOut, wOut, tOut);
            img.Dispose(); txtTok.Dispose(); img = ni; txtTok = nt;
        }
        txtTok.Dispose();

        // Final AdaLN-continuous (device B==1 path, mod packed [shift, scale]) → proj_out → fixed buffer.
        Tensor mod = new(new TensorShape(1, 2 * hidden), DType.F32);
        Tensor tAct = new(_tembFixed!.Shape, DType.F32); backend.Silu(tAct, _tembFixed!);
        backend.Linear(mod, tAct, _finalModW!, _finalModB!); tAct.Dispose();
        Tensor normed = new(img.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, img, 1e-6f); img.Dispose();
        Tensor fShift = new(new TensorShape(hidden), DType.F32); backend.SliceRows(fShift, mod, 0);
        Tensor fScale = new(new TensorShape(hidden), DType.F32); backend.SliceRows(fScale, mod, 1); mod.Dispose();
        Tensor fScale1 = new(new TensorShape(hidden), DType.F32); backend.AddScalar(fScale1, fScale, 1f); fScale.Dispose();
        Tensor modded = new(normed.Shape, DType.F32);
        backend.AffineBroadcastLastDim(modded, normed, fScale1, fShift);
        normed.Dispose(); fScale1.Dispose(); fShift.Dispose();
        Tensor projected = new(new TensorShape(1, sImg, outVec), DType.F32);
        backend.Linear(projected, modded, _outW!, _outB!); modded.Dispose();
        backend.CopyInto(_graphProjected!, projected); projected.Dispose();
    }

    /// <summary>Device-copies the fixed proj_out buffer into a fresh disposable tensor and unpatchifies THAT into a fresh, caller-owned velocity <c>[1, OutChannels, T, H, W]</c>. Reading <see cref="_graphProjected"/>'s DataPointer directly would D2H-AND-FREE the graph-owned buffer the captured graph writes (the Krea2 <c>SnapshotGraphLatent</c> rule) — freeing it corrupts every subsequent replay's output.</summary>
    private Tensor UnpatchifyToVelocity(IBackend backend, int T, int H, int W, int pT, int pH, int pW,
        int tOut, int hOut, int wOut)
    {
        Tensor snap = new(_graphProjected!.Shape, DType.F32);
        backend.CopyInto(snap, _graphProjected!);
        Tensor velocity = new(new TensorShape([1L, _cfg.OutChannels, T, H, W]), DType.F32);
        Unpatchify(snap, velocity, 1, _cfg.OutChannels, T, H, W, pT, pH, pW, tOut, hOut, wOut);
        snap.Dispose();
        return velocity;
    }

    private Tensor BuildTemb(IBackend backend, int b, int hidden, float timestep, float guidance, Tensor pooled)
    {
        Tensor tTime = TimestepMlp(backend, b, hidden, timestep, _timeW0!, _timeB0!, _timeW1!, _timeB1!);

        Tensor v0 = new(new TensorShape(b, hidden), DType.F32); backend.Linear(v0, pooled, _vecW0!, _vecB0!);
        Tensor v0a = new(v0.Shape, DType.F32); backend.Silu(v0a, v0); v0.Dispose();
        Tensor tVec = new(new TensorShape(b, hidden), DType.F32); backend.Linear(tVec, v0a, _vecW1!, _vecB1!); v0a.Dispose();

        Tensor temb = new(new TensorShape(b, hidden), DType.F32); backend.Add(temb, tTime, tVec);
        tTime.Dispose(); tVec.Dispose();

        if (_cfg.GuidanceEmbed && _guidW0 is not null)
        {
            Tensor tGuid = TimestepMlp(backend, b, hidden, guidance, _guidW0!, _guidB0!, _guidW1!, _guidB1!);
            Tensor sum = new(temb.Shape, DType.F32); backend.Add(sum, temb, tGuid);
            temb.Dispose(); tGuid.Dispose(); temb = sum;
        }
        return temb;
    }

    /// <summary>Sinusoidal(256) → Linear → SiLU → Linear, shared by the time and guidance embedders.</summary>
    private static Tensor TimestepMlp(IBackend backend, int b, int hidden, float scalar,
        Tensor w0, Tensor b0, Tensor w1, Tensor b1)
    {
        Tensor sin = new(new TensorShape(b, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, scalar, b, 256);
        Tensor t0 = new(new TensorShape(b, hidden), DType.F32); backend.Linear(t0, sin, w0, b0); sin.Dispose();
        Tensor t0a = new(t0.Shape, DType.F32); backend.Silu(t0a, t0); t0.Dispose();
        Tensor t1 = new(new TensorShape(b, hidden), DType.F32); backend.Linear(t1, t0a, w1, b1); t0a.Dispose();
        return t1;
    }

    private static void Patchify(Tensor latent, Tensor patches, int b, int c, int T, int H, int W,
        int pT, int pH, int pW, int tOut, int hOut, int wOut)
    {
        float* lp = (float*)latent.DataPointer; float* dp = (float*)patches.DataPointer;
        int patchVec = c * pT * pH * pW;
        for (int bb = 0; bb < b; bb++)
            for (int ti = 0; ti < tOut; ti++) for (int hi = 0; hi < hOut; hi++) for (int wi = 0; wi < wOut; wi++)
            {
                long token = ((long)ti * hOut + hi) * wOut + wi;
                long dstBase = ((long)bb * (tOut * hOut * wOut) + token) * patchVec;
                int idx = 0;
                for (int cc = 0; cc < c; cc++) for (int kt = 0; kt < pT; kt++) for (int kh = 0; kh < pH; kh++) for (int kw = 0; kw < pW; kw++)
                {
                    long src = ((((long)bb * c + cc) * T + (ti * pT + kt)) * H + (hi * pH + kh)) * W + (wi * pW + kw);
                    dp[dstBase + idx++] = lp[src];
                }
            }
    }

    private static void Unpatchify(Tensor tokens, Tensor outVol, int b, int oc, int T, int H, int W,
        int pT, int pH, int pW, int tOut, int hOut, int wOut)
    {
        float* tp = (float*)tokens.DataPointer; float* op = (float*)outVol.DataPointer;
        int outVec = oc * pT * pH * pW;
        for (int bb = 0; bb < b; bb++)
            for (int ti = 0; ti < tOut; ti++) for (int hi = 0; hi < hOut; hi++) for (int wi = 0; wi < wOut; wi++)
            {
                long token = ((long)ti * hOut + hi) * wOut + wi;
                long srcBase = ((long)bb * (tOut * hOut * wOut) + token) * outVec;
                int idx = 0;
                for (int cc = 0; cc < oc; cc++) for (int kt = 0; kt < pT; kt++) for (int kh = 0; kh < pH; kh++) for (int kw = 0; kw < pW; kw++)
                {
                    long dst = ((((long)bb * oc + cc) * T + (ti * pT + kt)) * H + (hi * pH + kh)) * W + (wi * pW + kw);
                    op[dst] = tp[srcBase + idx++];
                }
            }
    }

    /// <summary>In-place AdaLN-continuous: <c>x = x*(1+scale)+shift</c> from packed mod <c>[B, 2*hidden]</c>, chunked <c>[shift, scale]</c> (Tencent <c>final_layer.adaLN_modulation</c> order).</summary>
    private static void Modulate(Tensor x, Tensor mod, int b, int s, int hidden)
    {
        float* xp = (float*)x.DataPointer; float* mp = (float*)mod.DataPointer;
        for (int bb = 0; bb < b; bb++)
        {
            float* shift = mp + (long)bb * 2 * hidden;
            float* scale = shift + hidden;
            for (int ss = 0; ss < s; ss++)
            {
                float* row = xp + ((long)bb * s + ss) * hidden;
                for (int c = 0; c < hidden; c++) row[c] = row[c] * (1f + scale[c]) + shift[c];
            }
        }
    }

    public void Dispose() { }
}
