using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Video DiT block (<c>WanTransformerBlock</c>), ported from diffusers. B=1 over <c>[S, dim]</c>: self-attention + per-head 3D RoPE → cross-attention to umT5 → gelu-approx FFN. AdaLN is 6-param (self-attn shift/scale/gate + FFN shift/scale/gate; cross-attn ungated). Pre-norms are FP32 LayerNorm (norm1/norm3 no-affine, norm2 affine when cross_attn_norm); QK-norm is RMSNorm-across-heads. Reuses backend <c>LayerNorm</c>/<c>RmsNorm</c>/<c>ScaledDotProductAttention</c> + <see cref="WanRope"/>.</summary>
public sealed unsafe class WanVideoBlock : IStreamingBlock
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly float _eps;
    private readonly bool _crossAttnNorm;

    private Tensor? _scaleShift;        // [6, dim]
    private Tensor? _norm2W, _norm2B;   // FP32LayerNorm affine (cross), when cross_attn_norm
    // attn1 (self) + attn2 (cross)
    private Tensor?[] _q = new Tensor?[2], _qB = new Tensor?[2], _k = new Tensor?[2], _kB = new Tensor?[2],
        _v = new Tensor?[2], _vB = new Tensor?[2], _o = new Tensor?[2], _oB = new Tensor?[2], _nq = new Tensor?[2], _nk = new Tensor?[2];
    private Tensor? _ffProjW, _ffProjB, _ffOutW, _ffOutB;
    // I2V image cross-attention KV (cross-attn to CLIP image context); present only when the checkpoint ships them.
    private Tensor? _addK, _addKB, _addV, _addVB, _normAddedK;

    public WanVideoBlock(WanVideoConfig c, bool crossAttnNorm)
    {
        _dim = c.InnerDim;
        _heads = c.NumHeads;
        _headDim = c.HeadDim;
        _eps = c.Eps;
        _crossAttnNorm = crossAttnNorm;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _scaleShift = LoadF32(w, $"{prefix}.scale_shift_table");   // [1,6,dim] → flat 6*dim
        LoadAttn(w, $"{prefix}.attn1", 0);
        LoadAttn(w, $"{prefix}.attn2", 1);
        // I2V image cross-attention (only present in image-conditioned checkpoints).
        if (w.TryGetValue($"{prefix}.attn2.add_k_proj.weight", out Tensor? addK))
        {
            _addK = addK; w.TryGetValue($"{prefix}.attn2.add_k_proj.bias", out _addKB);
            _addV = w[$"{prefix}.attn2.add_v_proj.weight"]; w.TryGetValue($"{prefix}.attn2.add_v_proj.bias", out _addVB);
            _normAddedK = LoadF32(w, $"{prefix}.attn2.norm_added_k.weight");
        }
        // norm2 affine is read by the manual host-pointer LayerNorm loop → cast BOTH weight and bias to F32
        // (bf16 checkpoints else feed garbage bias bytes reinterpreted as f32).
        if (_crossAttnNorm) { _norm2W = LoadF32(w, $"{prefix}.norm2.weight"); _norm2B = w.TryGetValue($"{prefix}.norm2.bias", out Tensor? n2b) ? (n2b.DType == DType.F32 ? n2b : n2b.CastTo(DType.F32)) : null; }
        _ffProjW = w[$"{prefix}.ffn.net.0.proj.weight"]; w.TryGetValue($"{prefix}.ffn.net.0.proj.bias", out _ffProjB);
        _ffOutW = w[$"{prefix}.ffn.net.2.weight"]; w.TryGetValue($"{prefix}.ffn.net.2.bias", out _ffOutB);
        long bytes = 0;
        foreach (Tensor t in EnumerateWeights()) bytes += t.DType.ComputeByteCount(t.ElementCount);
        EstimatedWeightBytes = bytes;
    }

    /// <inheritdoc/>
    /// <remarks>Via <see cref="DType.ComputeByteCount"/>, not <c>ElementCount * SizeInBytes</c> — the latter reports
    /// 0 for block-quantized dtypes, which would size a streaming window at zero.</remarks>
    public long EstimatedWeightBytes { get; private set; }

    private void LoadAttn(IReadOnlyDictionary<string, Tensor> w, string p, int i)
    {
        _q[i] = w[$"{p}.to_q.weight"]; w.TryGetValue($"{p}.to_q.bias", out Tensor? qb); _qB[i] = qb;
        _k[i] = w[$"{p}.to_k.weight"]; w.TryGetValue($"{p}.to_k.bias", out Tensor? kb); _kB[i] = kb;
        _v[i] = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out Tensor? vb); _vB[i] = vb;
        _o[i] = w[$"{p}.to_out.0.weight"]; w.TryGetValue($"{p}.to_out.0.bias", out Tensor? ob); _oB[i] = ob;
        _nq[i] = LoadF32(w, $"{p}.norm_q.weight");
        _nk[i] = LoadF32(w, $"{p}.norm_k.weight");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_scaleShift is not null) yield return _scaleShift;
        if (_norm2W is not null) yield return _norm2W;
        if (_norm2B is not null) yield return _norm2B;
        for (int i = 0; i < 2; i++)
            foreach (Tensor? t in new[] { _q[i], _qB[i], _k[i], _kB[i], _v[i], _vB[i], _o[i], _oB[i], _nq[i], _nk[i] })
                if (t is not null) yield return t;
        foreach (Tensor? t in new[] { _ffProjW, _ffProjB, _ffOutW, _ffOutB }) if (t is not null) yield return t;
        foreach (Tensor? t in new[] { _addK, _addKB, _addV, _addVB, _normAddedK }) if (t is not null) yield return t;
    }

    /// <summary>Forward over <c>[S, dim]</c>. <paramref name="temb"/> is the per-block modulation <c>[G, 6, dim]</c>
    /// (timestep_proj); the block adds its <c>scale_shift_table</c>. G=1 is the standard scalar-timestep path;
    /// G=latent-frames with <paramref name="tokensPerGroup"/> tokens each is the TI2V per-frame-timestep path
    /// (tokens are in (t,h,w) order, so each frame's tokens are contiguous). <paramref name="postCrossAttnHook"/>
    /// runs between the cross-attention residual and the FFN — Matrix-Game attaches its ActionModule there (the hook
    /// mutates the hidden state in place). <paramref name="selfAttnMask"/> is an optional additive mask broadcastable
    /// to <c>[1, heads, S, S]</c> — Matrix-Game 2's block-causal + local-window attention.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor encoder, Tensor temb, WanRope rope, Tensor cos, Tensor sin, int tokensPerGroup,
        Action<Tensor>? postCrossAttnHook = null, Tensor? selfAttnMask = null, int imageContextLen = 0, string? dbg = null,
        Action<Tensor>? postSelfAttnHook = null, Func<Tensor, Tensor, (Tensor K, Tensor V)>? selfAttnKvExchange = null)
    {
        int s = (int)hidden.Shape[0];
        (Tensor shiftMsa, Tensor scaleMsa, Tensor gateMsa, Tensor cShift, Tensor cScale, Tensor cGate) = Modulation(backend, temb);
        if (dbg != null) { WanVideoDebugDump.Dump($"{dbg}_scaleMsa", scaleMsa); WanVideoDebugDump.Dump($"{dbg}_gateMsa", gateMsa); }

        // 1. self-attn
        Tensor n1 = ApplyShiftScale(backend, LayerNorm(backend, hidden, null, null, s), scaleMsa, shiftMsa, s, tokensPerGroup);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_n1", n1);
        Tensor attn1 = Attention(backend, n1, n1, 0, applyRope: true, rope, cos, sin, s, s, selfAttnMask, selfAttnKvExchange);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_attn1", attn1);
        n1.Dispose();
        Tensor h1 = GatedAdd(backend, hidden, attn1, gateMsa, s, tokensPerGroup);
        attn1.Dispose();

        // Optional post-self-attn injection (Matrix-Game 3.0's per-block Plücker camera modulation
        // `x = (1+cam_scale)·x + cam_shift`, applied between the self-attn residual and cross-attn); mutates h1 in place.
        postSelfAttnHook?.Invoke(h1);

        // 2. cross-attn (to umT5 text, plus optional CLIP image context for I2V)
        Tensor n2 = LayerNorm(backend, h1, _norm2W, _norm2B, s);
        Tensor attn2 = CrossAttention(backend, n2, encoder, imageContextLen);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_attn2", attn2);
        // Matrix-Game 3.0's memory-mode block builds the cross-attn residual on the NORMED hidden state
        // (`x = norm3(x); x = x + cross_attn(x)` in wan/modules/model.py cross_attn_ffn), so norm3 is destructive.
        // Default (whole video fleet) keeps the standard `x + cross_attn(norm3(x))` residual on the un-normed h1.
        Tensor h2 = CrossAttnResidualNormed ? AddRows(backend, n2, attn2, s) : AddRows(backend, h1, attn2, s);
        n2.Dispose();
        h1.Dispose();
        attn2.Dispose();

        postCrossAttnHook?.Invoke(h2);

        // 3. ffn
        Tensor n3 = ApplyShiftScale(backend, LayerNorm(backend, h2, null, null, s), cScale, cShift, s, tokensPerGroup);
        Tensor ff = Ffn(backend, n3, s);
        if (dbg != null) WanVideoDebugDump.Dump($"{dbg}_ff", ff);
        n3.Dispose();
        Tensor outT = GatedAdd(backend, h2, ff, cGate, s, tokensPerGroup);
        h2.Dispose();
        ff.Dispose();

        foreach (Tensor t in new[] { shiftMsa, scaleMsa, gateMsa, cShift, cScale, cGate }) t.Dispose();
        return outT;
    }

    private Tensor Attention(IBackend backend, Tensor qInput, Tensor kvInput, int idx, bool applyRope,
        WanRope rope, Tensor cos, Tensor sin, int sq, int sk, Tensor? mask = null,
        Func<Tensor, Tensor, (Tensor K, Tensor V)>? kvExchange = null)
    {
        Tensor q = new Tensor(new TensorShape(sq, _dim), DType.F32); backend.Linear(q, qInput, _q[idx]!, _qB[idx]);
        Tensor k = new Tensor(new TensorShape(sk, _dim), DType.F32); backend.Linear(k, kvInput, _k[idx]!, _kB[idx]);
        Tensor v = new Tensor(new TensorShape(sk, _dim), DType.F32); backend.Linear(v, kvInput, _v[idx]!, _vB[idx]);

        Tensor qn = new Tensor(q.Shape, DType.F32); backend.RmsNorm(qn, q, _nq[idx]!, _eps); q.Dispose();
        Tensor kn = new Tensor(k.Shape, DType.F32); backend.RmsNorm(kn, k, _nk[idx]!, _eps); k.Dispose();

        if (applyRope)   // per-head; [S,dim] is contiguous as [S,heads,headDim]
        {
            // GPU RoPE for the standard shared-cos path (cos rank-2 [S, headDim]) — keeps qn/kn on-device so the
            // whole attention chain stays GPU-resident. The per-head sigma_theta variant (rank-3 cos) keeps the CPU ref.
            if (cos.Shape.Rank == 2)
            {
                backend.WanRopeInterleaved(qn, cos, sin, sq, _heads, _headDim);
                backend.WanRopeInterleaved(kn, cos, sin, sk, _heads, _headDim);
            }
            else
            {
                // Per-head sigma_theta rope (rank-3 cos) — GPU kernel keeps qn/kn device-resident (the host
                // ApplyRotary loop was the dominant MG3 backbone cost); CPU falls back to the identical default impl.
                backend.WanRopeInterleavedPerHead(qn, cos, sin, sq, _heads, _headDim);
                backend.WanRopeInterleavedPerHead(kn, cos, sin, sk, _heads, _headDim);
            }
        }

        if (kvExchange is not null)   // context parallel: trade local post-RoPE K/V for the full sequence's
        {
            (kn, v) = kvExchange(kn, v);   // consumes the locals; returns [S, dim]
            sk = (int)kn.Shape[0];
        }

        Tensor qMh = ToBhsd(backend, qn, sq); qn.Dispose();
        Tensor kMh = ToBhsd(backend, kn, sk); kn.Dispose();
        Tensor vMh = ToBhsd(backend, v, sk); v.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attn = new Tensor(new TensorShape(1, _heads, sq, _headDim), DType.F32);
        // allowF16: Wan RMS-norms Q and K (qn/kn above), so pre-softmax scores are bounded → F16 attention is safe
        // and ~halves the (dominant) score-matrix memory traffic. The engine still keeps F32 when a mask is present.
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, scale, allowF16: true);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = FromBhsd(backend, attn, sq); attn.Dispose();
        Tensor outT = new Tensor(new TensorShape(sq, _dim), DType.F32); backend.Linear(outT, flat, _o[idx]!, _oB[idx]);
        flat.Dispose();
        return outT;
    }

    /// <summary>Cross-attention to the umT5 text context, with an optional CLIP image branch (I2V): the first
    /// <paramref name="imageContextLen"/> encoder rows are the projected image context (attended via
    /// <c>add_k_proj</c>/<c>add_v_proj</c>), the rest are text. Both branches share the query; their per-head outputs
    /// are summed before the output projection (matching diffusers' WanAttnProcessor).</summary>
    private Tensor CrossAttention(IBackend backend, Tensor qInput, Tensor encoder, int imageContextLen)
    {
        int sq = (int)qInput.Shape[0];
        int l = (int)encoder.Shape[0];
        int textLen = l - imageContextLen;

        Tensor q = new Tensor(new TensorShape(sq, _dim), DType.F32); backend.Linear(q, qInput, _q[1]!, _qB[1]);
        Tensor qn = new Tensor(q.Shape, DType.F32); backend.RmsNorm(qn, q, _nq[1]!, _eps); q.Dispose();
        Tensor qMh = ToBhsd(backend, qn, sq); qn.Dispose();

        Tensor textRows = imageContextLen > 0 ? SliceRows(backend, encoder, imageContextLen, textLen) : encoder;
        Tensor flat = AttnBranch(backend, qMh, textRows, _k[1]!, _kB[1], _v[1]!, _vB[1], _nk[1]!, sq, textLen);
        if (imageContextLen > 0) textRows.Dispose();

        if (imageContextLen > 0 && _addK is not null)
        {
            Tensor imgRows = SliceRows(backend, encoder, 0, imageContextLen);
            Tensor flatImg = AttnBranch(backend, qMh, imgRows, _addK!, _addKB, _addV!, _addVB, _normAddedK!, sq, imageContextLen);
            imgRows.Dispose();
            // Device add — the old host AddInPlace pointer-loop drained BOTH [sq, dim] branch outputs D2H, summed on
            // the CPU, and re-uploaded on next use: ~0.5 GB of synchronous PCIe round-trips per block per forward
            // (×40 blocks ×2 CFG = the dominant Wan2.1-CLIP-I2V cost over T2V).
            Tensor summed = AddRows(backend, flat, flatImg, sq);
            flat.Dispose();
            flatImg.Dispose();
            flat = summed;
        }
        qMh.Dispose();

        Tensor outT = new Tensor(new TensorShape(sq, _dim), DType.F32); backend.Linear(outT, flat, _o[1]!, _oB[1]);
        flat.Dispose();
        return outT;
    }

    /// <summary>One cross-attention KV branch: project + QK-norm the key, SDPA against the shared multi-head query
    /// <paramref name="qMh"/> <c>[1, heads, sq, headDim]</c>, return the flattened <c>[sq, dim]</c> (pre output-proj).</summary>
    private Tensor AttnBranch(IBackend backend, Tensor qMh, Tensor kvRows, Tensor kW, Tensor? kB, Tensor vW, Tensor? vB,
        Tensor kNorm, int sq, int sk)
    {
        Tensor k = new Tensor(new TensorShape(sk, _dim), DType.F32); backend.Linear(k, kvRows, kW, kB);
        Tensor v = new Tensor(new TensorShape(sk, _dim), DType.F32); backend.Linear(v, kvRows, vW, vB);
        Tensor kn = new Tensor(k.Shape, DType.F32); backend.RmsNorm(kn, k, kNorm, _eps); k.Dispose();
        Tensor kMh = ToBhsd(backend, kn, sk); kn.Dispose();
        Tensor vMh = ToBhsd(backend, v, sk); v.Dispose();
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attn = new Tensor(new TensorShape(1, _heads, sq, _headDim), DType.F32);
        // allowF16: cross-attn Q and K are RMS-normed too (qn / kNorm) → bounded scores → F16-safe.
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, null, scale, allowF16: true);
        kMh.Dispose(); vMh.Dispose();
        Tensor flat = FromBhsd(backend, attn, sq); attn.Dispose();
        return flat;
    }

    /// <summary>Device row slice of the encoder context — a host <c>Buffer.MemoryCopy</c> here produced a FRESH
    /// host tensor per block per forward whose first device use was a cache MISS (full stream drain + pageable H2D,
    /// the sync-H2D disease).</summary>
    private Tensor SliceRows(IBackend backend, Tensor x, int start, int len)
    {
        Tensor o = new Tensor(new TensorShape(len, _dim), DType.F32);
        backend.SliceRows(o, x, start);
        return o;
    }

    /// <summary>Activation dtype for the FFN's big [s, ffn_dim] intermediate — <c>DType.F16</c> halves its
    /// bandwidth (the dominant per-block activation), F32-accumulated in the GEMMs, tiny cast boundaries. Default
    /// F32 keeps every existing caller (the whole video fleet) byte-identical; opt-in models (Matrix-Game 2.0) set
    /// it to <see cref="DiTBlocks.DitDtype.Act"/>. QK/cross-attn already run F16 via <c>allowF16</c> SDPA.</summary>
    internal DType FfnDtype = DType.F32;

    /// <summary>When true, the cross-attention residual is built on the NORMED hidden state (Matrix-Game 3.0's
    /// memory-mode block: <c>x = norm3(x); x = x + cross_attn(x)</c>). Default false = the standard Wan
    /// <c>x + cross_attn(norm3(x))</c> residual (the whole video fleet). Opt-in per model; no video regression.</summary>
    internal bool CrossAttnResidualNormed = false;

    private Tensor Ffn(IBackend backend, Tensor x, int s)
    {
        int inner = (int)_ffProjW!.Shape[0];
        if (FfnDtype == DType.F16 && backend.SupportsF16Activations)
        {
            Tensor xF16 = new Tensor(x.Shape, DType.F16); backend.CastToF16(xF16, x);
            Tensor projH = new Tensor(new TensorShape(s, inner), DType.F16); backend.Linear(projH, xF16, _ffProjW!, _ffProjB);
            xF16.Dispose();
            Tensor actH = new Tensor(projH.Shape, DType.F16); backend.Gelu(actH, projH); projH.Dispose();
            Tensor outH = new Tensor(new TensorShape(s, _dim), DType.F16); backend.Linear(outH, actH, _ffOutW!, _ffOutB);
            actH.Dispose();
            Tensor outF = new Tensor(new TensorShape(s, _dim), DType.F32); backend.CastToF32(outF, outH);
            outH.Dispose();
            return outF;
        }
        Tensor proj = new Tensor(new TensorShape(s, inner), DType.F32); backend.Linear(proj, x, _ffProjW!, _ffProjB);
        Tensor act = new Tensor(proj.Shape, DType.F32); backend.Gelu(act, proj); proj.Dispose();
        Tensor outT = new Tensor(new TensorShape(s, _dim), DType.F32); backend.Linear(outT, act, _ffOutW!, _ffOutB);
        act.Dispose();
        return outT;
    }

    /// <summary>Adds <c>scale_shift_table</c> to the timestep_proj <c>[G, 6, dim]</c>; returns 6 modulation tensors of <c>[G, dim]</c>.
    /// <para>G=1 (scalar-timestep T2V/I2V, the hot path) builds the 6 tensors <b>on-device</b> via <see cref="IBackend.SliceRows"/>
    /// + <see cref="IBackend.GatedResidualLastDim"/>, so the results are GPU-resident activations. This is a load-bearing perf
    /// fix: the old host-pointer loop produced fresh CPU tensors whose first <c>CopyToDevice</c> (in every downstream AddScalar /
    /// AffineBroadcast / GatedResidual) is a cache MISS, and the miss path does a full stream <c>SyncStream</c> before its H2D —
    /// so each block drained the whole async attention/FFN pipeline 3× (measured: ~85 s of a ~90 s Wan-1.3B gen sat in that
    /// GatedResidual sync). Keeping modulation on-device makes every downstream upload a cache HIT → no per-block barriers.
    /// The multi-group TI2V path (per-frame timesteps) keeps the CPU reference.</para></summary>
    private (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) Modulation(IBackend backend, Tensor temb)
    {
        int g = (int)temb.Shape[0];
        if (g == 1)
        {
            // out[m] = scale_shift_table[m,:] + temb[0,m,:], each [1, dim], GPU-resident.
            Tensor ones = Ones();   // [1, dim], device-promoted after first use — the "+1·x" add gate
            Tensor[] outs = new Tensor[6];
            for (int m = 0; m < 6; m++)
            {
                Tensor ssM = new Tensor(new TensorShape(1, _dim), DType.F32);
                backend.SliceRows(ssM, _scaleShift!, m);            // scale_shift_table row m
                Tensor tbM = new Tensor(new TensorShape(1, _dim), DType.F32);
                backend.SliceRows(tbM, temb, m);                    // temb[0, m, :]
                Tensor o = new Tensor(new TensorShape(1, _dim), DType.F32);
                backend.GatedResidualLastDim(o, tbM, ssM, ones);    // o = tbM + 1·ssM
                ssM.Dispose(); tbM.Dispose();
                outs[m] = o;
            }
            return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]);
        }
        // Multi-group (per-frame timesteps: S2V/TI2V/Animate) — GPU-resident since 2026-07-09. The old
        // host loop here was the S2V 15x-vs-T2V wall: fresh CPU tensors -> downstream CopyToDevice cache
        // MISS -> full stream drain, ~4 big bounces per block per forward (profiled 324 H2D_MISS_BIG/step).
        // Permute [G,6,dim] -> [6,G,dim] so each modulation's G rows are contiguous, then slice + broadcast-add.
        Tensor ones6 = Ones();
        Tensor perm = new Tensor(new TensorShape(6 * g, _dim), DType.F32);
        backend.Permute0213(perm, temb, g, 6, _dim);        // [1,G,6,dim] -> [1,6,G,dim]
        Tensor[] outsG = new Tensor[6];
        for (int m = 0; m < 6; m++)
        {
            Tensor tbM = new Tensor(new TensorShape(g, _dim), DType.F32);
            backend.SliceRows(tbM, perm, m * g);            // temb[:, m, :]
            Tensor ssM = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.SliceRows(ssM, _scaleShift!, m);        // scale_shift_table row m
            Tensor o = new Tensor(new TensorShape(g, _dim), DType.F32);
            backend.AffineBroadcastLastDim(o, tbM, ones6, ssM);   // o = tbM*1 + ssM (row-broadcast)
            tbM.Dispose(); ssM.Dispose();
            outsG[m] = o;
        }
        perm.Dispose();
        return (outsG[0], outsG[1], outsG[2], outsG[3], outsG[4], outsG[5]);
    }

    /// <summary>Expands per-group rows <c>[G, dim]</c> to per-token rows <c>[G*tokensPerGroup, dim]</c> by row
    /// repetition — GPU-resident via <see cref="IBackend.UpsampleNearest2D"/> over an NCHW view.</summary>
    private Tensor ExpandGroups(IBackend backend, Tensor grouped, int groups, int tokensPerGroup)
    {
        Tensor src = new Tensor(new TensorShape(1, 1, groups, _dim), DType.F32);
        backend.SliceRows(src, grouped, 0);
        Tensor up = new Tensor(new TensorShape(1, 1, (long)groups * tokensPerGroup, _dim), DType.F32);
        backend.UpsampleNearest2D(up, src, tokensPerGroup, 1);
        src.Dispose();
        Tensor outT = new Tensor(new TensorShape((long)groups * tokensPerGroup, _dim), DType.F32);
        backend.SliceRows(outT, up, 0);
        up.Dispose();
        return outT;
    }

    /// <summary>FP32 LayerNorm over the last dim with optional affine. GPU-resident (backend ops) — the per-op
    /// host-pointer loop it replaced dominated full-res runtime (14040×3072 per call, forcing D2H/H2D each block).
    /// Affine-with-bias → <see cref="IBackend.LayerNorm"/>; no-affine → <see cref="IBackend.LayerNormNoAffine"/>;
    /// the rare affine-without-bias case keeps the CPU reference.</summary>
    private Tensor LayerNorm(IBackend backend, Tensor x, Tensor? weight, Tensor? bias, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        // No-affine (n1/n3): direct GPU op. Affine (n2): normalize on GPU then apply weight/bias as a broadcast
        // affine on GPU (backend.LayerNorm has no CUDA kernel, but LayerNormNoAffine + AffineBroadcastLastDim do,
        // so we stay GPU-resident). weight/bias are [dim] = [1, dim] broadcast over the sequence. The affine step
        // must NOT be in-place: re-caching the same tensor orphans its old device buffer (FreeDevice skips cached
        // pointers), leaking it — so normalize into a scratch tensor, then affine into the output.
        if (weight is null)
        {
            backend.LayerNormNoAffine(o, x, _eps);
            return o;
        }
        using Tensor normed = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.LayerNormNoAffine(normed, x, _eps);
        backend.AffineBroadcastLastDim(o, normed, weight, bias);
        return o;
    }

    /// <summary>AdaLN modulate: <c>out = x·(1+scale) + shift</c>, scale/shift broadcast per group over the last dim.
    /// Consumes <paramref name="x"/> (disposes it) and returns the result. G=1 (scalar timestep) is GPU-resident via
    /// <see cref="IBackend.AffineBroadcastLastDim"/> (pre-adding 1 to scale on-GPU) into a FRESH tensor — never
    /// in-place, which would orphan/leak x's cached device buffer. The multi-group TI2V path keeps the CPU reference
    /// (mutates x in place and returns it).</summary>
    private Tensor ApplyShiftScale(IBackend backend, Tensor x, Tensor scale, Tensor shift, int s, int tokensPerGroup)
    {
        if (tokensPerGroup == s)   // G == 1: scale/shift are [1, dim] broadcast over all S tokens
        {
            using Tensor scaleP1 = new Tensor(scale.Shape, DType.F32);
            backend.AddScalar(scaleP1, scale, 1.0f);           // (1 + scale)
            Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.AffineBroadcastLastDim(o, x, scaleP1, shift);   // o = x·(1+scale) + shift  (non-in-place)
            x.Dispose();
            return o;
        }
        int groupsSS = (int)scale.Shape[0];
        if ((long)groupsSS * tokensPerGroup == s)
        {
            // GPU multi-group path (2026-07-09): expand scale/shift to token rows, then elementwise FMA.
            using Tensor scaleExp = ExpandGroups(backend, scale, groupsSS, tokensPerGroup);
            using Tensor shiftExp = ExpandGroups(backend, shift, groupsSS, tokensPerGroup);
            using Tensor scaleP1 = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.AddScalar(scaleP1, scaleExp, 1.0f);
            using Tensor prod = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.Mul(prod, x, scaleP1);
            Tensor oG = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.Add(oG, prod, shiftExp);
            x.Dispose();
            return oG;
        }
        float* xp = (float*)x.DataPointer; float* sc = (float*)scale.DataPointer; float* sh = (float*)shift.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long g = (long)(i / tokensPerGroup) * _dim;
            for (int d = 0; d < _dim; d++) xp[i * _dim + d] = xp[i * _dim + d] * (1f + sc[g + d]) + sh[g + d];
        }
        return x;
    }

    /// <summary>Gated residual: <c>out = residual + gate·value</c>, gate broadcast per group over the last dim.
    /// G=1 is GPU-resident via <see cref="IBackend.GatedResidualLastDim"/>; multi-group TI2V keeps the CPU reference.</summary>
    private Tensor GatedAdd(IBackend backend, Tensor residual, Tensor value, Tensor gate, int s, int tokensPerGroup)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        if (tokensPerGroup == s)   // G == 1: gate is [1, dim]
        {
            backend.GatedResidualLastDim(o, residual, value, gate);
            return o;
        }
        int groupsGA = (int)gate.Shape[0];
        if ((long)groupsGA * tokensPerGroup == s)
        {
            // GPU multi-group path (2026-07-09): expand the gate to token rows, then elementwise ops.
            using Tensor gateExp = ExpandGroups(backend, gate, groupsGA, tokensPerGroup);
            using Tensor prod = new Tensor(new TensorShape(s, _dim), DType.F32);
            backend.Mul(prod, value, gateExp);
            backend.Add(o, residual, prod);
            return o;
        }
        float* rp = (float*)residual.DataPointer; float* vp = (float*)value.DataPointer; float* gp = (float*)gate.DataPointer; float* op = (float*)o.DataPointer;
        for (int i = 0; i < s; i++)
        {
            long g = (long)(i / tokensPerGroup) * _dim;
            for (int d = 0; d < _dim; d++) op[i * _dim + d] = rp[i * _dim + d] + gp[g + d] * vp[i * _dim + d];
        }
        return o;
    }

    // Cached [1, dim] ones row so plain elementwise add (backend.Add has no CUDA kernel) can run as a GPU
    // GatedResidualLastDim: out = a + 1·b. Built once on first use; fill-before-publish + CAS because two
    // branch/rank threads (CFG-parallel, context-parallel) share the block and race the first touch.
    private Tensor? _ones;
    private Tensor Ones()
    {
        if (_ones is null)
        {
            Tensor ones = new Tensor(new TensorShape(1, _dim), DType.F32);
            float* p = (float*)ones.DataPointer;
            for (int i = 0; i < _dim; i++) p[i] = 1f;
            if (Interlocked.CompareExchange(ref _ones, ones, null) is not null) ones.Dispose();
        }
        return _ones;
    }

    private Tensor AddRows(IBackend backend, Tensor a, Tensor b, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.GatedResidualLastDim(o, a, b, Ones());   // a + 1·b  (GPU)
        return o;
    }

    // [s, dim]=[s, heads, headDim] → [1, heads, s, headDim], GPU-resident via Permute0213 (explicit dims, reads the
    // device buffer directly — correct now that RoPE is GPU so x never leaves the device).
    private Tensor ToBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(1, _heads, s, _headDim), DType.F32);
        backend.Permute0213(o, x, s, _heads, _headDim);
        return o;
    }

    // [1, heads, s, headDim] → [s, dim]=[s, heads, headDim], GPU-resident via Permute0213 (inverse of ToBhsd).
    private Tensor FromBhsd(IBackend backend, Tensor x, int s)
    {
        Tensor o = new Tensor(new TensorShape(s, _dim), DType.F32);
        backend.Permute0213(o, x, _heads, s, _headDim);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string k) { Tensor t = w[k]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
