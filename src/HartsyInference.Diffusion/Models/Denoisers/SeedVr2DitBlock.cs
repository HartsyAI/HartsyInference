using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One SeedVR2 NaDiT block: windowed joint attention + SwiGLU MLP, both wrapped in AdaSingle
/// modulation (emb slice + per-block learned vectors, additive). Blocks below mm_layers hold separate
/// vid/txt weights (<c>.vid./.txt.</c>); above, one shared set (<c>.all.</c>). The LAST block computes the
/// text branch through attention (joint attention needs it) but skips text mlp/ada — <c>vid_only</c>.
/// <para>Device-resident: tokens stay <c>[L, C]</c> backend tensors end-to-end — RMS/modulate/gate via the
/// DiT glue kernels, fused QKV split + per-head RMS qk-norm, window-local rope via
/// <c>WanRopeInterleaved</c> over <see cref="SeedVr2DevicePlan"/>'s global-token-order tables, window
/// gather/scatter via <c>RowGather</c>/<c>RowScatterAdd</c> index tensors, same-shape windows batched into
/// one SDPA call. Text is REPLICATED into every window and MEAN-pooled across windows afterwards
/// (scatter-add + 1/N scale). The emb+ada modulation vectors are additive constants per timestep, combined
/// once and cached (<see cref="PrepareMod"/>).</para></summary>
public sealed class SeedVr2DitBlock
{
    private readonly SeedVr2Config _config;
    private readonly bool _shared;
    private readonly bool _isLast;
    private readonly bool _shifted;

    private Tensor _qkvVidW = null!, _projVidW = null!, _projVidB = null!;
    private Tensor? _qkvTxtW, _projTxtW, _projTxtB;
    private Tensor? _mlpVidGateW;
    private Tensor _mlpVidInW = null!, _mlpVidOutW = null!;
    private Tensor? _mlpVidInB, _mlpVidOutB;
    private Tensor? _mlpTxtGateW, _mlpTxtInW, _mlpTxtOutW, _mlpTxtInB, _mlpTxtOutB;
    private Tensor _normQVid = null!, _normKVid = null!;
    private Tensor? _normQTxt, _normKTxt;
    private float[][] _adaVid = null!;   // [attnShift, attnScale, attnGate, mlpShift, mlpScale, mlpGate]
    private float[][]? _adaTxt;

    private readonly List<Tensor> _owned = [];
    private Tensor[]? _mod;      // combined emb+ada vectors, [C]: vid attn s/sc/g, mlp s/sc/g, then txt ditto (when split)
    private float[]? _modEmb;    // emb the cached _mod was built from

    /// <summary>Creates block <paramref name="index"/>; even layers use the regular window partition, odd
    /// layers the shifted one (config <c>window_method</c> alternation).</summary>
    public SeedVr2DitBlock(SeedVr2Config config, int index)
    {
        _config = config;
        _shared = index >= config.MmLayers;
        _isLast = index == config.NumLayers - 1;
        _shifted = index % 2 == 1;
    }

    /// <summary>Whether this block uses the shifted window partition (odd layers).</summary>
    public int Parity => _shifted ? 1 : 0;

    /// <summary>Binds <c>blocks.{i}.*</c> checkpoint keys (verbatim names via the near-identity converter).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, int index)
    {
        string p = $"blocks.{index}";
        string b = _shared ? "all" : "vid";
        _qkvVidW = weights[$"{p}.attn.proj_qkv.{b}.weight"];
        _projVidW = weights[$"{p}.attn.proj_out.{b}.weight"];
        _projVidB = weights[$"{p}.attn.proj_out.{b}.bias"];
        _normQVid = ToF32Tensor(weights[$"{p}.attn.norm_q.{b}.weight"]);
        _normKVid = ToF32Tensor(weights[$"{p}.attn.norm_k.{b}.weight"]);
        _mlpVidGateW = _config.SwiGluMlp ? weights[$"{p}.mlp.{b}.proj_in_gate.weight"] : null;
        _mlpVidInW = weights[$"{p}.mlp.{b}.proj_in.weight"];
        _mlpVidOutW = weights[$"{p}.mlp.{b}.proj_out.weight"];
        weights.TryGetValue($"{p}.mlp.{b}.proj_in.bias", out _mlpVidInB);
        weights.TryGetValue($"{p}.mlp.{b}.proj_out.bias", out _mlpVidOutB);
        _adaVid = LoadAda(weights, p, b);
        if (!_shared)
        {
            _qkvTxtW = weights[$"{p}.attn.proj_qkv.txt.weight"];
            _projTxtW = weights[$"{p}.attn.proj_out.txt.weight"];
            _projTxtB = weights[$"{p}.attn.proj_out.txt.bias"];
            _normQTxt = ToF32Tensor(weights[$"{p}.attn.norm_q.txt.weight"]);
            _normKTxt = ToF32Tensor(weights[$"{p}.attn.norm_k.txt.weight"]);
            _mlpTxtGateW = _config.SwiGluMlp ? weights[$"{p}.mlp.txt.proj_in_gate.weight"] : null;
            _mlpTxtInW = weights[$"{p}.mlp.txt.proj_in.weight"];
            _mlpTxtOutW = weights[$"{p}.mlp.txt.proj_out.weight"];
            weights.TryGetValue($"{p}.mlp.txt.proj_in.bias", out _mlpTxtInB);
            weights.TryGetValue($"{p}.mlp.txt.proj_out.bias", out _mlpTxtOutB);
            _adaTxt = LoadAda(weights, p, "txt");
        }
    }

    private static float[][] LoadAda(IReadOnlyDictionary<string, Tensor> weights, string p, string b) =>
    [
        ToHost(weights[$"{p}.ada.{b}.attn_shift"]), ToHost(weights[$"{p}.ada.{b}.attn_scale"]),
        ToHost(weights[$"{p}.ada.{b}.attn_gate"]), ToHost(weights[$"{p}.ada.{b}.mlp_shift"]),
        ToHost(weights[$"{p}.ada.{b}.mlp_scale"]), ToHost(weights[$"{p}.ada.{b}.mlp_gate"]),
    ];

    // Combined-mod math runs on host F32 regardless of checkpoint dtype (the 7B ships F16-converted for disk).
    private static float[] ToHost(Tensor t)
    {
        if (t.DType == DType.F32)
            return t.AsSpan<float>().ToArray();
        Tensor cast = t.CastTo(DType.F32);
        float[] result = cast.AsSpan<float>().ToArray();
        cast.Dispose();
        return result;
    }

    /// <summary>F32 view of a (possibly F16) checkpoint vector; cast copies are block-owned.</summary>
    private Tensor ToF32Tensor(Tensor t)
    {
        if (t.DType == DType.F32)
            return t;
        Tensor cast = t.CastTo(DType.F32);
        _owned.Add(cast);
        return cast;
    }

    /// <summary>Builds (or reuses) the combined emb+ada modulation vectors for <paramref name="emb"/> —
    /// they are constants per timestep, so one build serves every chunk of a clip.</summary>
    public void PrepareMod(IBackend backend, float[] emb)
    {
        if (_mod is not null && ReferenceEquals(_modEmb, emb))
            return;
        ReleaseMod(backend);
        int c = _config.VidDim;
        List<Tensor> mod = [];
        foreach (float[][] ada in _adaTxt is null ? new[] { _adaVid } : [_adaVid, _adaTxt])
        {
            for (int layer = 0; layer < 2; layer++)
            {
                mod.Add(CombineVector(c, i => emb[i * 6 + layer * 3 + 0] + ada[layer * 3 + 0][i]));
                mod.Add(CombineVector(c, i => emb[i * 6 + layer * 3 + 1] + ada[layer * 3 + 1][i]));
                mod.Add(CombineVector(c, i => emb[i * 6 + layer * 3 + 2] + ada[layer * 3 + 2][i]));
            }
        }
        _mod = [.. mod];
        _modEmb = emb;
        backend.PreloadWeights(_mod);
    }

    private static Tensor CombineVector(int c, Func<int, float> f)
    {
        Tensor t = new Tensor(new TensorShape(c), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < c; i++)
            s[i] = f(i);
        return t;
    }

    private void ReleaseMod(IBackend backend)
    {
        if (_mod is null)
            return;
        backend.FreeWeights(_mod);
        foreach (Tensor t in _mod)
            t.Dispose();
        _mod = null;
        _modEmb = null;
    }

    /// <summary>Frees block-owned tensors (mod vectors, F32 casts of F16 checkpoint vectors).</summary>
    public void ReleaseOwned(IBackend backend)
    {
        ReleaseMod(backend);
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned.Clear();
    }

    // _mod layout: [vid attnShift, attnScale, attnGate, mlpShift, mlpScale, mlpGate, (txt ditto when split)]
    private Tensor ModVec(bool txt, int layer, int component)
    {
        int branch = txt && _adaTxt is not null ? 6 : 0;
        return _mod![branch + layer * 3 + component];
    }

    /// <summary>Runs the block on device-resident token tensors <paramref name="vid"/> [Lv,C] and
    /// <paramref name="txt"/> [Lt,C]; returns the updated pair (inputs are consumed/disposed).
    /// <paramref name="ones"/> is the shared all-ones [C] vector for affine-free RMS norms.</summary>
    public (Tensor Vid, Tensor Txt) Forward(IBackend backend, Tensor vid, Tensor txt,
        SeedVr2DevicePlan plan, Tensor ones)
    {
        if (_mod is null)
            throw new InvalidOperationException("PrepareMod must run before Forward.");
        int c = _config.VidDim;
        int lv = (int)vid.Shape[0], lt = (int)txt.Shape[0];
        SeedVr2DevicePlan.Partition part = plan.Partitions[Parity];
        // v2-only last-layer asymmetry; v1 (7B) runs the final text branch like any other block.
        bool lastVidOnly = _isLast && _config.LastLayerVidOnly;

        // ---- attention ----
        // Last layer (v2): ada is vid_only in the reference — text enters attention PLAIN-NORMED (no
        // ada-in) and its attention output is residual-added UNGATED. Missing this pollutes vid K/V
        // (caught by block3 parity: txt relL2 0.64, vid 1e-2).
        Tensor vidMod = Modulate(backend, vid, ones, txt: false, layer: 0);
        Tensor txtMod;
        if (lastVidOnly)
        {
            txtMod = new Tensor(txt.Shape, DType.F32);
            backend.RmsNorm(txtMod, txt, ones, 1e-5f);
        }
        else
        {
            txtMod = Modulate(backend, txt, ones, txt: true, layer: 0);
        }
        Tensor vidQkv = Linear(backend, vidMod, _qkvVidW, null);
        Tensor txtQkv = Linear(backend, txtMod, _qkvTxtW ?? _qkvVidW, null);
        vidMod.Dispose();
        txtMod.Dispose();

        (Tensor vidAttn, Tensor txtAttn) = WindowedAttention(backend, vidQkv, txtQkv, part, plan, lv, lt);
        vidQkv.Dispose();
        txtQkv.Dispose();

        Tensor vidProj = Linear(backend, vidAttn, _projVidW, _projVidB);
        Tensor txtProj = Linear(backend, txtAttn, _projTxtW ?? _projVidW, _projTxtB ?? _projVidB);
        vidAttn.Dispose();
        txtAttn.Dispose();

        vid = GateAdd(backend, vid, vidProj, txt: false, layer: 0);
        if (lastVidOnly)
        {
            Tensor sum = new Tensor(txt.Shape, DType.F32);
            backend.Add(sum, txt, txtProj);
            txt.Dispose();
            txt = sum;
        }
        else
        {
            txt = GateAdd(backend, txt, txtProj, txt: true, layer: 0);
        }
        vidProj.Dispose();
        txtProj.Dispose();

        // ---- mlp (video always; text skipped on the last layer) ----
        vid = MlpBranch(backend, vid, ones, txtBranch: false,
            _mlpVidGateW, _mlpVidInW, _mlpVidInB, _mlpVidOutW, _mlpVidOutB);
        if (!lastVidOnly)
        {
            txt = MlpBranch(backend, txt, ones, txtBranch: true, _mlpTxtGateW ?? _mlpVidGateW, _mlpTxtInW ?? _mlpVidInW,
                _mlpTxtInB ?? _mlpVidInB, _mlpTxtOutW ?? _mlpVidOutW, _mlpTxtOutB ?? _mlpVidOutB);
        }
        else
        {
            // Reference quirk: vid_only MMModules return txt UNCHANGED, so the "skipped" mlp branch
            // residual-adds txt to itself → final txt = 2×post-attention txt. txt is discarded after this
            // layer; doubled here only so parity traces match the reference exactly (relL2 0.5 otherwise).
            Tensor doubled = new Tensor(txt.Shape, DType.F32);
            backend.Add(doubled, txt, txt);
            txt.Dispose();
            txt = doubled;
        }
        return (vid, txt);
    }

    private Tensor MlpBranch(IBackend backend, Tensor tokens, Tensor ones, bool txtBranch,
        Tensor? gateW, Tensor inW, Tensor? inB, Tensor outW, Tensor? outB)
    {
        Tensor modded = Modulate(backend, tokens, ones, txtBranch, layer: 1);
        Tensor hidden;
        if (gateW is not null)
        {
            // SwiGLU (3B): proj_out(silu(gate(x)) · in(x)), bias-free.
            Tensor gate = Linear(backend, modded, gateW, null);
            Tensor lin = Linear(backend, modded, inW, null);
            backend.Silu(gate, gate);
            backend.Mul(gate, gate, lin);
            lin.Dispose();
            hidden = gate;
        }
        else
        {
            // Plain MLP (7B): proj_out(gelu_tanh(proj_in(x))), biased.
            hidden = Linear(backend, modded, inW, inB);
            backend.Gelu(hidden, hidden);
        }
        modded.Dispose();
        Tensor output = Linear(backend, hidden, outW, outB);
        hidden.Dispose();
        Tensor result = GateAdd(backend, tokens, output, txtBranch, layer: 1);
        output.Dispose();
        return result;
    }

    /// <summary>RMS-norm (no affine) then AdaSingle "in": <c>x̂·(embScale+adaScale) + (embShift+adaShift)</c>.</summary>
    private Tensor Modulate(IBackend backend, Tensor tokens, Tensor ones, bool txt, int layer)
    {
        Tensor normed = new Tensor(tokens.Shape, DType.F32);
        backend.RmsNorm(normed, tokens, ones, 1e-5f);
        Tensor output = new Tensor(tokens.Shape, DType.F32);
        backend.AffineBroadcastLastDim(output, normed, ModVec(txt, layer, 1), ModVec(txt, layer, 0));
        normed.Dispose();
        return output;
    }

    /// <summary>AdaSingle "out" + residual: <c>tokens + branchOut·(embGate+adaGate)</c>; consumes tokens.</summary>
    private Tensor GateAdd(IBackend backend, Tensor tokens, Tensor branchOut, bool txt, int layer)
    {
        Tensor output = new Tensor(tokens.Shape, DType.F32);
        backend.GatedResidualLastDim(output, tokens, branchOut, ModVec(txt, layer, 2));
        tokens.Dispose();
        return output;
    }

    private static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias)
    {
        Tensor output = new Tensor(new TensorShape(input.Shape[0], weight.Shape[0]), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    /// <summary>GEMM weights for backend preload/free (mod vectors and qk-norm affines are plan-lifetime
    /// weight-cache entries, not phase-staged — excluded here).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _qkvVidW;
        yield return _projVidW;
        yield return _projVidB;
        yield return _mlpVidInW;
        yield return _mlpVidOutW;
        if (_mlpVidGateW is not null) yield return _mlpVidGateW;
        if (_mlpVidInB is not null) yield return _mlpVidInB;
        if (_mlpVidOutB is not null) yield return _mlpVidOutB;
        if (_qkvTxtW is not null) yield return _qkvTxtW;
        if (_projTxtW is not null) yield return _projTxtW;
        if (_projTxtB is not null) yield return _projTxtB;
        if (_mlpTxtGateW is not null) yield return _mlpTxtGateW;
        if (_mlpTxtInW is not null) yield return _mlpTxtInW;
        if (_mlpTxtOutW is not null) yield return _mlpTxtOutW;
        if (_mlpTxtInB is not null) yield return _mlpTxtInB;
        if (_mlpTxtOutB is not null) yield return _mlpTxtOutB;
        yield return _normQVid;
        yield return _normKVid;
        if (_normQTxt is not null) yield return _normQTxt;
        if (_normKTxt is not null) yield return _normKTxt;
    }

    private (Tensor Vid, Tensor Txt) WindowedAttention(IBackend backend, Tensor vidQkv, Tensor txtQkv,
        SeedVr2DevicePlan.Partition part, SeedVr2DevicePlan plan, int lv, int lt)
    {
        int heads = _config.Heads, hd = _config.HeadDim, c = _config.VidDim;
        Tensor normQTxt = _normQTxt ?? _normQVid, normKTxt = _normKTxt ?? _normKVid;

        // Fused split + per-head RMS qk-norm, then window-local rope over the plan's per-token tables.
        Tensor vq = new Tensor(new TensorShape(lv, c), DType.F32);
        Tensor vk = new Tensor(new TensorShape(lv, c), DType.F32);
        Tensor vv = new Tensor(new TensorShape(lv, c), DType.F32);
        backend.QkvSplitNorm(vq, vk, vv, vidQkv, _normQVid, _normKVid, 1e-5f);
        Tensor tq = new Tensor(new TensorShape(lt, c), DType.F32);
        Tensor tk = new Tensor(new TensorShape(lt, c), DType.F32);
        Tensor tv = new Tensor(new TensorShape(lt, c), DType.F32);
        backend.QkvSplitNorm(tq, tk, tv, txtQkv, normQTxt, normKTxt, 1e-5f);
        backend.WanRopeInterleaved(vq, part.VidCos, part.VidSin, lv, heads, hd);
        backend.WanRopeInterleaved(vk, part.VidCos, part.VidSin, lv, heads, hd);
        if (plan.TxtCos is not null)
        {
            backend.WanRopeInterleaved(tq, plan.TxtCos, plan.TxtSin!, lt, heads, hd);
            backend.WanRopeInterleaved(tk, plan.TxtCos, plan.TxtSin!, lt, heads, hd);
        }

        Tensor catQ = Concat0(backend, vq, tq);
        Tensor catK = Concat0(backend, vk, tk);
        Tensor catV = Concat0(backend, vv, tv);
        vq.Dispose(); vk.Dispose(); vv.Dispose();
        tq.Dispose(); tk.Dispose(); tv.Dispose();

        // Scatter-add target: vid rows are written once (windows partition the grid); txt rows accumulate
        // across every window and are mean-pooled below (reference unconcat_coalesce .mean).
        Tensor attnCat = new Tensor(new TensorShape(lv + lt, c), DType.F32);
        backend.Fill(attnCat, 0f);

        foreach (SeedVr2DevicePlan.ShapeGroup group in part.Groups)
        {
            int s = group.WindowTokens + lt;
            int g = group.WindowCount;
            long gs = (long)g * s;
            Tensor gq = Gather(backend, catQ, group.Indices, gs, c);
            Tensor gk = Gather(backend, catK, group.Indices, gs, c);
            Tensor gv = Gather(backend, catV, group.Indices, gs, c);

            Tensor qp = PackHeads(backend, gq, g, s, heads, hd);
            Tensor kp = PackHeads(backend, gk, g, s, heads, hd);
            Tensor vp = PackHeads(backend, gv, g, s, heads, hd);
            gq.Dispose(); gk.Dispose(); gv.Dispose();

            Tensor attn = new Tensor(qp.Shape, DType.F32);
            backend.ScaledDotProductAttention(attn, qp, kp, vp, null, 1f / MathF.Sqrt(hd));
            qp.Dispose(); kp.Dispose(); vp.Dispose();

            // [G, heads, s, hd] → [G, s, heads·hd] rows, scattered back through the same index list.
            Tensor back = new Tensor(new TensorShape(gs, c), DType.F32);
            backend.Permute0213(back, attn, heads, s, hd);
            attn.Dispose();
            backend.RowScatterAdd(attnCat, back, group.Indices, (int)gs, c);
            back.Dispose();
        }
        catQ.Dispose(); catK.Dispose(); catV.Dispose();

        Tensor vidOut = new Tensor(new TensorShape(lv, c), DType.F32);
        backend.SliceRows(vidOut, attnCat, 0);
        Tensor txtOut = new Tensor(new TensorShape(lt, c), DType.F32);
        backend.SliceRows(txtOut, attnCat, lv);
        attnCat.Dispose();
        backend.Scale(txtOut, txtOut, 1f / part.WindowCount);
        return (vidOut, txtOut);
    }

    private static Tensor Concat0(IBackend backend, Tensor a, Tensor b)
    {
        Tensor output = new Tensor(new TensorShape(a.Shape[0] + b.Shape[0], a.Shape[1]), DType.F32);
        backend.Concat(output, [a, b], dim: 0);
        return output;
    }

    private static Tensor Gather(IBackend backend, Tensor src, Tensor indices, long rows, int channels)
    {
        Tensor output = new Tensor(new TensorShape(rows, channels), DType.F32);
        backend.RowGather(output, src, indices, (int)rows, channels);
        return output;
    }

    /// <summary>[G·s, heads·hd] rows → [G, heads, s, hd] SDPA layout.</summary>
    private static Tensor PackHeads(IBackend backend, Tensor rows, int g, int s, int heads, int hd)
    {
        Tensor output = new Tensor(new TensorShape([(long)g, heads, s, hd]), DType.F32);
        backend.Permute0213(output, rows, s, heads, hd);
        return output;
    }
}
