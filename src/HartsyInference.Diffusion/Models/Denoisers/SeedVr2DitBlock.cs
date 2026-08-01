using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One SeedVR2 NaDiT block: windowed joint attention + SwiGLU MLP, both wrapped in AdaSingle
/// modulation (emb slice + per-block learned vectors, additive). Blocks below mm_layers hold separate
/// vid/txt weights (<c>.vid./.txt.</c>); above, one shared set (<c>.all.</c>). The LAST block computes the
/// text branch through attention (joint attention needs it) but skips text mlp/ada — <c>vid_only</c>.
/// <para>Attention: per-window sequences <c>[window vid tokens ; full text]</c>, text REPLICATED into every
/// window and MEAN-pooled across windows afterwards; RMS qk-norm per 128-dim head; window-local
/// <see cref="SeedVr2Rope"/>; same-shape windows batched into one SDPA call. Host gather/scatter with
/// backend GEMM/SDPA — bring-up layout, residency/graph discipline lands in the pipeline pass.</para></summary>
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
    private float[] _normQVid = null!, _normKVid = null!;
    private float[]? _normQTxt, _normKTxt;
    private float[][] _adaVid = null!;   // [attnShift, attnScale, attnGate, mlpShift, mlpScale, mlpGate]
    private float[][]? _adaTxt;

    /// <summary>Creates block <paramref name="index"/>; even layers use the regular window partition, odd
    /// layers the shifted one (config <c>window_method</c> alternation).</summary>
    public SeedVr2DitBlock(SeedVr2Config config, int index)
    {
        _config = config;
        _shared = index >= config.MmLayers;
        _isLast = index == config.NumLayers - 1;
        _shifted = index % 2 == 1;
    }

    /// <summary>Binds <c>blocks.{i}.*</c> checkpoint keys (verbatim names via the near-identity converter).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, int index)
    {
        string p = $"blocks.{index}";
        string b = _shared ? "all" : "vid";
        _qkvVidW = weights[$"{p}.attn.proj_qkv.{b}.weight"];
        _projVidW = weights[$"{p}.attn.proj_out.{b}.weight"];
        _projVidB = weights[$"{p}.attn.proj_out.{b}.bias"];
        _normQVid = ToHost(weights[$"{p}.attn.norm_q.{b}.weight"]);
        _normKVid = ToHost(weights[$"{p}.attn.norm_k.{b}.weight"]);
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
            _normQTxt = ToHost(weights[$"{p}.attn.norm_q.txt.weight"]);
            _normKTxt = ToHost(weights[$"{p}.attn.norm_k.txt.weight"]);
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

    // Host-math vectors must be F32 regardless of checkpoint dtype (the 7B ships F16-converted for disk).
    private static float[] ToHost(Tensor t)
    {
        if (t.DType == DType.F32)
            return t.AsSpan<float>().ToArray();
        Tensor cast = t.CastTo(DType.F32);
        float[] result = cast.AsSpan<float>().ToArray();
        cast.Dispose();
        return result;
    }

    /// <summary>Runs the block in place on host-resident token tensors <paramref name="vid"/> [Lv,C] and
    /// <paramref name="txt"/> [Lt,C]. <paramref name="emb"/> is the 15360-dim AdaSingle vector, sliced
    /// (d l g)-wise here. Grid is the patchified token grid for the window partition.</summary>
    public void Forward(IBackend backend, Tensor vid, Tensor txt, int gridT, int gridH, int gridW, float[] emb)
    {
        int c = _config.VidDim;
        int lv = (int)vid.Shape[0], lt = (int)txt.Shape[0];
        (int nT, int nH, int nW) = _config.WindowCounts;
        SeedVr2WindowSlice[] windows = _shifted
            ? SeedVr2Windowing.MakeShiftedWindows(gridT, gridH, gridW, nT, nH, nW)
            : SeedVr2Windowing.MakeWindows(gridT, gridH, gridW, nT, nH, nW);

        // ---- attention ----
        // Last layer: ada is vid_only in the reference — text enters attention PLAIN-NORMED (no ada-in)
        // and its attention output is residual-added UNGATED. Missing this pollutes vid K/V (caught by
        // block3 parity: txt relL2 0.64, vid 1e-2).
        Tensor vidMod = ModulateIn(backend, vid, emb, _adaVid, layer: 0);
        Tensor txtMod = _isLast
            ? RmsNoAffine(txt)
            : ModulateIn(backend, txt, emb, _adaTxt ?? _adaVid, layer: 0);
        Tensor vidQkv = Linear(backend, vidMod, _qkvVidW, null);
        Tensor txtQkv = Linear(backend, txtMod, _qkvTxtW ?? _qkvVidW, null);
        vidMod.Dispose();
        txtMod.Dispose();
        backend.Sync();

        (Tensor vidAttn, Tensor txtAttn) = WindowedAttention(backend, vidQkv, txtQkv, windows, gridH, gridW, lv, lt);
        vidQkv.Dispose();
        txtQkv.Dispose();

        Tensor vidProj = Linear(backend, vidAttn, _projVidW, _projVidB);
        Tensor txtProj = Linear(backend, txtAttn, _projTxtW ?? _projVidW, _projTxtB ?? _projVidB);
        vidAttn.Dispose();
        txtAttn.Dispose();
        backend.Sync();
        GateAndAdd(vid, vidProj, emb, _adaVid, layer: 0);
        if (_isLast)
            AddInPlace(txt, txtProj);
        else
            GateAndAdd(txt, txtProj, emb, _adaTxt ?? _adaVid, layer: 0);
        vidProj.Dispose();
        txtProj.Dispose();

        // ---- mlp (video always; text skipped on the last layer) ----
        MlpBranch(backend, vid, emb, _adaVid, _mlpVidGateW, _mlpVidInW, _mlpVidInB, _mlpVidOutW, _mlpVidOutB);
        if (!_isLast)
        {
            MlpBranch(backend, txt, emb, _adaTxt ?? _adaVid,
                _mlpTxtGateW ?? _mlpVidGateW, _mlpTxtInW ?? _mlpVidInW, _mlpTxtInB ?? _mlpVidInB,
                _mlpTxtOutW ?? _mlpVidOutW, _mlpTxtOutB ?? _mlpVidOutB);
        }
        else
        {
            // Reference quirk: vid_only MMModules return txt UNCHANGED, so the "skipped" mlp branch
            // residual-adds txt to itself → final txt = 2×post-attention txt. txt is discarded after this
            // layer; doubled here only so parity traces match the reference exactly (relL2 0.5 otherwise).
            AddInPlace(txt, txt);
        }
    }

    private void MlpBranch(IBackend backend, Tensor tokens, float[] emb, float[][] ada,
        Tensor? gateW, Tensor inW, Tensor? inB, Tensor outW, Tensor? outB)
    {
        Tensor modded = ModulateIn(backend, tokens, emb, ada, layer: 1);
        Tensor hidden;
        if (gateW is not null)
        {
            // SwiGLU (3B): proj_out(silu(gate(x)) · in(x)), bias-free.
            Tensor gate = Linear(backend, modded, gateW, null);
            Tensor lin = Linear(backend, modded, inW, null);
            backend.Sync();
            Span<float> g = gate.AsSpan<float>();
            ReadOnlySpan<float> l = lin.AsSpan<float>();
            for (int i = 0; i < g.Length; i++)
            {
                float x = g[i];
                g[i] = x / (1f + MathF.Exp(-x)) * l[i];
            }
            lin.Dispose();
            hidden = gate;
        }
        else
        {
            // Plain MLP (7B): proj_out(gelu_tanh(proj_in(x))), biased.
            hidden = Linear(backend, modded, inW, inB);
            backend.Sync();
            Span<float> h = hidden.AsSpan<float>();
            for (int i = 0; i < h.Length; i++)
            {
                float x = h[i];
                float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
                h[i] = 0.5f * x * (1f + MathF.Tanh(inner));
            }
        }
        modded.Dispose();
        Tensor output = Linear(backend, hidden, outW, outB);
        hidden.Dispose();
        backend.Sync();
        GateAndAdd(tokens, output, emb, ada, layer: 1);
        output.Dispose();
    }

    /// <summary>RMS-norm (no affine) then AdaSingle "in": <c>x̂·(embScale+adaScale) + (embShift+adaShift)</c>.
    /// emb layout (d l g): element for channel d, layer l, component g is emb[d·6 + l·3 + g].</summary>
    private Tensor ModulateIn(IBackend backend, Tensor tokens, float[] emb, float[][] ada, int layer)
    {
        int c = _config.VidDim;
        int rows = (int)tokens.Shape[0];
        Tensor output = new Tensor(tokens.Shape, DType.F32);
        ReadOnlySpan<float> src = tokens.AsSpan<float>();
        Span<float> dst = output.AsSpan<float>();
        float[] shift = ada[layer * 3 + 0], scale = ada[layer * 3 + 1];
        const float eps = 1e-5f;
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = src.Slice(r * c, c);
            double ss = 0;
            for (int i = 0; i < c; i++)
                ss += (double)row[i] * row[i];
            float inv = (float)(1.0 / Math.Sqrt(ss / c + eps));
            Span<float> outRow = dst.Slice(r * c, c);
            for (int i = 0; i < c; i++)
                outRow[i] = row[i] * inv * (emb[i * 6 + layer * 3 + 1] + scale[i])
                    + emb[i * 6 + layer * 3 + 0] + shift[i];
        }
        return output;
    }

    /// <summary>AdaSingle "out" + residual: <c>tokens += branchOut·(embGate+adaGate)</c>.</summary>
    private void GateAndAdd(Tensor tokens, Tensor branchOut, float[] emb, float[][] ada, int layer)
    {
        int c = _config.VidDim;
        Span<float> acc = tokens.AsSpan<float>();
        ReadOnlySpan<float> add = branchOut.AsSpan<float>();
        float[] gate = ada[layer * 3 + 2];
        for (int i = 0; i < acc.Length; i++)
        {
            int ch = i % c;
            acc[i] += add[i] * (emb[ch * 6 + layer * 3 + 2] + gate[ch]);
        }
    }

    private Tensor RmsNoAffine(Tensor tokens)
    {
        int c = _config.VidDim;
        int rows = (int)tokens.Shape[0];
        Tensor output = new Tensor(tokens.Shape, DType.F32);
        ReadOnlySpan<float> src = tokens.AsSpan<float>();
        Span<float> dst = output.AsSpan<float>();
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = src.Slice(r * c, c);
            double ss = 0;
            for (int i = 0; i < c; i++)
                ss += (double)row[i] * row[i];
            float inv = (float)(1.0 / Math.Sqrt(ss / c + 1e-5));
            Span<float> outRow = dst.Slice(r * c, c);
            for (int i = 0; i < c; i++)
                outRow[i] = row[i] * inv;
        }
        return output;
    }

    private static void AddInPlace(Tensor tokens, Tensor add)
    {
        Span<float> acc = tokens.AsSpan<float>();
        ReadOnlySpan<float> a = add.AsSpan<float>();
        for (int i = 0; i < acc.Length; i++)
            acc[i] += a[i];
    }

    private static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias)
    {
        Tensor output = new Tensor(new TensorShape(input.Shape[0], weight.Shape[0]), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    /// <summary>GEMM weights for backend preload/free (host-copied ada/norm vectors excluded — never uploaded).</summary>
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
    }

    private (Tensor Vid, Tensor Txt) WindowedAttention(IBackend backend, Tensor vidQkv, Tensor txtQkv,
        SeedVr2WindowSlice[] windows, int gridH, int gridW, int lv, int lt)
    {
        int heads = _config.Heads, hd = _config.HeadDim, c = _config.VidDim;
        float[] normQTxt = _normQTxt ?? _normQVid, normKTxt = _normKTxt ?? _normKVid;
        ReadOnlySpan<float> vq = vidQkv.AsSpan<float>();
        ReadOnlySpan<float> tq = txtQkv.AsSpan<float>();

        Tensor vidOut = new Tensor(new TensorShape(lv, c), DType.F32);
        Tensor txtOut = new Tensor(new TensorShape(lt, c), DType.F32);
        txtOut.AsSpan<float>().Clear();

        // Pre-normalized+roped text q/k and raw v, reused for every window (rope restarts per window and
        // text positions are 0..lt−1 in every copy, so one preparation serves all windows).
        float[] txtQ = new float[lt * heads * hd], txtK = new float[lt * heads * hd], txtV = new float[lt * heads * hd];
        ExtractQkv(tq, txtQ, txtK, txtV, lt, heads, hd, normQTxt, normKTxt);

        foreach (IGrouping<(int F, int R, int Cc), SeedVr2WindowSlice> group in
            windows.GroupBy(s => (s.Frames, s.Rows, s.Cols)))
        {
            SeedVr2WindowSlice[] grp = [.. group];
            int winTokens = group.Key.F * group.Key.R * group.Key.Cc;
            int s = winTokens + lt;
            (float[] cos, float[] sin) = SeedVr2Rope.BuildTables(group.Key.F, group.Key.R, group.Key.Cc, lt);
            (float[] txtQr, float[] txtKr) = RopeText(txtQ, txtK, cos, sin, winTokens, lt, heads, hd);

            Tensor q = new Tensor(new TensorShape([(long)grp.Length, heads, s, hd]), DType.F32);
            Tensor k = new Tensor(new TensorShape([(long)grp.Length, heads, s, hd]), DType.F32);
            Tensor v = new Tensor(new TensorShape([(long)grp.Length, heads, s, hd]), DType.F32);
            Span<float> qs = q.AsSpan<float>(), ks = k.AsSpan<float>(), vs = v.AsSpan<float>();

            for (int g = 0; g < grp.Length; g++)
            {
                int row = 0;
                foreach (int tokenIdx in TokensOf(grp[g], gridH, gridW))
                {
                    PackToken(vq, tokenIdx, qs, ks, vs, g, row, s, heads, hd, _normQVid, _normKVid,
                        cos, sin, row, winTokens + lt);
                    row++;
                }
                for (int p = 0; p < lt; p++, row++)
                    PackText(txtQr, txtKr, txtV, p, qs, ks, vs, g, row, s, heads, hd);
            }

            Tensor attn = new Tensor(q.Shape, DType.F32);
            backend.ScaledDotProductAttention(attn, q, k, v, null, 1f / MathF.Sqrt(hd));
            backend.Sync();
            q.Dispose(); k.Dispose(); v.Dispose();

            ReadOnlySpan<float> asp = attn.AsSpan<float>();
            Span<float> vo = vidOut.AsSpan<float>();
            Span<float> to = txtOut.AsSpan<float>();
            for (int g = 0; g < grp.Length; g++)
            {
                int row = 0;
                foreach (int tokenIdx in TokensOf(grp[g], gridH, gridW))
                {
                    for (int h = 0; h < heads; h++)
                    {
                        int src = ((g * heads + h) * s + row) * hd;
                        asp.Slice(src, hd).CopyTo(vo.Slice(tokenIdx * c + h * hd, hd));
                    }
                    row++;
                }
                for (int p = 0; p < lt; p++, row++)
                    for (int h = 0; h < heads; h++)
                    {
                        int src = ((g * heads + h) * s + row) * hd;
                        Span<float> dst = to.Slice(p * c + h * hd, hd);
                        ReadOnlySpan<float> add = asp.Slice(src, hd);
                        for (int i = 0; i < hd; i++)
                            dst[i] += add[i];
                    }
            }
            attn.Dispose();
        }

        // Mean-pool the text accumulation over ALL windows (reference unconcat_coalesce .mean).
        float invWindows = 1f / windows.Length;
        Span<float> tspan = txtOut.AsSpan<float>();
        for (int i = 0; i < tspan.Length; i++)
            tspan[i] *= invWindows;
        return (vidOut, txtOut);
    }

    private static IEnumerable<int> TokensOf(SeedVr2WindowSlice slice, int gridH, int gridW)
    {
        for (int t = slice.T0; t < slice.T1; t++)
        for (int h = slice.H0; h < slice.H1; h++)
        for (int w = slice.W0; w < slice.W1; w++)
            yield return (t * gridH + h) * gridW + w;
    }

    private void ExtractQkv(ReadOnlySpan<float> qkv, float[] q, float[] k, float[] v,
        int rows, int heads, int hd, float[] normQ, float[] normK)
    {
        int c = heads * hd;
        for (int r = 0; r < rows; r++)
        for (int h = 0; h < heads; h++)
        {
            int src = r * 3 * c + h * hd;
            int dst = (r * heads + h) * hd;
            RmsHead(qkv.Slice(src, hd), q.AsSpan(dst, hd), normQ);
            RmsHead(qkv.Slice(src + c, hd), k.AsSpan(dst, hd), normK);
            qkv.Slice(src + 2 * c, hd).CopyTo(v.AsSpan(dst, hd));
        }
    }

    private static void RmsHead(ReadOnlySpan<float> src, Span<float> dst, float[] weight)
    {
        double ss = 0;
        for (int i = 0; i < src.Length; i++)
            ss += (double)src[i] * src[i];
        float inv = (float)(1.0 / Math.Sqrt(ss / src.Length + 1e-5));
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] * inv * weight[i];
    }

    private (float[] Q, float[] K) RopeText(float[] txtQ, float[] txtK, float[] cos, float[] sin,
        int vidTokens, int lt, int heads, int hd)
    {
        float[] q = (float[])txtQ.Clone();
        float[] k = (float[])txtK.Clone();
        for (int p = 0; p < lt; p++)
        {
            int tbl = (vidTokens + p) * SeedVr2Rope.RotDim;
            for (int h = 0; h < heads; h++)
            {
                int b = (p * heads + h) * hd;
                for (int i = 0; i < SeedVr2Rope.RotDim; i += 2)
                {
                    float cc = cos[tbl + i], ss = sin[tbl + i];
                    (q[b + i], q[b + i + 1]) = (q[b + i] * cc - q[b + i + 1] * ss, q[b + i + 1] * cc + q[b + i] * ss);
                    (k[b + i], k[b + i + 1]) = (k[b + i] * cc - k[b + i + 1] * ss, k[b + i + 1] * cc + k[b + i] * ss);
                }
            }
        }
        return (q, k);
    }

    private void PackToken(ReadOnlySpan<float> vidQkv, int tokenIdx, Span<float> q, Span<float> k, Span<float> v,
        int g, int row, int s, int heads, int hd, float[] normQ, float[] normK, float[] cos, float[] sin,
        int ropeRow, int tableRows)
    {
        int c = heads * hd;
        Span<float> qh = stackalloc float[hd];
        Span<float> kh = stackalloc float[hd];
        int tbl = ropeRow * SeedVr2Rope.RotDim;
        for (int h = 0; h < heads; h++)
        {
            int src = tokenIdx * 3 * c + h * hd;
            RmsHead(vidQkv.Slice(src, hd), qh, normQ);
            RmsHead(vidQkv.Slice(src + c, hd), kh, normK);
            for (int i = 0; i < SeedVr2Rope.RotDim; i += 2)
            {
                float cc = cos[tbl + i], ss = sin[tbl + i];
                (qh[i], qh[i + 1]) = (qh[i] * cc - qh[i + 1] * ss, qh[i + 1] * cc + qh[i] * ss);
                (kh[i], kh[i + 1]) = (kh[i] * cc - kh[i + 1] * ss, kh[i + 1] * cc + kh[i] * ss);
            }
            int dst = ((g * heads + h) * s + row) * hd;
            qh.CopyTo(q.Slice(dst, hd));
            kh.CopyTo(k.Slice(dst, hd));
            vidQkv.Slice(src + 2 * c, hd).CopyTo(v.Slice(dst, hd));
        }
    }

    private static void PackText(float[] txtQ, float[] txtK, float[] txtV, int p,
        Span<float> q, Span<float> k, Span<float> v, int g, int row, int s, int heads, int hd)
    {
        for (int h = 0; h < heads; h++)
        {
            int src = (p * heads + h) * hd;
            int dst = ((g * heads + h) * s + row) * hd;
            txtQ.AsSpan(src, hd).CopyTo(q.Slice(dst, hd));
            txtK.AsSpan(src, hd).CopyTo(k.Slice(dst, hd));
            txtV.AsSpan(src, hd).CopyTo(v.Slice(dst, hd));
        }
    }
}
