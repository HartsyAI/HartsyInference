using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 shape DiT: an image-conditioned flow-match transformer over a VecSet latent
/// (<c>N</c> set tokens × <c>C</c> channels, no spatial grid). Each block applies timestep AdaLN
/// (6-way shift/scale/gate) self-attention, cross-attention to the DINOv2 conditioning tokens, and a GELU
/// MLP; the network predicts the rectified-flow velocity. Reuses <see cref="DiTUtils"/> for the sinusoidal
/// timestep embedding and no-affine LayerNorm, and <see cref="Hunyuan3DAttention"/> for MHA.
/// <para><b>Numerics validation-pending</b> — block wiring, timestep scaling, and QK-norm presence are
/// validation-gated (see the research doc). Structurally green on synthetic weights.</para></summary>
public sealed unsafe class Hunyuan3DDit
{
    private readonly Hunyuan3DConfig _cfg;
    private readonly Hunyuan3DBlock[] _blocks;

    private Tensor? _inProjW, _inProjB;     // C → Width
    private Tensor? _condProjW, _condProjB; // CondDim → Width
    private Tensor? _tMlp1W, _tMlp1B;       // TimestepEmbedDim → Width
    private Tensor? _tMlp2W, _tMlp2B;       // Width → Width
    private Tensor? _finalModW, _finalModB; // Width → 2*Width (shift,scale)
    private Tensor? _outProjW, _outProjB;   // Width → C

    public Hunyuan3DConfig Config => _cfg;

    public Hunyuan3DDit(Hunyuan3DConfig cfg)
    {
        _cfg = cfg;
        _blocks = new Hunyuan3DBlock[cfg.Depth];
        for (int i = 0; i < cfg.Depth; i++) _blocks[i] = new Hunyuan3DBlock(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inProjW = F32(w[$"{p}in_proj.weight"]); _inProjB = F32(w[$"{p}in_proj.bias"]);
        _condProjW = F32(w[$"{p}cond_proj.weight"]); _condProjB = F32(w[$"{p}cond_proj.bias"]);
        _tMlp1W = F32(w[$"{p}t_embed.mlp.0.weight"]); _tMlp1B = F32(w[$"{p}t_embed.mlp.0.bias"]);
        _tMlp2W = F32(w[$"{p}t_embed.mlp.2.weight"]); _tMlp2B = F32(w[$"{p}t_embed.mlp.2.bias"]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{p}blocks.{i}");
        _finalModW = F32(w[$"{p}final_layer.mod.weight"]); _finalModB = F32(w[$"{p}final_layer.mod.bias"]);
        _outProjW = F32(w[$"{p}final_layer.out_proj.weight"]); _outProjB = F32(w[$"{p}final_layer.out_proj.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_inProjW, _inProjB, _condProjW, _condProjB, _tMlp1W, _tMlp1B, _tMlp2W, _tMlp2B, _finalModW, _finalModB, _outProjW, _outProjB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (Hunyuan3DBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts the flow velocity for <paramref name="latent"/> <c>[B,N,C]</c> at timestep
    /// <paramref name="timestep"/> ∈ [0,1], conditioned on DINOv2 tokens <paramref name="cond"/>
    /// <c>[B, Scond, CondDim]</c>. Returns velocity <c>[B,N,C]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor cond, float timestep)
    {
        int b = (int)latent.Shape[0], n = (int)latent.Shape[1];
        int width = _cfg.Width;

        // Token embed: C → Width.
        Tensor x = new(new TensorShape(b, n, width), DType.F32);
        backend.Linear(x, latent, _inProjW!, _inProjB!);

        // Cond project: CondDim → Width.
        int scond = (int)cond.Shape[1];
        Tensor c = new(new TensorShape(b, scond, width), DType.F32);
        backend.Linear(c, cond, _condProjW!, _condProjB!);

        // Timestep: sinusoidal → MLP (SiLU between) → [B, Width].
        Tensor tSin = new(new TensorShape(b, _cfg.TimestepEmbedDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(tSin, timestep, b, _cfg.TimestepEmbedDim);
        Tensor t1 = new(new TensorShape(b, width), DType.F32); backend.Linear(t1, tSin, _tMlp1W!, _tMlp1B!); tSin.Dispose();
        Tensor t1a = new(t1.Shape, DType.F32); backend.Silu(t1a, t1); t1.Dispose();
        Tensor tVec = new(new TensorShape(b, width), DType.F32); backend.Linear(tVec, t1a, _tMlp2W!, _tMlp2B!); t1a.Dispose();

        foreach (Hunyuan3DBlock block in _blocks)
        {
            Tensor next = block.Forward(backend, x, c, tVec);
            x.Dispose();
            x = next;
        }
        c.Dispose();

        // Final AdaLN (2-way) + project Width → C.
        Tensor mod = new(new TensorShape(b, 2 * width), DType.F32);
        Tensor tAct = new(tVec.Shape, DType.F32); backend.Silu(tAct, tVec); tVec.Dispose();
        backend.Linear(mod, tAct, _finalModW!, _finalModB!); tAct.Dispose();

        Tensor normed = new(x.Shape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, b, n, width, 1e-6f);
        x.Dispose();
        ModOps.Modulate(normed, mod, shiftParam: 0, scaleParam: 1, b, n, width);
        mod.Dispose();

        Tensor velocity = new(new TensorShape(b, n, _cfg.LatentChannels), DType.F32);
        backend.Linear(velocity, normed, _outProjW!, _outProjB!);
        normed.Dispose();
        return velocity;
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

/// <summary>One Hunyuan3D DiT block: AdaLN self-attention + cross-attention to cond + GELU MLP.</summary>
internal sealed unsafe class Hunyuan3DBlock
{
    private readonly Hunyuan3DConfig _cfg;
    private Tensor? _adaW, _adaB;             // Width → 6*Width
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;          // self-attn
    private Tensor? _cqW, _cqB, _ckW, _ckB, _cvW, _cvB, _coW, _coB;  // cross-attn
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;                       // mlp

    public Hunyuan3DBlock(Hunyuan3DConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _adaW = Hunyuan3DDit.F32(w[$"{p}.ada_mod.weight"]); _adaB = Hunyuan3DDit.F32(w[$"{p}.ada_mod.bias"]);
        _qW = Hunyuan3DDit.F32(w[$"{p}.self_attn.q.weight"]); _qB = Hunyuan3DDit.F32(w[$"{p}.self_attn.q.bias"]);
        _kW = Hunyuan3DDit.F32(w[$"{p}.self_attn.k.weight"]); _kB = Hunyuan3DDit.F32(w[$"{p}.self_attn.k.bias"]);
        _vW = Hunyuan3DDit.F32(w[$"{p}.self_attn.v.weight"]); _vB = Hunyuan3DDit.F32(w[$"{p}.self_attn.v.bias"]);
        _oW = Hunyuan3DDit.F32(w[$"{p}.self_attn.o.weight"]); _oB = Hunyuan3DDit.F32(w[$"{p}.self_attn.o.bias"]);
        _cqW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.q.weight"]); _cqB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.q.bias"]);
        _ckW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.k.weight"]); _ckB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.k.bias"]);
        _cvW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.v.weight"]); _cvB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.v.bias"]);
        _coW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.o.weight"]); _coB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.o.bias"]);
        _fc1W = Hunyuan3DDit.F32(w[$"{p}.mlp.fc1.weight"]); _fc1B = Hunyuan3DDit.F32(w[$"{p}.mlp.fc1.bias"]);
        _fc2W = Hunyuan3DDit.F32(w[$"{p}.mlp.fc2.weight"]); _fc2B = Hunyuan3DDit.F32(w[$"{p}.mlp.fc2.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_adaW, _adaB, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB,
            _cqW, _cqB, _ckW, _ckB, _cvW, _cvB, _coW, _coB, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor cond, Tensor tVec)
    {
        int b = (int)x.Shape[0], n = (int)x.Shape[1], width = _cfg.Width;

        // AdaLN params: SiLU(tVec) → Linear → [B, 6*Width] = shift_a,scale_a,gate_a,shift_m,scale_m,gate_m.
        Tensor tAct = new(tVec.Shape, DType.F32); backend.Silu(tAct, tVec);
        Tensor mod = new(new TensorShape(b, 6 * width), DType.F32); backend.Linear(mod, tAct, _adaW!, _adaB!); tAct.Dispose();

        // Self-attention with modulated pre-norm + gate.
        Tensor n1 = new(x.Shape, DType.F32); DiTUtils.LayerNormNoAffine(n1, x, b, n, width, 1e-6f);
        ModOps.Modulate(n1, mod, shiftParam: 0, scaleParam: 1, b, n, width);
        Tensor sa = SelfAttn(backend, n1, b, n, width); n1.Dispose();
        Tensor x1 = new(x.Shape, DType.F32); ModOps.GatedResidual(x1, x, sa, mod, gateParam: 2, b, n, width); sa.Dispose();

        // Cross-attention to cond (no gate; ungated residual).
        Tensor n2 = new(x1.Shape, DType.F32); DiTUtils.LayerNormNoAffine(n2, x1, b, n, width, 1e-6f);
        Tensor ca = CrossAttn(backend, n2, cond, b, n, width); n2.Dispose();
        Tensor x2 = new(x1.Shape, DType.F32); backend.Add(x2, x1, ca); x1.Dispose(); ca.Dispose();

        // MLP with modulated pre-norm + gate.
        Tensor n3 = new(x2.Shape, DType.F32); DiTUtils.LayerNormNoAffine(n3, x2, b, n, width, 1e-6f);
        ModOps.Modulate(n3, mod, shiftParam: 3, scaleParam: 4, b, n, width);
        Tensor mlp = Mlp(backend, n3, b, n); n3.Dispose();
        Tensor x3 = new(x2.Shape, DType.F32); ModOps.GatedResidual(x3, x2, mlp, mod, gateParam: 5, b, n, width); x2.Dispose(); mlp.Dispose();
        mod.Dispose();
        return x3;
    }

    private Tensor SelfAttn(IBackend backend, Tensor x, int b, int n, int width)
    {
        Tensor q = new(new TensorShape(b, n, width), DType.F32); backend.Linear(q, x, _qW!, _qB!);
        Tensor k = new(new TensorShape(b, n, width), DType.F32); backend.Linear(k, x, _kW!, _kB!);
        Tensor v = new(new TensorShape(b, n, width), DType.F32); backend.Linear(v, x, _vW!, _vB!);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, k, v, _cfg.NumHeads);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor o = new(new TensorShape(b, n, width), DType.F32); backend.Linear(o, a, _oW!, _oB!); a.Dispose();
        return o;
    }

    private Tensor CrossAttn(IBackend backend, Tensor x, Tensor cond, int b, int n, int width)
    {
        int scond = (int)cond.Shape[1];
        Tensor q = new(new TensorShape(b, n, width), DType.F32); backend.Linear(q, x, _cqW!, _cqB!);
        Tensor k = new(new TensorShape(b, scond, width), DType.F32); backend.Linear(k, cond, _ckW!, _ckB!);
        Tensor v = new(new TensorShape(b, scond, width), DType.F32); backend.Linear(v, cond, _cvW!, _cvB!);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, k, v, _cfg.NumHeads);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor o = new(new TensorShape(b, n, width), DType.F32); backend.Linear(o, a, _coW!, _coB!); a.Dispose();
        return o;
    }

    private Tensor Mlp(IBackend backend, Tensor x, int b, int n)
    {
        Tensor f1 = new(new TensorShape(b, n, _cfg.MlpDim), DType.F32); backend.Linear(f1, x, _fc1W!, _fc1B!);
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(new TensorShape(b, n, _cfg.Width), DType.F32); backend.Linear(f2, act, _fc2W!, _fc2B!); act.Dispose();
        return f2;
    }
}

/// <summary>AdaLN modulation helpers operating on a packed params tensor <c>[B, K*Width]</c>.</summary>
internal static unsafe class ModOps
{
    /// <summary>In-place <c>x = x * (1 + scale) + shift</c>, broadcasting per-channel params over the
    /// sequence. <paramref name="mod"/> is <c>[B, K*Width]</c>; param index selects the K-slice.</summary>
    public static void Modulate(Tensor x, Tensor mod, int shiftParam, int scaleParam, int b, int s, int width)
    {
        float* xp = (float*)x.DataPointer; float* mp = (float*)mod.DataPointer;
        int k = (int)(mod.Shape[1] / width);
        for (int bb = 0; bb < b; bb++)
        {
            float* shift = mp + (long)bb * k * width + (long)shiftParam * width;
            float* scale = mp + (long)bb * k * width + (long)scaleParam * width;
            for (int ss = 0; ss < s; ss++)
            {
                float* row = xp + ((long)bb * s + ss) * width;
                for (int c = 0; c < width; c++) row[c] = row[c] * (1f + scale[c]) + shift[c];
            }
        }
    }

    /// <summary>Writes <c>dst = residual + gate * sublayer</c>, gate broadcast per-channel over the sequence.</summary>
    public static void GatedResidual(Tensor dst, Tensor residual, Tensor sublayer, Tensor mod, int gateParam, int b, int s, int width)
    {
        float* dp = (float*)dst.DataPointer; float* rp = (float*)residual.DataPointer; float* sp = (float*)sublayer.DataPointer; float* mp = (float*)mod.DataPointer;
        int k = (int)(mod.Shape[1] / width);
        for (int bb = 0; bb < b; bb++)
        {
            float* gate = mp + (long)bb * k * width + (long)gateParam * width;
            for (int ss = 0; ss < s; ss++)
            {
                long off = ((long)bb * s + ss) * width;
                for (int c = 0; c < width; c++) dp[off + c] = rp[off + c] + gate[c] * sp[off + c];
            }
        }
    }
}
