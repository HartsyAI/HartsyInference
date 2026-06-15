using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Models.TripoSr;

/// <summary>TripoSR image→triplane transformer (LRM): learned triplane position tokens self-attend and
/// cross-attend to the DINO image tokens through a stack of blocks, then project to triplane channels and
/// reshape into a <see cref="Triplane"/>. Feed-forward — no timestep/diffusion. Reuses
/// <see cref="Hunyuan3DAttention"/> for MHA and <see cref="DiTUtils.LayerNormNoAffine"/>.
/// <para><b>Numerics validation-pending</b> — block wiring, norm affine-ness, and the triplane token layout
/// are validation-gated. Structurally green on synthetic weights.</para></summary>
public sealed unsafe class TripoSrTransformer
{
    private readonly TripoSrConfig _cfg;
    private readonly TripoSrBlock[] _blocks;

    private Tensor? _triplanePos;     // [1, TriplaneTokens, Width] learned init tokens
    private Tensor? _imgProjW, _imgProjB; // ImageTokenDim → Width
    private Tensor? _outW, _outB;     // Width → TriplaneChannels

    public TripoSrConfig Config => _cfg;

    public TripoSrTransformer(TripoSrConfig cfg)
    {
        _cfg = cfg;
        _blocks = new TripoSrBlock[cfg.Depth];
        for (int i = 0; i < cfg.Depth; i++) _blocks[i] = new TripoSrBlock(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _triplanePos = Hunyuan3DDit.F32(w[$"{p}triplane_pos"]);
        _imgProjW = Hunyuan3DDit.F32(w[$"{p}image_proj.weight"]); _imgProjB = Hunyuan3DDit.F32(w[$"{p}image_proj.bias"]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{p}blocks.{i}");
        _outW = Hunyuan3DDit.F32(w[$"{p}out_proj.weight"]); _outB = Hunyuan3DDit.F32(w[$"{p}out_proj.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_triplanePos, _imgProjW, _imgProjB, _outW, _outB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (TripoSrBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Produces a <see cref="Triplane"/> from DINO image tokens <c>[1, S, ImageTokenDim]</c>.</summary>
    public Triplane Forward(IBackend backend, Tensor imageTokens)
    {
        int width = _cfg.Width, t = _cfg.TriplaneTokens, c = _cfg.TriplaneChannels, res = _cfg.TriplaneResolution;

        // Tokens start from the learned triplane position embedding (copy so we don't mutate the weight).
        Tensor tokens = new(new TensorShape(1, t, width), DType.F32);
        new ReadOnlySpan<float>((float*)_triplanePos!.DataPointer, t * width)
            .CopyTo(new Span<float>((float*)tokens.DataPointer, t * width));

        // Project image tokens → width.
        int s = (int)imageTokens.Shape[1];
        Tensor img = new(new TensorShape(1, s, width), DType.F32);
        backend.Linear(img, imageTokens, _imgProjW!, _imgProjB!);

        foreach (TripoSrBlock block in _blocks) { Tensor n = block.Forward(backend, tokens, img); tokens.Dispose(); tokens = n; }
        img.Dispose();

        Tensor normed = new(tokens.Shape, DType.F32);
        DiTUtils.LayerNormNoAffine(normed, tokens, 1, t, width, 1e-6f);
        tokens.Dispose();

        Tensor feat = new(new TensorShape(1, t, c), DType.F32);
        backend.Linear(feat, normed, _outW!, _outB!);
        normed.Dispose();

        // Rearrange [1, T, C] (token-major, channel-last) → [3, C, res, res] (plane/channel-major).
        float[] planes = new float[3 * c * res * res];
        float* fp = (float*)feat.DataPointer;
        for (int pl = 0; pl < 3; pl++)
            for (int r = 0; r < res; r++)
                for (int col = 0; col < res; col++)
                {
                    long token = ((long)pl * res + r) * res + col;
                    for (int ch = 0; ch < c; ch++)
                        planes[(((long)pl * c + ch) * res + r) * res + col] = fp[token * c + ch];
                }
        feat.Dispose();

        return new Triplane { Features = planes, Channels = c, Height = res, Width = res };
    }
}

/// <summary>One TripoSR transformer block: self-attention (triplane tokens) + cross-attention to image
/// tokens + GELU MLP, pre-norm with no-affine LayerNorm and residuals.</summary>
internal sealed unsafe class TripoSrBlock
{
    private readonly TripoSrConfig _cfg;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;          // self-attn
    private Tensor? _cqW, _cqB, _ckW, _ckB, _cvW, _cvB, _coW, _coB;  // cross-attn
    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B;

    public TripoSrBlock(TripoSrConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
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
        Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _cqW, _cqB, _ckW, _ckB, _cvW, _cvB, _coW, _coB, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor tokens, Tensor img)
    {
        int n = (int)tokens.Shape[1], width = _cfg.Width, s = (int)img.Shape[1];

        Tensor n1 = new(tokens.Shape, DType.F32); DiTUtils.LayerNormNoAffine(n1, tokens, 1, n, width, 1e-6f);
        Tensor sa = Attn(backend, n1, n1, n, n, _qW!, _qB!, _kW!, _kB!, _vW!, _vB!, _oW!, _oB!); n1.Dispose();
        Tensor x1 = new(tokens.Shape, DType.F32); backend.Add(x1, tokens, sa); sa.Dispose();

        Tensor n2 = new(x1.Shape, DType.F32); DiTUtils.LayerNormNoAffine(n2, x1, 1, n, width, 1e-6f);
        Tensor ca = Attn(backend, n2, img, n, s, _cqW!, _cqB!, _ckW!, _ckB!, _cvW!, _cvB!, _coW!, _coB!); n2.Dispose();
        Tensor x2 = new(x1.Shape, DType.F32); backend.Add(x2, x1, ca); x1.Dispose(); ca.Dispose();

        Tensor n3 = new(x2.Shape, DType.F32); DiTUtils.LayerNormNoAffine(n3, x2, 1, n, width, 1e-6f);
        Tensor f1 = new(new TensorShape(1, n, _cfg.MlpDim), DType.F32); backend.Linear(f1, n3, _fc1W!, _fc1B!); n3.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(new TensorShape(1, n, width), DType.F32); backend.Linear(f2, act, _fc2W!, _fc2B!); act.Dispose();
        Tensor x3 = new(x2.Shape, DType.F32); backend.Add(x3, x2, f2); x2.Dispose(); f2.Dispose();
        return x3;
    }

    private Tensor Attn(IBackend backend, Tensor qSrc, Tensor kvSrc, int sq, int sk,
        Tensor qW, Tensor qB, Tensor kW, Tensor kB, Tensor vW, Tensor vB, Tensor oW, Tensor oB)
    {
        int width = _cfg.Width;
        Tensor q = new(new TensorShape(1, sq, width), DType.F32); backend.Linear(q, qSrc, qW, qB);
        Tensor k = new(new TensorShape(1, sk, width), DType.F32); backend.Linear(k, kvSrc, kW, kB);
        Tensor v = new(new TensorShape(1, sk, width), DType.F32); backend.Linear(v, kvSrc, vW, vB);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, k, v, _cfg.NumHeads); q.Dispose(); k.Dispose(); v.Dispose();
        Tensor o = new(new TensorShape(1, sq, width), DType.F32); backend.Linear(o, a, oW, oB); a.Dispose();
        return o;
    }
}
