using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS stage-1 sparse-structure flow DiT (<c>ss_flow_img_dit_L_16l8</c>): predicts the rectified-flow
/// velocity for a dense <c>[1,8,16³]</c> latent, image-conditioned via cross-attention to DINOv2 tokens. Patchify is
/// identity (patch 1); tokens = 16³ = 4096. 24 <see cref="SsFlowBlock"/> (modulated self-attn + cross-attn + tanh-GELU
/// MLP, adaLN-6, per-head QK-RMSNorm ×√64). Absolute position via a loaded <c>pos_emb</c> buffer. All F32
/// (correctness-first). See <c>docs/Research/TRELLIS_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class SparseStructureFlow
{
    private const int Tokens = 4096, Width = 1024, Heads = 16, HeadDim = 64, InCh = 8;
    private const float NormEps = 1e-6f, OutNormEps = 1e-5f;

    private Tensor? _inW, _inB, _outW, _outB, _posEmb;
    private Tensor? _tIn1W, _tIn1B, _tIn2W, _tIn2B;
    private Tensor? _ones64;
    private readonly SsFlowBlock[] _blocks = new SsFlowBlock[24];

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inW = F(w, "input_layer.weight"); _inB = F(w, "input_layer.bias");
        _outW = F(w, "out_layer.weight"); _outB = F(w, "out_layer.bias");
        _posEmb = F(w, "pos_emb").Reshape(new TensorShape(1, Tokens, Width));
        _tIn1W = F(w, "t_embedder.mlp.0.weight"); _tIn1B = F(w, "t_embedder.mlp.0.bias");
        _tIn2W = F(w, "t_embedder.mlp.2.weight"); _tIn2B = F(w, "t_embedder.mlp.2.bias");
        for (int i = 0; i < 24; i++) _blocks[i] = SsFlowBlock.Load(w, $"blocks.{i}");
        _ones64 = new(new TensorShape(HeadDim), DType.F32);
        float* o = (float*)_ones64.DataPointer; for (int i = 0; i < HeadDim; i++) o[i] = 1f;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _inW, _inB, _outW, _outB, _posEmb, _tIn1W, _tIn1B, _tIn2W, _tIn2B, _ones64 }) if (t is not null) yield return t;
        foreach (SsFlowBlock b in _blocks) foreach (Tensor t in b.Weights()) yield return t;
    }

    /// <summary>Predicts velocity <c>[1,8,16,16,16]</c> for <paramref name="latent"/> at model-timestep
    /// <paramref name="tModel"/> (= 1000·t), conditioned on <paramref name="cond"/> <c>[1, Lc, 1024]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float tModel, Tensor cond)
    {
        // patchify (identity) + reshape [1,8,16³] → [1,8,4096] → transpose → [1,4096,8].
        Tensor flat = latent.Reshape(new TensorShape(1, InCh, Tokens));
        Tensor tok = new(new TensorShape(1, Tokens, InCh), DType.F32); backend.Transpose2D(tok, flat, InCh, Tokens);
        Tensor h = new(new TensorShape(1, Tokens, Width), DType.F32); backend.Linear(h, tok, _inW!, _inB!); tok.Dispose();
        Tensor h2 = new(h.Shape, DType.F32); backend.Add(h2, h, _posEmb!); h.Dispose();

        Tensor vec = TimestepEmbed(backend, tModel);

        Tensor x = h2;
        foreach (SsFlowBlock b in _blocks)
        {
            Tensor nx = b.Forward(backend, x, vec, cond, _ones64!);
            x.Dispose(); x = nx;
        }
        vec.Dispose();

        Tensor normed = new(x.Shape, DType.F32); backend.LayerNormNoAffine(normed, x, OutNormEps); x.Dispose();
        Tensor outTok = new(new TensorShape(1, Tokens, InCh), DType.F32); backend.Linear(outTok, normed, _outW!, _outB!); normed.Dispose();
        // [1,4096,8] → transpose → [1,8,4096] → reshape [1,8,16,16,16].
        Tensor outT = new(new TensorShape(1, InCh, Tokens), DType.F32); backend.Transpose2D(outT, outTok, Tokens, InCh); outTok.Dispose();
        Tensor velocity = new(new TensorShape(new long[] { 1, InCh, 16, 16, 16 }), DType.F32); backend.CopyInto(velocity, outT.Reshape(velocity.Shape)); outT.Dispose();
        return velocity;
    }

    /// <summary>MLPEmbedder(sinusoid(t, 256, cos-first, max_period 1e4)) → Linear256→1024 → SiLU → Linear1024→1024.</summary>
    private Tensor TimestepEmbed(IBackend backend, float tModel)
    {
        const int dim = 256; int half = dim / 2;
        Tensor sin = new(new TensorShape(1, dim), DType.F32);
        float* p = (float*)sin.DataPointer;
        float logMax = MathF.Log(10000f);
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-logMax * i / half), a = tModel * freq;
            p[i] = MathF.Cos(a); p[half + i] = MathF.Sin(a);   // cos first, then sin
        }
        Tensor t1 = new(new TensorShape(1, Width), DType.F32); backend.Linear(t1, sin, _tIn1W!, _tIn1B!); sin.Dispose();
        Tensor t1a = new(t1.Shape, DType.F32); backend.Silu(t1a, t1); t1.Dispose();
        Tensor vec = new(new TensorShape(1, Width), DType.F32); backend.Linear(vec, t1a, _tIn2W!, _tIn2B!); t1a.Dispose();
        return vec;
    }

    internal static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => w[k].DType != DType.F32 ? w[k].CastTo(DType.F32) : w[k];
}
