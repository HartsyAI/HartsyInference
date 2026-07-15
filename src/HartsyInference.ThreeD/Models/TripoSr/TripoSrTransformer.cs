using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.Vision;

namespace HartsyInference.ThreeD.Models.TripoSr;

/// <summary>TripoSR backbone: the learned <c>Triplane1DTokenizer</c> embeddings (<c>[3,C,P,P]</c>) are run
/// through a diffusers <c>Transformer1D</c> (GroupNorm → proj_in → 16× <see cref="TripoSrBlock"/> → proj_out
/// → residual) cross-attending to the DINO image tokens, detokenized to a <c>[3,C,P,P]</c> triplane, then
/// upsampled per-plane by a ConvTranspose2d (k2/s2) into the final <c>[3, TriplaneChannels, 2P, 2P]</c>
/// <see cref="Triplane"/>. Feed-forward — no timestep. Mirrors <c>tsr.system.TSR.forward</c>.</summary>
public sealed unsafe class TripoSrTransformer
{
    private readonly TripoSrConfig _cfg;
    private readonly TripoSrBlock[] _blocks;

    private Tensor? _triplaneEmb;         // tokenizer.embeddings [3, NumChannels, P, P]
    private Tensor? _normW, _normB;       // backbone.norm (GroupNorm) [NumChannels]
    private Tensor? _projInW, _projInB;   // backbone.proj_in NumChannels -> Width
    private Tensor? _projOutW, _projOutB; // backbone.proj_out Width -> NumChannels
    private Tensor? _upW, _upB;           // post_processor.upsample (ConvTranspose2d) [NumChannels, TriplaneChannels, 2, 2]

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
        _triplaneEmb = Hunyuan3DDit.F32(w[$"{p}tokenizer.embeddings"]);
        _normW = Hunyuan3DDit.F32(w[$"{p}backbone.norm.weight"]); _normB = Hunyuan3DDit.F32(w[$"{p}backbone.norm.bias"]);
        _projInW = Hunyuan3DDit.F32(w[$"{p}backbone.proj_in.weight"]); _projInB = Hunyuan3DDit.F32(w[$"{p}backbone.proj_in.bias"]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{p}backbone.transformer_blocks.{i}");
        _projOutW = Hunyuan3DDit.F32(w[$"{p}backbone.proj_out.weight"]); _projOutB = Hunyuan3DDit.F32(w[$"{p}backbone.proj_out.bias"]);
        _upW = Hunyuan3DDit.F32(w[$"{p}post_processor.upsample.weight"]); _upB = Hunyuan3DDit.F32(w[$"{p}post_processor.upsample.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_triplaneEmb, _normW, _normB, _projInW, _projInB, _projOutW, _projOutB, _upW, _upB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (TripoSrBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Produces the upsampled <see cref="Triplane"/> from DINO image tokens <c>[1, S, 768]</c>.</summary>
    public Triplane Forward(IBackend backend, Tensor imageTokens)
    {
        int c = _cfg.NumChannels, ps = _cfg.PlaneSize, n = _cfg.TriplaneTokens, width = _cfg.Width;

        // tokens [1, C, N]: token[ct, (np·P+hp)·P+wp] = emb[np, ct, hp, wp]
        Tensor tokensCN = new(new TensorShape(1, c, n), DType.F32);
        float* tp = (float*)tokensCN.DataPointer;
        float* ep = (float*)_triplaneEmb!.DataPointer;   // [3, C, P, P]
        for (int np = 0; np < 3; np++)
            for (int ct = 0; ct < c; ct++)
                for (int hp = 0; hp < ps; hp++)
                    for (int wp = 0; wp < ps; wp++)
                    {
                        long seqIdx = ((long)np * ps + hp) * ps + wp;
                        tp[(long)ct * n + seqIdx] = ep[(((long)np * c + ct) * ps + hp) * ps + wp];
                    }

        // GroupNorm over [1,C,N,1]. NOTE: the output must be the SAME Tensor object fed to the next op — the
        // CUDA activation cache is keyed by object identity, so writing GroupNorm into `x.Reshape(...)` (a new
        // identity) and then reading the original `x` cache-misses → stale host zeros. Keep gnOut as the
        // canonical [1,C,N,1] tensor and hand it straight to Transpose2D (same c·n memory layout as [1,C,N]).
        Tensor gnIn = tokensCN.Reshape(new TensorShape(1, c, n, 1));
        Tensor gnOut = new(new TensorShape(1, c, n, 1), DType.F32);
        backend.GroupNorm(gnOut, gnIn, _normW!, _normB!, _cfg.NormNumGroups, 1e-6f);

        // permute [1,C,N] -> [1,N,C], proj_in -> [1,N,Width].
        Tensor seqNC = new(new TensorShape(1, n, c), DType.F32);
        backend.Transpose2D(seqNC, gnOut, c, n); gnOut.Dispose();
        Tensor h = new(new TensorShape(1, n, width), DType.F32);
        backend.Linear(h, seqNC, _projInW!, _projInB!); seqNC.Dispose();

        foreach (TripoSrBlock block in _blocks) { Tensor nh = block.Forward(backend, h, imageTokens); h.Dispose(); h = nh; }

        // proj_out -> [1,N,C], permute back -> [1,C,N], + residual.
        Tensor outNC = new(new TensorShape(1, n, c), DType.F32);
        backend.Linear(outNC, h, _projOutW!, _projOutB!); h.Dispose();
        Tensor outCN = new(new TensorShape(1, c, n), DType.F32);
        backend.Transpose2D(outCN, outNC, n, c); outNC.Dispose();
        float* op = (float*)outCN.DataPointer;
        for (long i = 0; i < (long)c * n; i++) op[i] += tp[i];
        tokensCN.Dispose();

        // detokenize [1,C,N] -> [3,C,P,P], then ConvTranspose2d upsample each plane -> [3,Cup,2P,2P].
        int cup = _cfg.TriplaneChannels, res = _cfg.TriplaneResolution;
        Tensor planeIn = new(new TensorShape(3, c, ps, ps), DType.F32);
        float* pi = (float*)planeIn.DataPointer;
        for (int np = 0; np < 3; np++)
            for (int ct = 0; ct < c; ct++)
                for (int hp = 0; hp < ps; hp++)
                    for (int wp = 0; wp < ps; wp++)
                    {
                        long seqIdx = ((long)np * ps + hp) * ps + wp;
                        pi[(((long)np * c + ct) * ps + hp) * ps + wp] = op[(long)ct * n + seqIdx];
                    }
        outCN.Dispose();

        Tensor planeOut = new(new TensorShape(3, cup, res, res), DType.F32);
        backend.ConvTranspose2d(planeOut, planeIn, _upW!, _upB!, 2, 2, 0, 0);
        planeIn.Dispose();

        float[] planes = new float[3L * cup * res * res > int.MaxValue ? throw new InvalidOperationException() : 3 * cup * res * res];
        new ReadOnlySpan<float>((float*)planeOut.DataPointer, planes.Length).CopyTo(planes);
        planeOut.Dispose();
        return new Triplane { Features = planes, Channels = cup, Height = res, Width = res };
    }
}

/// <summary>One diffusers <c>BasicTransformerBlock</c>: pre-norm (affine LayerNorm) self-attention + cross-
/// attention to image tokens + GEGLU feed-forward, each with a residual. Biasless q/k/v; biased output proj.</summary>
internal sealed unsafe class TripoSrBlock
{
    private readonly TripoSrConfig _cfg;
    private Tensor? _n1W, _n1B, _qW, _kW, _vW, _oW, _oB;             // self-attn (attn1)
    private Tensor? _n2W, _n2B, _cqW, _ckW, _cvW, _coW, _coB;        // cross-attn (attn2)
    private Tensor? _n3W, _n3B, _ffProjW, _ffProjB, _ffOutW, _ffOutB; // GEGLU ff

    public TripoSrBlock(TripoSrConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _n1W = Hunyuan3DDit.F32(w[$"{p}.norm1.weight"]); _n1B = Hunyuan3DDit.F32(w[$"{p}.norm1.bias"]);
        _qW = Hunyuan3DDit.F32(w[$"{p}.attn1.to_q.weight"]);
        _kW = Hunyuan3DDit.F32(w[$"{p}.attn1.to_k.weight"]);
        _vW = Hunyuan3DDit.F32(w[$"{p}.attn1.to_v.weight"]);
        _oW = Hunyuan3DDit.F32(w[$"{p}.attn1.to_out.0.weight"]); _oB = Hunyuan3DDit.F32(w[$"{p}.attn1.to_out.0.bias"]);
        _n2W = Hunyuan3DDit.F32(w[$"{p}.norm2.weight"]); _n2B = Hunyuan3DDit.F32(w[$"{p}.norm2.bias"]);
        _cqW = Hunyuan3DDit.F32(w[$"{p}.attn2.to_q.weight"]);
        _ckW = Hunyuan3DDit.F32(w[$"{p}.attn2.to_k.weight"]);
        _cvW = Hunyuan3DDit.F32(w[$"{p}.attn2.to_v.weight"]);
        _coW = Hunyuan3DDit.F32(w[$"{p}.attn2.to_out.0.weight"]); _coB = Hunyuan3DDit.F32(w[$"{p}.attn2.to_out.0.bias"]);
        _n3W = Hunyuan3DDit.F32(w[$"{p}.norm3.weight"]); _n3B = Hunyuan3DDit.F32(w[$"{p}.norm3.bias"]);
        _ffProjW = Hunyuan3DDit.F32(w[$"{p}.ff.net.0.proj.weight"]); _ffProjB = Hunyuan3DDit.F32(w[$"{p}.ff.net.0.proj.bias"]);
        _ffOutW = Hunyuan3DDit.F32(w[$"{p}.ff.net.2.weight"]); _ffOutB = Hunyuan3DDit.F32(w[$"{p}.ff.net.2.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_n1W, _n1B, _qW, _kW, _vW, _oW, _oB, _n2W, _n2B, _cqW, _ckW, _cvW, _coW, _coB, _n3W, _n3B, _ffProjW, _ffProjB, _ffOutW, _ffOutB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor h, Tensor img)
    {
        int n = (int)h.Shape[1], width = _cfg.Width, s = (int)img.Shape[1];

        Tensor n1 = new(h.Shape, DType.F32); backend.LayerNorm(n1, h, _n1W!, _n1B!, 1e-5f);
        Tensor sa = SelfAttn(backend, n1, n); n1.Dispose();
        Tensor x1 = new(h.Shape, DType.F32); backend.Add(x1, h, sa); sa.Dispose();

        Tensor n2 = new(x1.Shape, DType.F32); backend.LayerNorm(n2, x1, _n2W!, _n2B!, 1e-5f);
        Tensor ca = CrossAttn(backend, n2, img, n, s); n2.Dispose();
        Tensor x2 = new(x1.Shape, DType.F32); backend.Add(x2, x1, ca); x1.Dispose(); ca.Dispose();

        Tensor n3 = new(x2.Shape, DType.F32); backend.LayerNorm(n3, x2, _n3W!, _n3B!, 1e-5f);
        Tensor ff = GegluFeedForward(backend, n3, n); n3.Dispose();
        Tensor x3 = new(x2.Shape, DType.F32); backend.Add(x3, x2, ff); x2.Dispose(); ff.Dispose();
        return x3;
    }

    private Tensor SelfAttn(IBackend backend, Tensor src, int n)
    {
        int width = _cfg.Width;
        Tensor q = new(new TensorShape(1, n, width), DType.F32); backend.Linear(q, src, _qW!, null);
        Tensor k = new(new TensorShape(1, n, width), DType.F32); backend.Linear(k, src, _kW!, null);
        Tensor v = new(new TensorShape(1, n, width), DType.F32); backend.Linear(v, src, _vW!, null);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, k, v, _cfg.NumHeads); q.Dispose(); k.Dispose(); v.Dispose();
        Tensor o = new(new TensorShape(1, n, width), DType.F32); backend.Linear(o, a, _oW!, _oB!); a.Dispose();
        return o;
    }

    private Tensor CrossAttn(IBackend backend, Tensor src, Tensor img, int n, int s)
    {
        int width = _cfg.Width;
        Tensor q = new(new TensorShape(1, n, width), DType.F32); backend.Linear(q, src, _cqW!, null);
        Tensor k = new(new TensorShape(1, s, width), DType.F32); backend.Linear(k, img, _ckW!, null);
        Tensor v = new(new TensorShape(1, s, width), DType.F32); backend.Linear(v, img, _cvW!, null);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, k, v, _cfg.NumHeads); q.Dispose(); k.Dispose(); v.Dispose();
        Tensor o = new(new TensorShape(1, n, width), DType.F32); backend.Linear(o, a, _coW!, _coB!); a.Dispose();
        return o;
    }

    private Tensor GegluFeedForward(IBackend backend, Tensor src, int n)
    {
        int width = _cfg.Width, inner = (int)_ffProjW!.Shape[0] / 2;   // proj emits 2·inner
        Tensor proj = new(new TensorShape(1, n, 2 * inner), DType.F32); backend.Linear(proj, src, _ffProjW!, _ffProjB!);
        Tensor gated = new(new TensorShape(1, n, inner), DType.F32);
        // out = first_half * gelu_erf(second_half), split along last dim — fused device pass (no host round-trip).
        backend.GegluErf(gated, proj, n, inner);
        proj.Dispose();
        Tensor f2 = new(new TensorShape(1, n, width), DType.F32); backend.Linear(f2, gated, _ffOutW!, _ffOutB!); gated.Dispose();
        return f2;
    }
}
