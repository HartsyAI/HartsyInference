using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using static HartsyInference.ThreeD.Models.CoLa3D.CoLa3DOps;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>One CoLa3D block (rope path): query self-attention, queries reading points, points reading the updated
/// queries, then a tanh-GELU feed-forward on each stream. Borrows its weights and owns only its head-norm scale.</summary>
internal sealed class CoLa3DBlock : IDisposable
{
    private readonly int _heads, _headDim, _width;
    private readonly float _scale;
    private readonly Tensor _ones;
    private readonly Affine _qSelfNorm, _qReadNorm, _pReadNorm, _qFfnNorm, _pWriteNorm, _pFfnNorm;
    private readonly Dense _qSelfQkv, _qSelfOut, _qReadQuery, _pReadKv, _qReadOut, _qFfnUp, _qFfnDown;
    private readonly Dense _qWriteKv, _pWriteQuery, _pWriteOut, _pFfnUp, _pFfnDown;
    private readonly Tensor _pWriteQkNorm, _qWriteKNorm;
    private int _disposed;

    /// <summary>Binds layer <paramref name="layer"/> of the named weights; the tensors stay owned by <paramref name="w"/>.</summary>
    internal CoLa3DBlock(IReadOnlyDictionary<string, Tensor> w, int layer, CoLa3DConfig config)
    {
        string p = $"backbone.blocks.{layer}.";
        _heads = config.NumHeads;
        _headDim = config.HeadDim;
        _width = config.HiddenDim;
        _scale = 1f / MathF.Sqrt(_headDim);
        _ones = Alloc(_headDim);
        _ones.AsSpan<float>().Fill(1f);

        _qSelfNorm = LoadAffine(w, p + "qSelfNorm");
        _qSelfQkv = LoadDense(w, p + "qSelfQkv");
        _qSelfOut = LoadDense(w, p + "qSelfOut");
        _qReadNorm = LoadAffine(w, p + "qReadNorm");
        _qReadQuery = LoadDense(w, p + "qReadQuery");
        _pReadNorm = LoadAffine(w, p + "pReadNorm");
        _pReadKv = LoadDense(w, p + "pReadKv");
        _qReadOut = LoadDense(w, p + "qReadOut");
        _qFfnNorm = LoadAffine(w, p + "qFfnNorm");
        _qFfnUp = LoadDense(w, p + "qFfnUp");
        _qFfnDown = LoadDense(w, p + "qFfnDown");
        _qWriteKv = LoadDense(w, p + "qWriteKv");
        _pWriteNorm = LoadAffine(w, p + "pWriteNorm");
        _pWriteQuery = LoadDense(w, p + "pWriteQuery");
        _pWriteQkNorm = w[p + "pWriteQkNorm.gamma"];
        _qWriteKNorm = w[p + "qWriteKNorm.gamma"];
        _pWriteOut = LoadDense(w, p + "pWriteOut");
        _pFfnNorm = LoadAffine(w, p + "pFfnNorm");
        _pFfnUp = LoadDense(w, p + "pFfnUp");
        _pFfnDown = LoadDense(w, p + "pFfnDown");
    }

    /// <summary>Consumes <paramref name="p"/> and <paramref name="q"/> (<c>[1, N, C]</c>) and returns the updated pair.</summary>
    internal (Tensor P, Tensor Q) Forward(IBackend backend, Tensor p, Tensor q, CoLa3DRopeTable pRope, CoLa3DRopeTable qRope, int rotaryDim)
    {
        q = SelfAttention(backend, q, qRope, rotaryDim);
        q = ReadPoints(backend, p, q, pRope, qRope, rotaryDim);
        p = WritePoints(backend, p, q, pRope, qRope, rotaryDim);
        p = FeedForward(backend, p, _pFfnNorm, _pFfnUp, _pFfnDown);
        q = FeedForward(backend, q, _qFfnNorm, _qFfnUp, _qFfnDown);
        return (p, q);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _ones.Dispose();
    }

    private Tensor SelfAttention(IBackend backend, Tensor q, CoLa3DRopeTable qRope, int rotaryDim)
    {
        int nq = (int)q.Shape[1];
        Tensor normed = Norm(backend, q, _qSelfNorm, BlockNormEps);
        Tensor qkv = Linear(backend, normed, _qSelfQkv, 3L * _width);
        normed.Dispose();
        Tensor qT = TokenHeads(backend, qkv, nq, 0);
        Tensor kT = TokenHeads(backend, qkv, nq, _width);
        Tensor vT = TokenHeads(backend, qkv, nq, 2 * _width);
        qkv.Dispose();
        CoLa3DRope.Apply(backend, qT, qRope, rotaryDim);
        CoLa3DRope.Apply(backend, kT, qRope, rotaryDim);
        Tensor ctx = Attend(backend, ToHeads(backend, qT, nq), ToHeads(backend, kT, nq), ToHeads(backend, vT, nq), nq);
        qT.Dispose();
        kT.Dispose();
        vT.Dispose();
        Tensor proj = Linear(backend, ctx, _qSelfOut, _width);
        ctx.Dispose();
        return Residual(backend, q, proj);
    }

    /// <summary>Queries attend to the points: queries supply the queries, points supply rotated keys and values.</summary>
    private Tensor ReadPoints(IBackend backend, Tensor p, Tensor q, CoLa3DRopeTable pRope, CoLa3DRopeTable qRope, int rotaryDim)
    {
        int np = (int)p.Shape[1], nq = (int)q.Shape[1];
        Tensor normedQ = Norm(backend, q, _qReadNorm, BlockNormEps);
        Tensor queryProj = Linear(backend, normedQ, _qReadQuery, _width);
        normedQ.Dispose();
        Tensor qT = TokenHeads(backend, queryProj, nq, 0);
        queryProj.Dispose();
        CoLa3DRope.Apply(backend, qT, qRope, rotaryDim);

        Tensor normedP = Norm(backend, p, _pReadNorm, BlockNormEps);
        Tensor kv = Linear(backend, normedP, _pReadKv, 2L * _width);
        normedP.Dispose();
        Tensor kT = TokenHeads(backend, kv, np, 0);
        Tensor vT = TokenHeads(backend, kv, np, _width);
        kv.Dispose();
        CoLa3DRope.Apply(backend, kT, pRope, rotaryDim);

        Tensor ctx = Attend(backend, ToHeads(backend, qT, nq), ToHeads(backend, kT, np), ToHeads(backend, vT, np), nq);
        qT.Dispose();
        kT.Dispose();
        vT.Dispose();
        Tensor proj = Linear(backend, ctx, _qReadOut, _width);
        ctx.Dispose();
        return Residual(backend, q, proj);
    }

    /// <summary>Points attend to the queries: points supply the queries, queries supply rotated keys and values.</summary>
    private Tensor WritePoints(IBackend backend, Tensor p, Tensor q, CoLa3DRopeTable pRope, CoLa3DRopeTable qRope, int rotaryDim)
    {
        int np = (int)p.Shape[1], nq = (int)q.Shape[1];
        Tensor normedP = Norm(backend, p, _pWriteNorm, BlockNormEps);
        Tensor pq = Linear(backend, normedP, _pWriteQuery, _width);
        normedP.Dispose();
        Tensor pqTok = TokenHeads(backend, pq, np, 0);
        pq.Dispose();
        Tensor pqNormed = RmsHeadNorm(backend, ToHeads(backend, pqTok, np), _pWriteQkNorm);
        pqTok.Dispose();
        Tensor pqT = HeadsToTokenHeads(backend, pqNormed, np);
        pqNormed.Dispose();
        CoLa3DRope.Apply(backend, pqT, pRope, rotaryDim);

        Tensor kvq = Linear(backend, q, _qWriteKv, 2L * _width);
        Tensor kT = TokenHeads(backend, kvq, nq, 0);
        Tensor vT = TokenHeads(backend, kvq, nq, _width);
        kvq.Dispose();
        Tensor kNormed = RmsHeadNorm(backend, ToHeads(backend, kT, nq), _qWriteKNorm);
        kT.Dispose();
        Tensor kTok = HeadsToTokenHeads(backend, kNormed, nq);
        kNormed.Dispose();
        CoLa3DRope.Apply(backend, kTok, qRope, rotaryDim);

        Tensor ctx = Attend(backend, ToHeads(backend, pqT, np), ToHeads(backend, kTok, nq), ToHeads(backend, vT, nq), np);
        pqT.Dispose();
        kTok.Dispose();
        vT.Dispose();
        Tensor proj = Linear(backend, ctx, _pWriteOut, _width);
        ctx.Dispose();
        return Residual(backend, p, proj);
    }

    private Tensor FeedForward(IBackend backend, Tensor x, Affine norm, Dense up, Dense down)
    {
        Tensor normed = Norm(backend, x, norm, BlockNormEps);
        Tensor hidden = Linear(backend, normed, up, up.Weight.Shape[0]);
        normed.Dispose();
        Tensor act = new(hidden.Shape, DType.F32);
        backend.Gelu(act, hidden);
        hidden.Dispose();
        Tensor delta = Linear(backend, act, down, _width);
        act.Dispose();
        return Residual(backend, x, delta);
    }

    /// <summary>Scaled dot-product attention over head-major operands, returned flat as <c>[1, Lq, C]</c>; consumes q, k and v.</summary>
    private Tensor Attend(IBackend backend, Tensor qh, Tensor kh, Tensor vh, int lq)
    {
        Tensor att = Alloc(1, _heads, lq, _headDim);
        backend.ScaledDotProductAttention(att, qh, kh, vh, null, _scale);
        qh.Dispose();
        kh.Dispose();
        vh.Dispose();
        Tensor flat = Alloc(1, lq, _width);
        backend.Permute0213(flat, att, _heads, lq, _headDim);
        att.Dispose();
        return flat;
    }

    /// <summary>Columns <c>[offset, offset + C)</c> of a flat <c>[1, tokens, C]</c> tensor as <c>[1, tokens, H, D]</c>.</summary>
    private Tensor TokenHeads(IBackend backend, Tensor flat, int tokens, int offset)
    {
        Tensor o = Alloc(1, tokens, _heads, _headDim);
        backend.SliceLastDim(o, flat, offset);
        return o;
    }

    /// <summary>Token-major <c>[1, S, H, D]</c> to head-major <c>[1, H, S, D]</c>; borrows its input.</summary>
    private Tensor ToHeads(IBackend backend, Tensor tokenHeads, int tokens)
    {
        Tensor o = Alloc(1, _heads, tokens, _headDim);
        backend.Permute0213(o, tokenHeads, tokens, _heads, _headDim);
        return o;
    }

    /// <summary>Head-major <c>[1, H, S, D]</c> to token-major <c>[1, S, H, D]</c>; borrows its input.</summary>
    private Tensor HeadsToTokenHeads(IBackend backend, Tensor heads, int tokens)
    {
        Tensor o = Alloc(1, tokens, _heads, _headDim);
        backend.Permute0213(o, heads, _heads, tokens, _headDim);
        return o;
    }

    /// <summary>Per-head RMS normalization times <c>gamma [H, D]</c>, the <c>F.normalize · gamma · sqrt(D)</c> rule; consumes <paramref name="heads"/>.</summary>
    private Tensor RmsHeadNorm(IBackend backend, Tensor heads, Tensor gamma)
    {
        Tensor rms = new(heads.Shape, DType.F32);
        backend.RmsNorm(rms, heads, _ones, 0f);
        heads.Dispose();
        Tensor o = new(rms.Shape, DType.F32);
        backend.AffineBroadcastLastDim(o, rms, gamma, null);
        rms.Dispose();
        return o;
    }
}
