using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 ShapeVAE — <b>decoder only</b> (generation doesn't need the encoder). Turns a VecSet
/// shape latent <c>[1,N,C]</c> into a dense occupancy/SDF field by, for each 3D query point, Fourier-embedding
/// the coordinate and cross-attending it to the latent token set through a few transformer blocks, then
/// projecting to a scalar. Queries are evaluated over a regular grid in chunks (the grid can be millions of
/// points) and assembled into a <see cref="ScalarField3D"/> for marching cubes.
/// <para><b>Numerics validation-pending</b> — query positional encoding (Fourier bands), latent
/// normalization, decoder depth, and iso level are validation-gated. Structurally green on synthetic weights.</para></summary>
public sealed unsafe class Hunyuan3DShapeVae
{
    private readonly Hunyuan3DConfig _cfg;
    private readonly int _fourierDim;
    private readonly Hunyuan3DVaeBlock[] _blocks;

    private Tensor? _latentProjW, _latentProjB; // C → VaeWidth (KV space)
    private Tensor? _queryProjW, _queryProjB;   // fourierDim → VaeWidth
    private Tensor? _outW, _outB;               // VaeWidth → 1

    public Hunyuan3DConfig Config => _cfg;

    public Hunyuan3DShapeVae(Hunyuan3DConfig cfg)
    {
        _cfg = cfg;
        _fourierDim = 3 + 3 * 2 * cfg.FourierBands; // raw xyz + (sin,cos) per axis per band
        _blocks = new Hunyuan3DVaeBlock[cfg.VaeDepth];
        for (int i = 0; i < cfg.VaeDepth; i++) _blocks[i] = new Hunyuan3DVaeBlock(cfg);
    }

    /// <summary>Number of Fourier-embedded query features.</summary>
    public int FourierDim => _fourierDim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _latentProjW = Hunyuan3DDit.F32(w[$"{p}latent_proj.weight"]); _latentProjB = Hunyuan3DDit.F32(w[$"{p}latent_proj.bias"]);
        _queryProjW = Hunyuan3DDit.F32(w[$"{p}query_proj.weight"]); _queryProjB = Hunyuan3DDit.F32(w[$"{p}query_proj.bias"]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{p}blocks.{i}");
        _outW = Hunyuan3DDit.F32(w[$"{p}out.weight"]); _outB = Hunyuan3DDit.F32(w[$"{p}out.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_latentProjW, _latentProjB, _queryProjW, _queryProjB, _outW, _outB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (Hunyuan3DVaeBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes <paramref name="latent"/> <c>[1,N,C]</c> into a <paramref name="resolution"/>³ occupancy
    /// field over the cube <c>[-bound, bound]³</c>, evaluating query points in chunks of
    /// <paramref name="chunkSize"/>.</summary>
    public ScalarField3D Decode(IBackend backend, Tensor latent, int resolution, float bound, int chunkSize = 32768)
    {
        if (latent.Shape.Rank != 3 || latent.Shape[0] != 1)
            throw new ArgumentException($"latent must be [1,N,C]; got {latent.Shape}.", nameof(latent));

        int width = _cfg.VaeWidth;
        // Project latent → KV once.
        Tensor kv = new(new TensorShape(1, (int)latent.Shape[1], width), DType.F32);
        backend.Linear(kv, latent, _latentProjW!, _latentProjB!);

        int res = resolution;
        long total = (long)res * res * res;
        float[] values = new float[total];

        float[] coordBuf = new float[chunkSize * 3];
        long produced = 0;
        try
        {
            while (produced < total)
            {
                int count = (int)Math.Min(chunkSize, total - produced);
                // Build chunk query coords in field order (x fastest, then y, then z).
                for (int i = 0; i < count; i++)
                {
                    long lin = produced + i;
                    int x = (int)(lin % res);
                    int y = (int)((lin / res) % res);
                    int z = (int)(lin / ((long)res * res));
                    coordBuf[i * 3] = Map(x, res, bound);
                    coordBuf[i * 3 + 1] = Map(y, res, bound);
                    coordBuf[i * 3 + 2] = Map(z, res, bound);
                }

                Tensor occ = DecodePoints(backend, kv, coordBuf.AsSpan(0, count * 3), count);
                new ReadOnlySpan<float>((float*)occ.DataPointer, count).CopyTo(values.AsSpan((int)produced, count));
                occ.Dispose();
                produced += count;
            }
        }
        finally { kv.Dispose(); }

        return new ScalarField3D
        {
            Values = values, ResX = res, ResY = res, ResZ = res,
            Min = (-bound, -bound, -bound), Max = (bound, bound, bound),
        };
    }

    /// <summary>Evaluates occupancy at an arbitrary set of points (<paramref name="coords"/> = 3*count xyz),
    /// returning <c>[1, count]</c>. Exposed for testing / sparse evaluation; <paramref name="kv"/> is the
    /// projected latent <c>[1,N,VaeWidth]</c>.</summary>
    public Tensor DecodePoints(IBackend backend, Tensor kv, ReadOnlySpan<float> coords, int count)
    {
        int width = _cfg.VaeWidth;
        // Fourier embed → [1, count, fourierDim].
        Tensor feat = new(new TensorShape(1, count, _fourierDim), DType.F32);
        FourierEmbed(feat, coords, count);

        // Query proj → [1, count, VaeWidth].
        Tensor q = new(new TensorShape(1, count, width), DType.F32);
        backend.Linear(q, feat, _queryProjW!, _queryProjB!); feat.Dispose();

        Tensor h = q;
        foreach (Hunyuan3DVaeBlock block in _blocks) { Tensor n = block.Forward(backend, h, kv); h.Dispose(); h = n; }

        // Final scalar projection → [1, count, 1] → reshape [1, count].
        Tensor outp = new(new TensorShape(1, count, 1), DType.F32);
        backend.Linear(outp, h, _outW!, _outB!); h.Dispose();
        Tensor scalar = outp.Reshape(new TensorShape(1, count));
        Tensor copy = new(scalar.Shape, DType.F32);
        new ReadOnlySpan<float>((float*)scalar.DataPointer, count).CopyTo(new Span<float>((float*)copy.DataPointer, count));
        outp.Dispose();
        return copy;
    }

    private void FourierEmbed(Tensor dst, ReadOnlySpan<float> coords, int count)
    {
        float* dp = (float*)dst.DataPointer;
        int bands = _cfg.FourierBands;
        for (int i = 0; i < count; i++)
        {
            long b = (long)i * _fourierDim;
            float x = coords[i * 3], y = coords[i * 3 + 1], z = coords[i * 3 + 2];
            dp[b] = x; dp[b + 1] = y; dp[b + 2] = z;
            int o = 3;
            for (int band = 0; band < bands; band++)
            {
                float f = MathF.PI * (1 << band);
                dp[b + o++] = MathF.Sin(f * x); dp[b + o++] = MathF.Cos(f * x);
                dp[b + o++] = MathF.Sin(f * y); dp[b + o++] = MathF.Cos(f * y);
                dp[b + o++] = MathF.Sin(f * z); dp[b + o++] = MathF.Cos(f * z);
            }
        }
    }

    private static float Map(int i, int res, float bound) =>
        -bound + (res > 1 ? i / (float)(res - 1) : 0f) * (2f * bound);
}

/// <summary>ShapeVAE decoder block: cross-attention (query → latent) + GELU MLP, each with a no-affine
/// pre-norm and residual. Queries are independent (no query self-attention).</summary>
internal sealed unsafe class Hunyuan3DVaeBlock
{
    private readonly Hunyuan3DConfig _cfg;
    private readonly int _mlpDim;
    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _fc1W, _fc1B, _fc2W, _fc2B;

    public Hunyuan3DVaeBlock(Hunyuan3DConfig cfg) { _cfg = cfg; _mlpDim = 4 * cfg.VaeWidth; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _qW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.q.weight"]); _qB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.q.bias"]);
        _kW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.k.weight"]); _kB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.k.bias"]);
        _vW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.v.weight"]); _vB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.v.bias"]);
        _oW = Hunyuan3DDit.F32(w[$"{p}.cross_attn.o.weight"]); _oB = Hunyuan3DDit.F32(w[$"{p}.cross_attn.o.bias"]);
        _fc1W = Hunyuan3DDit.F32(w[$"{p}.mlp.fc1.weight"]); _fc1B = Hunyuan3DDit.F32(w[$"{p}.mlp.fc1.bias"]);
        _fc2W = Hunyuan3DDit.F32(w[$"{p}.mlp.fc2.weight"]); _fc2B = Hunyuan3DDit.F32(w[$"{p}.mlp.fc2.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _fc1W, _fc1B, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor query, Tensor kv)
    {
        int n = (int)query.Shape[1], width = _cfg.VaeWidth, sk = (int)kv.Shape[1];

        Tensor qn = new(query.Shape, DType.F32); DiTUtils.LayerNormNoAffine(qn, query, 1, n, width, 1e-6f);
        Tensor q = new(new TensorShape(1, n, width), DType.F32); backend.Linear(q, qn, _qW!, _qB!); qn.Dispose();
        Tensor k = new(new TensorShape(1, sk, width), DType.F32); backend.Linear(k, kv, _kW!, _kB!);
        Tensor v = new(new TensorShape(1, sk, width), DType.F32); backend.Linear(v, kv, _vW!, _vB!);
        Tensor a = Hunyuan3DAttention.Attend(backend, q, k, v, _cfg.VaeNumHeads); q.Dispose(); k.Dispose(); v.Dispose();
        Tensor ao = new(new TensorShape(1, n, width), DType.F32); backend.Linear(ao, a, _oW!, _oB!); a.Dispose();
        Tensor x1 = new(query.Shape, DType.F32); backend.Add(x1, query, ao); ao.Dispose();

        Tensor mn = new(x1.Shape, DType.F32); DiTUtils.LayerNormNoAffine(mn, x1, 1, n, width, 1e-6f);
        Tensor f1 = new(new TensorShape(1, n, _mlpDim), DType.F32); backend.Linear(f1, mn, _fc1W!, _fc1B!); mn.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(new TensorShape(1, n, width), DType.F32); backend.Linear(f2, act, _fc2W!, _fc2B!); act.Dispose();
        Tensor x2 = new(x1.Shape, DType.F32); backend.Add(x2, x1, f2); x1.Dispose(); f2.Dispose();
        return x2;
    }
}
