using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS stage-2 structured-latent (SLAT) flow DiT (<c>slat_flow_img_dit_L_64l8p2</c>): a sparse U-Net predicting rectified-flow velocity over a <see cref="SparseTensor"/> of active voxels, image-conditioned via cross-attention.</summary>
public sealed unsafe class SlatFlowModel
{
    private const int Width = 1024, HeadDim = 64;
    private Tensor? _inW, _inB, _outW, _outB, _tIn1W, _tIn1B, _tIn2W, _tIn2B, _ones64;
    private readonly SlatResBlock3d[] _inBlocks = new SlatResBlock3d[2];
    private readonly SlatResBlock3d[] _outBlocks = new SlatResBlock3d[2];
    private readonly SsFlowBlock[] _blocks = new SsFlowBlock[24];
    private Tensor? _apeCache; private int[]? _apeKey;   // APE is geometry-only → cache across sampler steps (keyed by input coords ref)

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inW = F(w, "input_layer.weight"); _inB = F(w, "input_layer.bias");
        _outW = F(w, "out_layer.weight"); _outB = F(w, "out_layer.bias");
        _tIn1W = F(w, "t_embedder.mlp.0.weight"); _tIn1B = F(w, "t_embedder.mlp.0.bias");
        _tIn2W = F(w, "t_embedder.mlp.2.weight"); _tIn2B = F(w, "t_embedder.mlp.2.bias");
        _inBlocks[0] = SlatResBlock3d.Load(w, "input_blocks.0", downsample: false, upsample: false);
        _inBlocks[1] = SlatResBlock3d.Load(w, "input_blocks.1", downsample: true, upsample: false);
        _outBlocks[0] = SlatResBlock3d.Load(w, "out_blocks.0", downsample: false, upsample: true);
        _outBlocks[1] = SlatResBlock3d.Load(w, "out_blocks.1", downsample: false, upsample: false);
        for (int i = 0; i < 24; i++) _blocks[i] = SsFlowBlock.Load(w, $"blocks.{i}");
        _ones64 = new(new TensorShape(HeadDim), DType.F32);
        float* o = (float*)_ones64.DataPointer; for (int i = 0; i < HeadDim; i++) o[i] = 1f;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _inW, _inB, _outW, _outB, _tIn1W, _tIn1B, _tIn2W, _tIn2B, _ones64 }) if (t is not null) yield return t;
        foreach (SlatResBlock3d b in _inBlocks) foreach (Tensor t in b.Weights()) yield return t;
        foreach (SlatResBlock3d b in _outBlocks) foreach (Tensor t in b.Weights()) yield return t;
        foreach (SsFlowBlock b in _blocks) foreach (Tensor t in b.Weights()) yield return t;
    }

    /// <summary>Predicts the per-voxel velocity for <paramref name="x"/> at model-timestep <paramref name="tModel"/>, conditioned on <paramref name="cond"/> <c>[1, Lc, 1024]</c>.</summary>
    public SparseTensor Forward(IBackend backend, SparseTensor x, float tModel, Tensor cond)
    {
        Tensor vec = TimestepEmbed(backend, tModel);

        Tensor h0 = SparseLinear(backend, x.Feats, _inW!, _inB!);
        SparseTensor h = x.Replace(h0);

        DownState? down = null;
        List<Tensor> skips = new();
        foreach (SlatResBlock3d b in _inBlocks) { h = b.Forward(backend, h, vec, ref down, null); skips.Add(h.Feats); }

        // APE on the (downsampled) coords, added to feats. The APE is a pure function of the voxel geometry, which is
        // constant across all sampler steps (the sampler threads the SAME SparseTensor, mutating only feats in-place),
        // so cache it keyed by the input coords reference — recomputing the ~3M-trig host loop every step (44×, ~15 ms
        // each) is wasted work. Cached object is reused (not disposed), so it auto-promotes to a resident weight.
        if (_apeCache is null || !ReferenceEquals(_apeKey, x.Coords))
        {
            _apeCache?.Dispose();
            _apeCache = AbsolutePositionEmbed(h.Coords, h.Count, Width);
            _apeKey = x.Coords;
        }
        Tensor hAdd = new(h.Feats.Shape, DType.F32); backend.Add(hAdd, h.Feats, _apeCache);
        h = h.Replace(hAdd);

        // Thread each block's [1,n,Width] output straight into the next (no per-block reshape). As3D avoids a cache-
        // missing view entirely when h.Feats is already [1,n,Width]. Guard the dispose so the FIRST input — which may
        // alias the caller's h.Feats — is not freed mid-loop (h still references it until the Replace below).
        Tensor cur = SparseOps.As3D(h.Feats);
        foreach (SsFlowBlock b in _blocks)
        {
            Tensor nf = b.Forward(backend, cur, vec, cond, _ones64!);
            if (!ReferenceEquals(cur, h.Feats)) cur.Dispose();
            cur = nf;
        }
        h = h.Replace(cur);

        (int[] idx, int[] coords, int res) up = (down!.Value.Idx, down.Value.PreCoords, down.Value.PreResolution);
        for (int j = 0; j < _outBlocks.Length; j++)
        {
            Tensor skip = skips[skips.Count - 1 - j];
            int n = h.Count, c1 = h.Channels, c2 = (int)skip.Shape[skip.Shape.Rank - 1];
            Tensor cat = new(new TensorShape(1, n, c1 + c2), DType.F32);
            backend.Concat(cat, [SparseOps.As3D(h.Feats), SparseOps.As3D(skip)], 2);
            h = _outBlocks[j].Forward(backend, h.Replace(cat), vec, ref down, _outBlocks[j] == _outBlocks[0] ? up : null);
        }

        int fc = h.Channels;   // 128 after the out-blocks (out_layer maps 128 → 8)
        Tensor normed = new(new TensorShape(1, h.Count, fc), DType.F32); backend.LayerNormNoAffine(normed, SparseOps.As3D(h.Feats), 1e-5f);
        Tensor outFeats = SparseLinear(backend, normed, _outW!, _outB!); normed.Dispose();
        vec.Dispose();
        return h.Replace(outFeats);
    }

    private static Tensor SparseLinear(IBackend backend, Tensor feats, Tensor w, Tensor b)
    {
        int n = (int)(feats.ElementCount / feats.Shape[feats.Shape.Rank - 1]), cout = (int)w.Shape[0];
        Tensor o = new(new TensorShape(1, n, cout), DType.F32);
        backend.Linear(o, SparseOps.As3D(feats), w, b);
        return o;
    }

    /// <summary>AbsolutePositionEmbedder (channels 1024, 3D): per voxel, per axis, sin/cos frequency bands zero-padded to 1024 dims.</summary>
    private static Tensor AbsolutePositionEmbed(int[] coords, int n, int channels)
    {
        int freqDim = channels / 3 / 2;   // 170
        Tensor o = new(new TensorShape(1, n, channels), DType.F32);
        float* p = (float*)o.DataPointer; new Span<float>(p, n * channels).Clear();
        float[] freqs = new float[freqDim];
        for (int i = 0; i < freqDim; i++) freqs[i] = 1f / MathF.Pow(10000f, (float)i / freqDim);
        for (int v = 0; v < n; v++)
        {
            float* row = p + (long)v * channels;
            for (int axis = 0; axis < 3; axis++)
            {
                float coord = coords[v * 4 + 1 + axis];
                int baseIdx = axis * 2 * freqDim;
                for (int i = 0; i < freqDim; i++)
                {
                    float a = coord * freqs[i];
                    row[baseIdx + i] = MathF.Sin(a);
                    row[baseIdx + freqDim + i] = MathF.Cos(a);
                }
            }
        }
        return o;
    }

    private Tensor TimestepEmbed(IBackend backend, float tModel)
    {
        const int dim = 256; int half = dim / 2;
        Tensor sin = new(new TensorShape(1, dim), DType.F32);
        float* p = (float*)sin.DataPointer;
        float logMax = MathF.Log(10000f);
        for (int i = 0; i < half; i++) { float freq = MathF.Exp(-logMax * i / half), a = tModel * freq; p[i] = MathF.Cos(a); p[half + i] = MathF.Sin(a); }
        Tensor t1 = new(new TensorShape(1, Width), DType.F32); backend.Linear(t1, sin, _tIn1W!, _tIn1B!); sin.Dispose();
        Tensor t1a = new(t1.Shape, DType.F32); backend.Silu(t1a, t1); t1.Dispose();
        Tensor vec = new(new TensorShape(1, Width), DType.F32); backend.Linear(vec, t1a, _tIn2W!, _tIn2B!); t1a.Dispose();
        return vec;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => SparseStructureFlow.F(w, k);
}
