using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS stage-2 structured-latent (SLAT) flow DiT (<c>slat_flow_img_dit_L_64l8p2</c>): predicts the
/// rectified-flow velocity over a <see cref="SparseTensor"/> of active voxels, image-conditioned via cross-attention.
/// A sparse U-Net: <c>input_layer → io ResBlocks (1 downsample) → APE(coords) → 24 transformer blocks (full attn,
/// reuse <see cref="SsFlowBlock"/>) → io ResBlocks (upsample + skip-concat) → out_layer</c>. B=1 (single object) so
/// full attention = dense SDPA over all voxels. All F32. See <c>docs/Research/TRELLIS_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class SlatFlowModel
{
    private const int Width = 1024, HeadDim = 64;
    private Tensor? _inW, _inB, _outW, _outB, _tIn1W, _tIn1B, _tIn2W, _tIn2B, _ones64;
    private readonly SlatResBlock3d[] _inBlocks = new SlatResBlock3d[2];
    private readonly SlatResBlock3d[] _outBlocks = new SlatResBlock3d[2];
    private readonly SsFlowBlock[] _blocks = new SsFlowBlock[24];

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

    /// <summary>Predicts the per-voxel velocity for <paramref name="x"/> at model-timestep <paramref name="tModel"/>,
    /// conditioned on <paramref name="cond"/> <c>[1, Lc, 1024]</c>. Returns a SparseTensor with velocity feats
    /// <c>[1, N, 8]</c> at the input voxels.</summary>
    public SparseTensor Forward(IBackend backend, SparseTensor x, float tModel, Tensor cond)
    {
        Tensor vec = TimestepEmbed(backend, tModel);

        Tensor h0 = SparseLinear(backend, x.Feats, _inW!, _inB!);
        SparseTensor h = x.Replace(h0);

        DownState? down = null;
        List<Tensor> skips = new();
        foreach (SlatResBlock3d b in _inBlocks) { h = b.Forward(backend, h, vec, ref down, null); skips.Add(h.Feats); }

        // APE on the (downsampled) coords, added to feats.
        Tensor ape = AbsolutePositionEmbed(h.Coords, h.Count, Width);
        Tensor hAdd = new(h.Feats.Shape, DType.F32); backend.Add(hAdd, h.Feats, ape); ape.Dispose();
        h = h.Replace(hAdd);

        foreach (SsFlowBlock b in _blocks)
        {
            Tensor nf = b.Forward(backend, h.Feats.Reshape(new TensorShape(1, h.Count, Width)), vec, cond, _ones64!);
            h = h.Replace(nf);
        }

        (int[] idx, int[] coords, int res) up = (down!.Value.Idx, down.Value.PreCoords, down.Value.PreResolution);
        for (int j = 0; j < _outBlocks.Length; j++)
        {
            Tensor skip = skips[skips.Count - 1 - j];
            int n = h.Count, c1 = h.Channels, c2 = (int)skip.Shape[skip.Shape.Rank - 1];
            Tensor cat = new(new TensorShape(1, n, c1 + c2), DType.F32);
            backend.Concat(cat, [h.Feats.Reshape(new TensorShape(1, n, c1)), skip.Reshape(new TensorShape(1, n, c2))], 2);
            h = _outBlocks[j].Forward(backend, h.Replace(cat), vec, ref down, _outBlocks[j] == _outBlocks[0] ? up : null);
        }

        int fc = h.Channels;   // 128 after the out-blocks (out_layer maps 128 → 8)
        Tensor normed = new(new TensorShape(1, h.Count, fc), DType.F32); backend.LayerNormNoAffine(normed, h.Feats.Reshape(new TensorShape(1, h.Count, fc)), 1e-5f);
        Tensor outFeats = SparseLinear(backend, normed, _outW!, _outB!); normed.Dispose();
        vec.Dispose();
        return h.Replace(outFeats);
    }

    private static Tensor SparseLinear(IBackend backend, Tensor feats, Tensor w, Tensor b)
    {
        int n = (int)(feats.ElementCount / feats.Shape[feats.Shape.Rank - 1]), cout = (int)w.Shape[0];
        Tensor o = new(new TensorShape(1, n, cout), DType.F32);
        backend.Linear(o, feats.Reshape(new TensorShape(1, n, (int)feats.Shape[feats.Shape.Rank - 1])), w, b);
        return o;
    }

    /// <summary>AbsolutePositionEmbedder(channels 1024, 3D): per voxel, per axis a∈{x,y,z} the 340-dim
    /// <c>[sin(a·freqs), cos(a·freqs)]</c> (freq_dim 170, freqs 1/10000^(i/170)) → 1020 dims, zero-padded to 1024.</summary>
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
