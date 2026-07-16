using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS <c>SparseResBlock3d</c>: <c>updown → norm1(affine)·SiLU·conv1 → norm2·(1+scale)+shift·SiLU·conv2 →
/// + skip</c>. Modulation <c>scale/shift = chunk(SiLU(emb)·emb_layers, 2)</c> broadcast over all voxels. Convs are
/// submanifold (scatter→Conv3d→gather); downsample = avg-pool, upsample = gather via the paired downsample's idx.</summary>
internal sealed unsafe class SlatResBlock3d
{
    private Tensor _n1W = null!, _n1B = null!;   // norm2 is non-affine (no params) → handled by LayerNormModulate
    private Tensor[] _c1Slices = null!, _c2Slices = null!;   // 27 [Cout,Cin] per-offset GEMM weights (rulebook conv)
    private Tensor _c1B = null!, _c2B = null!;
    private Tensor _embW = null!, _embB = null!;
    private Tensor? _skipW, _skipB;
    private int _inCh, _outCh;
    private bool _downsample, _upsample;

    public static SlatResBlock3d Load(IReadOnlyDictionary<string, Tensor> w, string p, bool downsample, bool upsample)
    {
        SlatResBlock3d b = new();
        Tensor F(string k) => SparseStructureFlow.F(w, $"{p}.{k}");
        b._n1W = F("norm1.weight"); b._n1B = F("norm1.bias");
        b._c1Slices = SparseOps.ConvWeightSlices(F("conv1.conv.weight"), f16: true); b._c1B = F("conv1.conv.bias");
        b._c2Slices = SparseOps.ConvWeightSlices(F("conv2.conv.weight"), f16: true); b._c2B = F("conv2.conv.bias");
        b._embW = F("emb_layers.1.weight"); b._embB = F("emb_layers.1.bias");
        b._inCh = (int)b._c1Slices[0].Shape[1]; b._outCh = (int)b._c1Slices[0].Shape[0];
        b._downsample = downsample; b._upsample = upsample;
        if (w.ContainsKey($"{p}.skip_connection.weight")) { b._skipW = F("skip_connection.weight"); b._skipB = F("skip_connection.bias"); }
        return b;
    }

    public IEnumerable<Tensor> Weights()
    {
        foreach (Tensor? t in new[] { _n1W, _n1B, _c1B, _c2B, _embW, _embB, _skipW, _skipB }) if (t is not null) yield return t;
        foreach (Tensor t in _c1Slices) yield return t;
        foreach (Tensor t in _c2Slices) yield return t;
    }

    /// <summary>Applies the res-block. <paramref name="up"/> is the paired downsample state (idx + pre-down coords +
    /// resolution) used only when this block upsamples.</summary>
    public SparseTensor Forward(IBackend backend, SparseTensor x, Tensor vec, ref DownState? down, (int[] idx, int[] coords, int res)? up)
    {
        // updown FIRST (per reference).
        if (_downsample) { (SparseTensor d, int[] idx) = SparseOps.Downsample(backend, x, 2); down = new DownState(idx, x.Coords, x.Resolution); x = d; }
        else if (_upsample) { (int[] uidx, int[] ucoords, int ures) = up!.Value; x = SparseOps.Upsample(backend, x, uidx, ucoords, ures); }

        // emb → scale,shift  (SiLU(vec) → Linear(2*out)).
        Tensor sil = new(vec.Shape, DType.F32); backend.Silu(sil, vec);
        Tensor emb = new(new TensorShape(1, 2 * _outCh), DType.F32); backend.Linear(emb, sil, _embW, _embB); sil.Dispose();
        Tensor shift = new(new TensorShape(1, _outCh), DType.F32); backend.SliceLastDim(shift, emb, _outCh);   // chunk: [scale, shift]
        Tensor scale = new(new TensorShape(1, _outCh), DType.F32); backend.SliceLastDim(scale, emb, 0); emb.Dispose();

        int n = x.Count;
        Tensor xf = x.Feats;   // [1,N,inCh]
        Tensor h1 = new(new TensorShape(1, n, _inCh), DType.F32); backend.LayerNorm(h1, xf, _n1W, _n1B, 1e-6f);
        Tensor a1 = new(h1.Shape, DType.F32); backend.Silu(a1, h1); h1.Dispose();
        SparseTensor hc = SparseOps.SubmanifoldConv3dSparse(backend, x.Replace(a1), _c1Slices, _c1B); a1.Dispose();   // → [.,outCh]

        Tensor h2 = new(new TensorShape(1, n, _outCh), DType.F32); backend.LayerNormModulate(h2, SparseOps.As3D(hc.Feats), scale, shift, 1e-6f);
        Tensor a2 = new(h2.Shape, DType.F32); backend.Silu(a2, h2); h2.Dispose();
        SparseTensor conv2 = SparseOps.SubmanifoldConv3dSparse(backend, hc.Replace(a2), _c2Slices, _c2B); a2.Dispose();
        scale.Dispose(); shift.Dispose();

        // skip: SparseLinear(x) if channels differ, else x. Residual add.
        Tensor skipFeats;
        if (_skipW is not null) { skipFeats = new(new TensorShape(1, n, _outCh), DType.F32); backend.Linear(skipFeats, x.Feats, _skipW, _skipB); }
        else skipFeats = x.Feats;
        Tensor outFeats = new(new TensorShape(1, n, _outCh), DType.F32); backend.Add(outFeats, conv2.Feats, skipFeats);
        return conv2.Replace(outFeats);
    }
}

/// <summary>Downsample pairing state carried from a downsampling res-block to its paired upsampling block.</summary>
internal readonly record struct DownState(int[] Idx, int[] PreCoords, int PreResolution);
