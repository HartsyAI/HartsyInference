using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS SLAT → Gaussian-splat VAE decoder (<c>slat_dec_gs_swin8_B_64l8gs32</c>): a plain sparse
/// transformer (DiT-B, 12 blocks, swin window-8 attention, no modulation/cross-attn/qk-norm) → per-voxel
/// 448-dim head = 32 gaussians × (xyz3 + features_dc3 + scaling3 + rotation4 + opacity1). Windowed attention is one
/// masked dense SDPA over all voxels with a block-diagonal mask (attend within the same 8³ window; shift alternates
/// 0/4 → two precomputed masks). B=1. All F32. See <c>docs/Research/TRELLIS_ARCHITECTURE.md</c>.</summary>
public sealed unsafe class SlatGaussianDecoder
{
    private const int Width = 768, Heads = 12, HeadDim = 64, WindowSize = 8, OutCh = 448;
    private static readonly float Scale = 1f / MathF.Sqrt(HeadDim);

    private Tensor? _inW, _inB, _outW, _outB;
    private readonly GsBlock[] _blocks = new GsBlock[12];

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inW = w["input_layer.weight"]; _inB = F(w, "input_layer.bias");   // GEMM weight native F16 (F32-accum GEMM); see SsFlowBlock
        _outW = w["out_layer.weight"]; _outB = F(w, "out_layer.bias");
        for (int i = 0; i < 12; i++) _blocks[i] = GsBlock.Load(w, $"blocks.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _inW, _inB, _outW, _outB }) if (t is not null) yield return t;
        foreach (GsBlock b in _blocks) foreach (Tensor t in b.Weights()) yield return t;
    }

    /// <summary>Runs the decoder network, returning the per-voxel 448-dim gaussian-parameter feats (pre-representation).</summary>
    public SparseTensor Forward(IBackend backend, SparseTensor x)
    {
        int n = x.Count;
        Tensor h0 = SlatLinear(backend, x.Feats, _inW!, _inB!, n);
        SparseTensor h = x.Replace(h0);
        Tensor ape = SlatFlowModelApe(x.Coords, n);
        Tensor hAdd = new(h.Feats.Shape, DType.F32); backend.Add(hAdd, h.Feats, ape); ape.Dispose();
        h = h.Replace(hAdd);

        // Two block-diagonal window masks (shift 0 and window/2), reused across alternating blocks.
        Tensor mask0 = WindowMask(backend, x.Coords, n, 0);
        Tensor mask4 = WindowMask(backend, x.Coords, n, WindowSize / 2);
        // Thread each block's [1,n,Width] output straight into the next; As3D avoids a cache-missing view when the
        // feats are already [1,n,Width]. Guard the dispose so the aliased first input (h.Feats) is not freed mid-loop.
        Tensor cur = SparseOps.As3D(h.Feats);
        for (int i = 0; i < 12; i++)
        {
            Tensor nf = _blocks[i].Forward(backend, cur, (i % 2 == 0) ? mask0 : mask4, n);
            if (!ReferenceEquals(cur, h.Feats)) cur.Dispose();
            cur = nf;
        }
        h = h.Replace(cur);
        mask0.Dispose(); mask4.Dispose();

        Tensor normed = new(new TensorShape(1, n, Width), DType.F32); backend.LayerNormNoAffine(normed, SparseOps.As3D(h.Feats), 1e-5f);
        Tensor outFeats = SlatLinear(backend, normed, _outW!, _outB!, n); normed.Dispose();
        return h.Replace(outFeats);
    }

    /// <summary>Block-diagonal attention mask <c>[1,1,N,N]</c>: 0 where two voxels share the same
    /// <c>((coord+shift)/window)</c> cube, −1e30 otherwise (additive pre-softmax).</summary>
    private static Tensor WindowMask(IBackend backend, int[] coords, int n, int shift)
    {
        int[] wid = new int[n];
        int mx = 0, my = 0, mz = 0;
        for (int i = 0; i < n; i++) { mx = Math.Max(mx, (coords[i * 4 + 1] + shift) / WindowSize); my = Math.Max(my, (coords[i * 4 + 2] + shift) / WindowSize); mz = Math.Max(mz, (coords[i * 4 + 3] + shift) / WindowSize); }
        int ny = my + 1, nz = mz + 1;
        for (int i = 0; i < n; i++)
        {
            int wx = (coords[i * 4 + 1] + shift) / WindowSize, wy = (coords[i * 4 + 2] + shift) / WindowSize, wz = (coords[i * 4 + 3] + shift) / WindowSize;
            wid[i] = (wx * ny + wy) * nz + wz;
        }
        Tensor mask = new(new TensorShape(1, 1, n, n), DType.F32);
        float* m = (float*)mask.DataPointer;
        for (int i = 0; i < n; i++) { long row = (long)i * n; for (int j = 0; j < n; j++) m[row + j] = wid[i] == wid[j] ? 0f : -1e30f; }
        return mask;
    }

    internal static Tensor SlatLinear(IBackend backend, Tensor feats, Tensor w, Tensor b, int n)
    {
        int cout = (int)w.Shape[0];
        Tensor o = new(new TensorShape(1, n, cout), DType.F32);
        backend.Linear(o, SparseOps.As3D(feats), w, b);
        return o;
    }

    /// <summary>AbsolutePositionEmbedder for the 768-dim decoder (freq_dim = 768/3/2 = 128).</summary>
    private static Tensor SlatFlowModelApe(int[] coords, int n)
    {
        int freqDim = Width / 3 / 2;
        Tensor o = new(new TensorShape(1, n, Width), DType.F32);
        float* p = (float*)o.DataPointer; new Span<float>(p, n * Width).Clear();
        float[] freqs = new float[freqDim];
        for (int i = 0; i < freqDim; i++) freqs[i] = 1f / MathF.Pow(10000f, (float)i / freqDim);
        for (int v = 0; v < n; v++)
        {
            float* row = p + (long)v * Width;
            for (int axis = 0; axis < 3; axis++)
            {
                float coord = coords[v * 4 + 1 + axis]; int baseIdx = axis * 2 * freqDim;
                for (int i = 0; i < freqDim; i++) { float a = coord * freqs[i]; row[baseIdx + i] = MathF.Sin(a); row[baseIdx + freqDim + i] = MathF.Cos(a); }
            }
        }
        return o;
    }

    private static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => SparseStructureFlow.F(w, k);
}

/// <summary>Non-modulated sparse transformer block: <c>norm1·windowed-SDPA·+x → norm2·tanh-GELU-MLP·+x</c>
/// (norms non-affine eps 1e-6, no qk-norm).</summary>
internal sealed unsafe class GsBlock
{
    private const int W = 768, H = 12, Hd = 64, Mlp = 3072;
    private static readonly float Scale = 1f / MathF.Sqrt(Hd);
    private Tensor _qkvW = null!, _qkvB = null!, _oW = null!, _oB = null!, _m0W = null!, _m0B = null!, _m2W = null!, _m2B = null!;

    public static GsBlock Load(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        GsBlock b = new();
        Tensor F(string k) => SparseStructureFlow.F(w, $"{p}.{k}");
        Tensor Fw(string k) => w[$"{p}.{k}"];   // GEMM weight native F16 (F32-accum GEMM); norms are non-affine here
        b._qkvW = Fw("attn.to_qkv.weight"); b._qkvB = F("attn.to_qkv.bias");
        b._oW = Fw("attn.to_out.weight"); b._oB = F("attn.to_out.bias");
        b._m0W = Fw("mlp.mlp.0.weight"); b._m0B = F("mlp.mlp.0.bias");
        b._m2W = Fw("mlp.mlp.2.weight"); b._m2B = F("mlp.mlp.2.bias");
        return b;
    }

    public IEnumerable<Tensor> Weights() => new[] { _qkvW, _qkvB, _oW, _oB, _m0W, _m0B, _m2W, _m2B };

    public Tensor Forward(IBackend backend, Tensor x, Tensor mask, int n)
    {
        TensorShape flat = new(1, n, W), heads = new(1, n, H, Hd), mh = new(1, H, n, Hd);
        Tensor h = new(flat, DType.F32); backend.LayerNormNoAffine(h, x, 1e-6f);
        Tensor qkv = new(new TensorShape(1, n, 3 * W), DType.F32); backend.Linear(qkv, h, _qkvW, _qkvB); h.Dispose();
        Tensor q = new(heads, DType.F32); backend.SliceLastDim(q, qkv, 0);
        Tensor k = new(heads, DType.F32); backend.SliceLastDim(k, qkv, W);
        Tensor v = new(heads, DType.F32); backend.SliceLastDim(v, qkv, 2 * W); qkv.Dispose();
        Tensor qP = new(mh, DType.F32); backend.Permute0213(qP, q, n, H, Hd); q.Dispose();
        Tensor kP = new(mh, DType.F32); backend.Permute0213(kP, k, n, H, Hd); k.Dispose();
        Tensor vP = new(mh, DType.F32); backend.Permute0213(vP, v, n, H, Hd); v.Dispose();
        Tensor attn = new(mh, DType.F32); backend.ScaledDotProductAttention(attn, qP, kP, vP, mask, Scale, allowF16: true);
        qP.Dispose(); kP.Dispose(); vP.Dispose();
        Tensor attnFlat = new(flat, DType.F32); backend.Permute0213(attnFlat, attn, H, n, Hd); attn.Dispose();
        Tensor o = new(flat, DType.F32); backend.Linear(o, attnFlat, _oW, _oB); attnFlat.Dispose();
        Tensor x1 = new(flat, DType.F32); backend.Add(x1, x, o); o.Dispose();

        Tensor h2 = new(flat, DType.F32); backend.LayerNormNoAffine(h2, x1, 1e-6f);
        Tensor f1 = new(new TensorShape(1, n, Mlp), DType.F32); backend.Linear(f1, h2, _m0W, _m0B); h2.Dispose();
        Tensor act = new(f1.Shape, DType.F32); backend.Gelu(act, f1); f1.Dispose();
        Tensor f2 = new(flat, DType.F32); backend.Linear(f2, act, _m2W, _m2B); act.Dispose();
        Tensor x2 = new(flat, DType.F32); backend.Add(x2, x1, f2); x1.Dispose(); f2.Dispose();
        return x2;
    }
}
