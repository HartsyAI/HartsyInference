using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS sparse-structure VAE decoder (<c>ss_dec_conv3d_16l8</c>): decodes an <c>[1,8,16³]</c> latent to a <c>[1,1,64³]</c> occupancy-logit grid via 3D conv res-blocks + pixel-shuffle upsampling.</summary>
public sealed class SparseStructureDecoder
{
    private const float NormEps = 1e-5f;

    private Tensor? _inW, _inB;                 // input_layer Conv3d 8→512
    private readonly ResBlock3d[] _middle = new ResBlock3d[2];
    private readonly object[] _blocks = new object[8];   // ResBlock3d or UpsampleBlock3d
    private Tensor? _outNormW, _outNormB;       // out_layer.0 ChannelLayerNorm(32)
    private Tensor? _outConvW, _outConvB;       // out_layer.2 Conv3d 32→1

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inW = F(w, "input_layer.weight"); _inB = F(w, "input_layer.bias");
        _middle[0] = ResBlock3d.Load(w, "middle_block.0");
        _middle[1] = ResBlock3d.Load(w, "middle_block.1");
        int[] upsample = [2, 5];
        for (int i = 0; i < 8; i++)
            _blocks[i] = Array.IndexOf(upsample, i) >= 0 ? UpsampleBlock3d.Load(w, $"blocks.{i}") : ResBlock3d.Load(w, $"blocks.{i}");
        _outNormW = F(w, "out_layer.0.weight"); _outNormB = F(w, "out_layer.0.bias");
        _outConvW = F(w, "out_layer.2.weight"); _outConvB = F(w, "out_layer.2.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _inW, _inB, _outNormW, _outNormB, _outConvW, _outConvB }) if (t is not null) yield return t;
        foreach (ResBlock3d r in _middle) foreach (Tensor t in r.Weights()) yield return t;
        foreach (object b in _blocks)
            foreach (Tensor t in b is ResBlock3d rb ? rb.Weights() : ((UpsampleBlock3d)b).Weights()) yield return t;
    }

    /// <summary>Decodes <paramref name="latent"/> <c>[1,8,16,16,16]</c> → occupancy logits <c>[1,1,64,64,64]</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        Tensor h = Conv3d(backend, latent, _inW!, _inB!, 512);
        foreach (ResBlock3d r in _middle) { Tensor nh = r.Forward(backend, h); h.Dispose(); h = nh; }
        foreach (object b in _blocks)
        {
            Tensor nh = b is ResBlock3d rb ? rb.Forward(backend, h) : ((UpsampleBlock3d)b).Forward(backend, h);
            h.Dispose(); h = nh;
        }
        int c = (int)h.Shape[1];
        Tensor normed = new(h.Shape, DType.F32); backend.ChannelLayerNorm3d(normed, h, _outNormW!, _outNormB!, NormEps); h.Dispose();
        Tensor act = new(normed.Shape, DType.F32); backend.Silu(act, normed); normed.Dispose();
        Tensor occ = Conv3d(backend, act, _outConvW!, _outConvB!, 1); act.Dispose();
        return occ;
    }

    internal static Tensor Conv3d(IBackend backend, Tensor x, Tensor weight, Tensor bias, int cOut)
    {
        int d = (int)x.Shape[2], hh = (int)x.Shape[3], ww = (int)x.Shape[4];
        Tensor o = new(new TensorShape(new long[] { x.Shape[0], cOut, d, hh, ww }), DType.F32);
        backend.Conv3d(o, x, weight, bias, 1, 1, 1, 1, 1, 1);   // k3 s1 p1
        return o;
    }

    internal static Tensor F(IReadOnlyDictionary<string, Tensor> w, string k) => TensorCasts.LoadF32(w, k);
}

/// <summary>3D residual block: ChannelLayerNorm → SiLU → Conv3d(k3) ×2, identity skip (all TRELLIS SS-VAE res-blocks
/// have equal in/out channels).</summary>
internal sealed class ResBlock3d
{
    private Tensor _n1W = null!, _n1B = null!, _c1W = null!, _c1B = null!, _n2W = null!, _n2B = null!, _c2W = null!, _c2B = null!;
    private int _channels;

    public static ResBlock3d Load(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        ResBlock3d r = new();
        r._n1W = SparseStructureDecoder.F(w, $"{p}.norm1.weight"); r._n1B = SparseStructureDecoder.F(w, $"{p}.norm1.bias");
        r._c1W = SparseStructureDecoder.F(w, $"{p}.conv1.weight"); r._c1B = SparseStructureDecoder.F(w, $"{p}.conv1.bias");
        r._n2W = SparseStructureDecoder.F(w, $"{p}.norm2.weight"); r._n2B = SparseStructureDecoder.F(w, $"{p}.norm2.bias");
        r._c2W = SparseStructureDecoder.F(w, $"{p}.conv2.weight"); r._c2B = SparseStructureDecoder.F(w, $"{p}.conv2.bias");
        r._channels = (int)r._c1W.Shape[0];
        return r;
    }

    public IEnumerable<Tensor> Weights() => new[] { _n1W, _n1B, _c1W, _c1B, _n2W, _n2B, _c2W, _c2B };

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor n1 = new(x.Shape, DType.F32); backend.ChannelLayerNorm3d(n1, x, _n1W, _n1B, 1e-5f);
        Tensor a1 = new(x.Shape, DType.F32); backend.Silu(a1, n1); n1.Dispose();
        Tensor c1 = SparseStructureDecoder.Conv3d(backend, a1, _c1W, _c1B, _channels); a1.Dispose();
        Tensor n2 = new(c1.Shape, DType.F32); backend.ChannelLayerNorm3d(n2, c1, _n2W, _n2B, 1e-5f); c1.Dispose();
        Tensor a2 = new(n2.Shape, DType.F32); backend.Silu(a2, n2); n2.Dispose();
        Tensor c2 = SparseStructureDecoder.Conv3d(backend, a2, _c2W, _c2B, _channels); a2.Dispose();
        Tensor outp = new(x.Shape, DType.F32); backend.Add(outp, c2, x); c2.Dispose();
        return outp;
    }
}

/// <summary>3D upsample block: Conv3d(cin → cout·8, k3) → pixel-shuffle ×2 (depth-to-space), doubling each spatial dim.</summary>
internal sealed class UpsampleBlock3d
{
    private Tensor _convW = null!, _convB = null!;
    private int _cOut8;   // conv output channels (= cout·8)

    public static UpsampleBlock3d Load(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        UpsampleBlock3d u = new();
        u._convW = SparseStructureDecoder.F(w, $"{p}.conv.weight"); u._convB = SparseStructureDecoder.F(w, $"{p}.conv.bias");
        u._cOut8 = (int)u._convW.Shape[0];
        return u;
    }

    public IEnumerable<Tensor> Weights() => new[] { _convW, _convB };

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor conv = SparseStructureDecoder.Conv3d(backend, x, _convW, _convB, _cOut8);
        int cOut = _cOut8 / 8, d = (int)x.Shape[2], hh = (int)x.Shape[3], ww = (int)x.Shape[4];
        Tensor o = new(new TensorShape(new long[] { x.Shape[0], cOut, d * 2, hh * 2, ww * 2 }), DType.F32);
        backend.PixelShuffle3d(o, conv, 2); conv.Dispose();
        return o;
    }
}
