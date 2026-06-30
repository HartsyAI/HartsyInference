using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi SEANet decoder, HF <c>transformers</c> layout (pre-fused <c>decoder.layers.N.conv</c> keys).
/// Layout for ratios [8,6,5,4]: layer 0 = Conv1d(latent->mult*nf, k7); then per ratio an ELU + causal
/// ConvTranspose1d (k=2r, stride r, trim last k-r) + a MimiResnetBlock (ELU, Conv1d k3 dim->dim/2, ELU,
/// Conv1d k1 ->dim, residual); final ELU + Conv1d(nf->1, k3). All convs causal (left-pad k-1; transpose trims
/// the trailing k-stride). ELU alpha 1.0.</summary>
internal sealed unsafe class MimiSeanetDecoder
{
    private static readonly int[] Ratios = [8, 6, 5, 4];
    private const int NFilters = 64;
    private readonly string _p;
    private readonly int _latentDim;

    private Tensor? _inW, _inB;
    private readonly Tensor?[] _upW = new Tensor?[4], _upB = new Tensor?[4];
    private readonly Tensor?[] _r1W = new Tensor?[4], _r1B = new Tensor?[4];
    private readonly Tensor?[] _r2W = new Tensor?[4], _r2B = new Tensor?[4];
    private Tensor? _outW, _outB;

    public MimiSeanetDecoder(string prefix, int latentDim)
    {
        _p = prefix; _latentDim = latentDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inW = G(w, $"{_p}.layers.0.conv.weight"); _inB = G(w, $"{_p}.layers.0.conv.bias");
        for (int s = 0; s < 4; s++)
        {
            int up = 2 + 3 * s, rb = 3 + 3 * s;
            _upW[s] = G(w, $"{_p}.layers.{up}.conv.weight"); _upB[s] = G(w, $"{_p}.layers.{up}.conv.bias");
            _r1W[s] = G(w, $"{_p}.layers.{rb}.block.1.conv.weight"); _r1B[s] = G(w, $"{_p}.layers.{rb}.block.1.conv.bias");
            _r2W[s] = G(w, $"{_p}.layers.{rb}.block.3.conv.weight"); _r2B[s] = G(w, $"{_p}.layers.{rb}.block.3.conv.bias");
        }
        _outW = G(w, $"{_p}.layers.14.conv.weight"); _outB = G(w, $"{_p}.layers.14.conv.bias");
    }

    private static Tensor G(IReadOnlyDictionary<string, Tensor> w, string k) => WhisperOps.EnsureF32(w[k]);

    /// <summary>z <c>[B, latentDim, T]</c> -> pcm <c>[B, 1, T*960]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor z, int batch, int t)
    {
        int mult = 1 << Ratios.Length;           // 16
        int dim = mult * NFilters;               // 1024
        Tensor x = CausalConv(backend, z, _inW!, _inB!, batch, _latentDim, dim, t, 7);
        int curT = t;
        for (int s = 0; s < Ratios.Length; s++)
        {
            int inCh = dim, outCh = dim / 2, ratio = Ratios[s];
            Tensor e = new(x.Shape, DType.F32); backend.Elu(e, x, 1.0f); x.Dispose(); x = e;
            int tUp = curT * ratio;
            Tensor up = new(new TensorShape(batch, outCh, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _upW[s]!, _upB[s], stride: ratio, padLeft: 0, padRight: 2 * ratio - ratio, dilation: 1, groups: 1);
            x.Dispose(); x = up; curT = tUp;
            x = ResnetBlock(backend, x, _r1W[s]!, _r1B[s]!, _r2W[s]!, _r2B[s]!, outCh, curT, batch);
            dim = outCh;
        }
        Tensor ef = new(x.Shape, DType.F32); backend.Elu(ef, x, 1.0f); x.Dispose(); x = ef;
        Tensor pcm = CausalConv(backend, x, _outW!, _outB!, batch, NFilters, 1, curT, 3);
        x.Dispose();
        return pcm;
    }

    private static Tensor ResnetBlock(IBackend backend, Tensor x, Tensor c1w, Tensor c1b, Tensor c2w, Tensor c2b, int dim, int t, int batch)
    {
        int hidden = dim / 2;   // compress = 2
        Tensor e1 = new(x.Shape, DType.F32); backend.Elu(e1, x, 1.0f);
        Tensor mid = CausalConv(backend, e1, c1w, c1b, batch, dim, hidden, t, 3);
        e1.Dispose();
        Tensor e2 = new(mid.Shape, DType.F32); backend.Elu(e2, mid, 1.0f); mid.Dispose();
        Tensor proj = CausalConv(backend, e2, c2w, c2b, batch, hidden, dim, t, 1);
        e2.Dispose();
        Tensor outp = new(x.Shape, DType.F32);
        backend.Add(outp, x, proj);
        proj.Dispose(); x.Dispose();
        return outp;
    }

    private static Tensor CausalConv(IBackend backend, Tensor x, Tensor wt, Tensor b, int batch, int inDim, int outDim, int t, int kernel)
    {
        int padLeft = kernel - 1;
        Tensor o = new(new TensorShape(batch, outDim, t), DType.F32);
        backend.Conv1d(o, x, wt, b, stride: 1, padLeft: padLeft, padRight: 0, dilation: 1, groups: 1);
        return o;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inW, _inB, _outW, _outB, .. _upW, .. _upB, .. _r1W, .. _r1B, .. _r2W, .. _r2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
