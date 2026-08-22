using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi SEANet ENCODER, HF <c>transformers</c> layout (pre-fused <c>encoder.layers.N.conv</c> keys) — the mirror of <see cref="MimiSeanetDecoder"/>; verified end-to-end by Kyutai-STT transcription on the real weights.</summary>
/// <remarks>For ratios [8,6,5,4] the encoder applies them in REVERSE
/// (downsampling by 4,5,6,8): layer 0 = causal Conv1d(1->nf, k7); then per ratio a <see cref="MimiSeanetDecoder"/>-style
/// MimiResnetBlock (ELU, Conv1d k3 dim->dim/2, ELU, Conv1d k1 ->dim, residual) + an ELU + a causal strided
/// Conv1d (k=2r, stride r, dim->2·dim); final ELU + Conv1d(mult·nf->latent, k3). All convs causal
/// (left-pad k-stride). ELU alpha 1.0.</remarks>
internal sealed unsafe class MimiSeanetEncoder
{
    // Encoder downsampling ratios = the config EncoderRates REVERSED (checkpoint kernels 8/10/12/16 = 2·[4,5,6,8]).
    private static readonly int[] Ratios = [4, 5, 6, 8];
    private const int NFilters = 64;
    private readonly string _p;
    private readonly int _latentDim;

    private Tensor? _inW, _inB;
    private readonly Tensor?[] _downW = new Tensor?[4], _downB = new Tensor?[4];
    private readonly Tensor?[] _r1W = new Tensor?[4], _r1B = new Tensor?[4];
    private readonly Tensor?[] _r2W = new Tensor?[4], _r2B = new Tensor?[4];
    private Tensor? _outW, _outB;

    public MimiSeanetEncoder(string prefix, int latentDim)
    {
        _p = prefix; _latentDim = latentDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inW = G(w, $"{_p}.layers.0.conv.weight"); _inB = G(w, $"{_p}.layers.0.conv.bias");
        for (int s = 0; s < 4; s++)
        {
            int rb = 1 + 3 * s, down = 3 + 3 * s;
            _r1W[s] = G(w, $"{_p}.layers.{rb}.block.1.conv.weight"); _r1B[s] = G(w, $"{_p}.layers.{rb}.block.1.conv.bias");
            _r2W[s] = G(w, $"{_p}.layers.{rb}.block.3.conv.weight"); _r2B[s] = G(w, $"{_p}.layers.{rb}.block.3.conv.bias");
            _downW[s] = G(w, $"{_p}.layers.{down}.conv.weight"); _downB[s] = G(w, $"{_p}.layers.{down}.conv.bias");
        }
        _outW = G(w, $"{_p}.layers.14.conv.weight"); _outB = G(w, $"{_p}.layers.14.conv.bias");
    }

    private static Tensor G(IReadOnlyDictionary<string, Tensor> w, string k) => WhisperOps.EnsureF32(w[k]);

    /// <summary>pcm <c>[B, 1, L]</c> -> latent <c>[B, latentDim, L/960]</c> (25 Hz for 24 kHz).</summary>
    public Tensor Forward(IBackend backend, Tensor pcm, int batch, int lPcm)
    {
        int dim = NFilters;                          // 64
        Tensor x = CausalConv(backend, pcm, _inW!, _inB!, batch, 1, dim, lPcm, 7, 1);
        int curT = lPcm;
        for (int s = 0; s < Ratios.Length; s++)
        {
            int ratio = Ratios[s];
            x = ResnetBlock(backend, x, _r1W[s]!, _r1B[s]!, _r2W[s]!, _r2B[s]!, dim, curT, batch);
            Tensor e = new(x.Shape, DType.F32); backend.Elu(e, x, 1.0f); x.Dispose(); x = e;
            int outCh = dim * 2, tDown = curT / ratio;
            x = CausalConvOut(backend, x, _downW[s]!, _downB[s]!, batch, dim, outCh, curT, tDown, 2 * ratio, ratio);
            dim = outCh; curT = tDown;
        }
        Tensor ef = new(x.Shape, DType.F32); backend.Elu(ef, x, 1.0f); x.Dispose(); x = ef;
        Tensor outp = CausalConv(backend, x, _outW!, _outB!, batch, dim, _latentDim, curT, 3, 1);
        x.Dispose();
        return outp;
    }

    private static Tensor ResnetBlock(IBackend backend, Tensor x, Tensor c1w, Tensor c1b, Tensor c2w, Tensor c2b, int dim, int t, int batch)
    {
        int hidden = dim / 2;   // compress = 2
        Tensor e1 = new(x.Shape, DType.F32); backend.Elu(e1, x, 1.0f);
        Tensor mid = CausalConv(backend, e1, c1w, c1b, batch, dim, hidden, t, 3, 1);
        e1.Dispose();
        Tensor e2 = new(mid.Shape, DType.F32); backend.Elu(e2, mid, 1.0f); mid.Dispose();
        Tensor proj = CausalConv(backend, e2, c2w, c2b, batch, hidden, dim, t, 1, 1);
        e2.Dispose();
        Tensor outp = new(x.Shape, DType.F32);
        backend.Add(outp, x, proj);
        proj.Dispose(); x.Dispose();
        return outp;
    }

    /// <summary>Stride-1 causal conv (left-pad kernel-1, output length preserved).</summary>
    private static Tensor CausalConv(IBackend backend, Tensor x, Tensor wt, Tensor b, int batch, int inDim, int outDim, int t, int kernel, int stride)
    {
        Tensor o = new(new TensorShape(batch, outDim, t), DType.F32);
        backend.Conv1d(o, x, wt, b, stride: stride, padLeft: kernel - stride, padRight: 0, dilation: 1, groups: 1);
        return o;
    }

    /// <summary>Strided causal downsampling conv: left-pad (kernel-stride) so the output length is exactly <paramref name="tOut"/> = t/stride (causal, no look-ahead).</summary>
    private static Tensor CausalConvOut(IBackend backend, Tensor x, Tensor wt, Tensor b, int batch, int inDim, int outDim, int t, int tOut, int kernel, int stride)
    {
        Tensor o = new(new TensorShape(batch, outDim, tOut), DType.F32);
        backend.Conv1d(o, x, wt, b, stride: stride, padLeft: kernel - stride, padRight: 0, dilation: 1, groups: 1);
        x.Dispose();
        return o;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_inW, _inB, _outW, _outB, .. _downW, .. _downB, .. _r1W, .. _r1B, .. _r2W, .. _r2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
