using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi SEANet decoder, HF <c>transformers</c> layout (pre-fused <c>decoder.layers.N.conv</c> keys).</summary>
/// <remarks>Layout for ratios [8,6,5,4]: layer 0 = Conv1d(latent->mult*nf, k7); then per ratio an ELU + causal
/// ConvTranspose1d (k=2r, stride r, trim last k-r) + a MimiResnetBlock (ELU, Conv1d k3 dim->dim/2, ELU,
/// Conv1d k1 ->dim, residual); final ELU + Conv1d(nf->1, k3). All convs causal (left-pad k-1; transpose trims
/// the trailing k-stride). ELU alpha 1.0.</remarks>
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

    /// <summary>Streaming counterpart to <see cref="Forward"/>: same computation restructured so each causal conv's left context is <paramref name="state"/>'s carried real-audio tail, and each upsample stage's overlap-add tail carries across calls; z <c>[B, latentDim, t]</c> -> pcm <c>[B, 1, t*960]</c> for THIS chunk only.</summary>
    /// <remarks>Verified equivalent (to float rounding) to a single monolithic <see cref="Forward"/> call over the whole utterance by <c>MimiStreamParityTests</c>.</remarks>
    public Tensor ForwardStreaming(IBackend backend, Tensor z, int batch, int t, MimiSeanetDecoderStreamState state)
    {
        int mult = 1 << Ratios.Length;
        int dim = mult * NFilters;
        Tensor x = CausalConvStreaming(backend, state, slot: 0, z, _inW!, _inB!, batch, _latentDim, dim, t, 7);
        int curT = t;
        for (int s = 0; s < Ratios.Length; s++)
        {
            int inCh = dim, outCh = dim / 2, ratio = Ratios[s];
            Tensor e = new(x.Shape, DType.F32); backend.Elu(e, x, 1.0f); x.Dispose(); x = e;
            int tUp = curT * ratio;
            int overlap = ratio; // kernel(2r) - stride(r)
            Tensor rawFull = new(new TensorShape(batch, outCh, tUp + overlap), DType.F32);
            // bias: null here (not _upB[s]) — ConvTranspose1d's kernel broadcasts bias into EVERY output
            // position before the scatter-accumulate, so a biased rawFull's overlap region would carry bias
            // from BOTH the chunk that first produced it and the chunk it gets added into on the next call,
            // double-counting it there. Overlap-add on bias-free values instead, then add bias once to the
            // whole emitted chunk below — that matches what a single monolithic call produces exactly.
            backend.ConvTranspose1d(rawFull, x, _upW[s]!, null, stride: ratio, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            x.Dispose();
            x = state.SplitTransposeOutput(s, rawFull, outCh, tUp, overlap, batch);
            AddBiasInPlace(x, _upB[s]!, batch, outCh, tUp);
            curT = tUp;
            x = ResnetBlockStreaming(backend, state, slot: 1 + s, x, _r1W[s]!, _r1B[s]!, _r2W[s]!, _r2B[s]!, outCh, curT, batch);
            dim = outCh;
        }
        Tensor ef = new(x.Shape, DType.F32); backend.Elu(ef, x, 1.0f); x.Dispose(); x = ef;
        Tensor pcm = CausalConvStreaming(backend, state, slot: 5, x, _outW!, _outB!, batch, NFilters, 1, curT, 3);
        x.Dispose();
        return pcm;
    }

    private static Tensor ResnetBlockStreaming(IBackend backend, MimiSeanetDecoderStreamState state, int slot,
        Tensor x, Tensor c1w, Tensor c1b, Tensor c2w, Tensor c2b, int dim, int t, int batch)
    {
        int hidden = dim / 2;
        Tensor e1 = new(x.Shape, DType.F32); backend.Elu(e1, x, 1.0f);
        // History is e1's tail (this layer's real input), per the class doc — NOT x's tail and NOT the block's
        // residual output; getting this off by one layer would produce plausible-but-wrong audio.
        Tensor mid = CausalConvStreaming(backend, state, slot, e1, c1w, c1b, batch, dim, hidden, t, 3);
        e1.Dispose();
        Tensor e2 = new(mid.Shape, DType.F32); backend.Elu(e2, mid, 1.0f); mid.Dispose();
        // c2 is kernel=1 -> padLeft=0 already, genuinely stateless; CausalConv (not CausalConvStreaming) is
        // correct here unchanged, no history slot consumed.
        Tensor proj = CausalConv(backend, e2, c2w, c2b, batch, hidden, dim, t, 1);
        e2.Dispose();
        Tensor outp = new(x.Shape, DType.F32);
        backend.Add(outp, x, proj);
        proj.Dispose(); x.Dispose();
        return outp;
    }

    /// <summary>Left-pads with <paramref name="state"/>'s carried real tail (zero on the first chunk of an utterance) instead of the non-streaming path's virtual zero-pad.</summary>
    private static Tensor CausalConvStreaming(IBackend backend, MimiSeanetDecoderStreamState state, int slot,
        Tensor x, Tensor wt, Tensor b, int batch, int inDim, int outDim, int t, int kernel)
    {
        Tensor augmented = state.Augment(slot, x, inDim, t, kernel, batch);
        Tensor o = new(new TensorShape(batch, outDim, t), DType.F32);
        backend.Conv1d(o, augmented, wt, b, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        if (!ReferenceEquals(augmented, x)) augmented.Dispose();   // kernel=1 path returns x itself, caller owns x
        return o;
    }

    /// <summary>Adds per-output-channel <paramref name="bias"/> to every sample of <paramref name="t"/> <c>[B,C,T]</c> in place; companion to the bias-free streaming <c>ConvTranspose1d</c> call above (see that call site for why).</summary>
    private static void AddBiasInPlace(Tensor t, Tensor bias, int batch, int channels, int len)
    {
        float* tp = (float*)t.DataPointer;
        float* bp = (float*)bias.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long batchBase = (long)b * channels * len;
            for (int c = 0; c < channels; c++)
            {
                float bv = bp[c];
                long rowBase = batchBase + (long)c * len;
                for (int i = 0; i < len; i++) tp[rowBase + i] += bv;
            }
        }
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
