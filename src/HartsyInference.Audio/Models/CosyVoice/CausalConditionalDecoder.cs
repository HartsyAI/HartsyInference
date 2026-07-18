using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>The CFM velocity estimator — the CosyVoice 2 / Matcha-TTS <c>ConditionalDecoder</c> (causal),
/// reimplemented to match the real <c>flow.decoder.estimator</c> checkpoint. A 1-D U-Net with a single
/// resolution: pack <c>[x, μ, spk-broadcast, cond]</c> (4·80 = 320 ch) → 1 down block → 12 mid blocks →
/// 1 up block (skip concat) → final block → final 1×1 proj to 80 mel channels. Each down/mid/up block is one
/// <see cref="CausalResnet1D"/> (timestep-injected) followed by <see cref="MatchaTransformerBlock"/>×NumBlocks.
///
/// <para><b>Causal</b>: every conv is left-padded only (<c>(k-1, 0)</c>); the per-frame norm inside a block is
/// a LayerNorm over channels (not GroupNorm), activation is Mish. Self-attention is full over valid frames
/// (no causal attention mask for the non-streaming path). Verified ~1e-5 against the PyTorch reference.</para></summary>
public sealed unsafe class CausalConditionalDecoder : ICfmEstimator
{
    private readonly CosyVoiceFlowConfig _cfg;
    private readonly int _ch;          // model width (256)
    private readonly int _inCh;        // 4·mel (320)
    private readonly int _mel;

    private readonly CausalResnet1D _downResnet;
    private readonly MatchaTransformerBlock[] _downAttn;
    private Tensor? _downSampleW, _downSampleB;     // CausalConv1d k3
    private readonly CausalResnet1D[] _midResnet;
    private readonly MatchaTransformerBlock[][] _midAttn;
    private readonly CausalResnet1D _upResnet;
    private readonly MatchaTransformerBlock[] _upAttn;
    private Tensor? _upSampleW, _upSampleB;         // CausalConv1d k3

    private Tensor? _timeLin1W, _timeLin1B, _timeLin2W, _timeLin2B;
    private Tensor? _finalBlockConvW, _finalBlockConvB, _finalBlockLnW, _finalBlockLnB;
    private Tensor? _finalProjW, _finalProjB;

    public CausalConditionalDecoder(CosyVoiceFlowConfig cfg)
    {
        _cfg = cfg;
        _mel = cfg.MelBins;
        _ch = cfg.UnetChannels[0];
        _inCh = _mel * 4;
        int nb = cfg.NumBlocks, heads = cfg.NumHeads, hd = cfg.AttentionHeadDim;

        _downResnet = new CausalResnet1D(_inCh, _ch);
        _downAttn = MakeAttn(nb, _ch, heads, hd);
        _midResnet = new CausalResnet1D[cfg.NumMidBlocks];
        _midAttn = new MatchaTransformerBlock[cfg.NumMidBlocks][];
        for (int i = 0; i < cfg.NumMidBlocks; i++)
        {
            _midResnet[i] = new CausalResnet1D(_ch, _ch);
            _midAttn[i] = MakeAttn(nb, _ch, heads, hd);
        }
        _upResnet = new CausalResnet1D(2 * _ch, _ch);   // skip concat doubles channels
        _upAttn = MakeAttn(nb, _ch, heads, hd);
    }

    private static MatchaTransformerBlock[] MakeAttn(int n, int ch, int heads, int hd)
    {
        MatchaTransformerBlock[] a = new MatchaTransformerBlock[n];
        for (int i = 0; i < n; i++) a[i] = new MatchaTransformerBlock(ch, heads, hd);
        return a;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "estimator")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _timeLin1W = WhisperOps.EnsureF32(w[$"{p}time_mlp.linear_1.weight"]);
        _timeLin1B = WhisperOps.EnsureF32(w[$"{p}time_mlp.linear_1.bias"]);
        _timeLin2W = WhisperOps.EnsureF32(w[$"{p}time_mlp.linear_2.weight"]);
        _timeLin2B = WhisperOps.EnsureF32(w[$"{p}time_mlp.linear_2.bias"]);

        _downResnet.LoadWeights(w, $"{p}down_blocks.0.0");
        for (int i = 0; i < _downAttn.Length; i++) _downAttn[i].LoadWeights(w, $"{p}down_blocks.0.1.{i}");
        _downSampleW = WhisperOps.EnsureF32(w[$"{p}down_blocks.0.2.weight"]);
        _downSampleB = WhisperOps.EnsureF32(w[$"{p}down_blocks.0.2.bias"]);

        for (int i = 0; i < _midResnet.Length; i++)
        {
            _midResnet[i].LoadWeights(w, $"{p}mid_blocks.{i}.0");
            for (int j = 0; j < _midAttn[i].Length; j++) _midAttn[i][j].LoadWeights(w, $"{p}mid_blocks.{i}.1.{j}");
        }

        _upResnet.LoadWeights(w, $"{p}up_blocks.0.0");
        for (int i = 0; i < _upAttn.Length; i++) _upAttn[i].LoadWeights(w, $"{p}up_blocks.0.1.{i}");
        _upSampleW = WhisperOps.EnsureF32(w[$"{p}up_blocks.0.2.weight"]);
        _upSampleB = WhisperOps.EnsureF32(w[$"{p}up_blocks.0.2.bias"]);

        _finalBlockConvW = WhisperOps.EnsureF32(w[$"{p}final_block.block.0.weight"]);
        _finalBlockConvB = WhisperOps.EnsureF32(w[$"{p}final_block.block.0.bias"]);
        _finalBlockLnW = WhisperOps.EnsureF32(w[$"{p}final_block.block.2.weight"]);
        _finalBlockLnB = WhisperOps.EnsureF32(w[$"{p}final_block.block.2.bias"]);
        _finalProjW = WhisperOps.EnsureF32(w[$"{p}final_proj.weight"]);
        _finalProjB = WhisperOps.EnsureF32(w[$"{p}final_proj.bias"]);
    }

    // TODO(gpu-residency): this estimator is the CFM ODE hot path — it runs once per Euler step (×NumEulerSteps,
    // typically 10) per generated frame, so it dominates flow runtime. The helpers below (PackInput, ConcatChannels,
    // Mish, ExactGelu, the time-embedding sinusoid, and MatchaTransformerBlock.ToHeads/FromHeads) read tensor
    // data via host `(float*)DataPointer`, which forces a device→host sync on CUDA and breaks GPU residency
    // (same anti-pattern that made Ideogram4 slow before its dit_f32.ptx kernels). To go fully GPU-resident,
    // port these element-wise/reshape ops to PTX kernels / backend ops so the whole estimator stays on-device.
    public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond)
    {
        int tt = (int)x.Shape[2];
        Tensor timeEmb = TimeEmbedding(backend, t);          // [1, 1, time_embed_dim]

        // Pack [x, μ, spk-broadcast, cond] → [1, 320, T].
        Tensor h = PackInput(backend, x, mu, spk, cond, _mel, tt);

        // Down block: resnet → 4 transformers → save skip → causal-conv downsample.
        Tensor d = _downResnet.Forward(backend, h, timeEmb); h.Dispose();
        d = RunAttn(backend, _downAttn, d, tt);
        Tensor skip = new(d.Shape, DType.F32); backend.CopyTo(skip, d);
        Tensor downSampled = CausalConv(backend, d, _downSampleW!, _downSampleB!, _ch, _ch, tt); d.Dispose();
        Tensor cur = downSampled;

        // Mid blocks.
        for (int i = 0; i < _midResnet.Length; i++)
        {
            Tensor r = _midResnet[i].Forward(backend, cur, timeEmb); cur.Dispose();
            cur = RunAttn(backend, _midAttn[i], r, tt);
        }

        // Up block: concat([cur, skip]) on channels → resnet → 4 transformers → causal-conv upsample.
        Tensor catted = ConcatChannels(backend, cur, skip, _ch, _ch, tt); cur.Dispose(); skip.Dispose();
        Tensor u = _upResnet.Forward(backend, catted, timeEmb); catted.Dispose();
        u = RunAttn(backend, _upAttn, u, tt);
        Tensor upSampled = CausalConv(backend, u, _upSampleW!, _upSampleB!, _ch, _ch, tt); u.Dispose();
        timeEmb.Dispose();

        // final_block (CausalBlock1D) → final_proj (1×1).
        Tensor fb = CausalBlock(backend, upSampled, _finalBlockConvW!, _finalBlockConvB!, _finalBlockLnW!, _finalBlockLnB!, _ch, _ch, tt);
        upSampled.Dispose();
        Tensor outT = new(new TensorShape(1, _mel, tt), DType.F32);
        backend.Conv1d(outT, fb, _finalProjW!, _finalProjB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        fb.Dispose();
        return outT;
    }

    private Tensor RunAttn(IBackend backend, MatchaTransformerBlock[] blocks, Tensor chFirst, int t)
    {
        // [1, C, T] → [1, T, C] → blocks → back.
        Tensor seq = new(new TensorShape(1, t, _ch), DType.F32);
        backend.Transpose2D(seq, chFirst, _ch, t);
        chFirst.Dispose();
        for (int i = 0; i < blocks.Length; i++)
        {
            Tensor next = blocks[i].Forward(backend, seq, t);
            seq.Dispose();
            seq = next;
        }
        Tensor outT = new(new TensorShape(1, _ch, t), DType.F32);
        backend.Transpose2D(outT, seq, t, _ch);
        seq.Dispose();
        return outT;
    }

    /// <summary>SinusoidalPosEmb(in_channels=320, scale=1000) → Linear(320→D) → SiLU → Linear(D→D).</summary>
    private Tensor TimeEmbedding(IBackend backend, float t)
    {
        int dim = _inCh;                  // SinusoidalPosEmb dim = in_channels (320)
        int half = dim / 2;
        Tensor sinEmb = new(new TensorShape(1, 1, dim), DType.F32);
        float* sp = (float*)sinEmb.DataPointer;
        double logBase = Math.Log(10000.0) / (half - 1);
        for (int i = 0; i < half; i++)
        {
            double freq = Math.Exp(-logBase * i);
            double arg = 1000.0 * t * freq;       // scale = 1000
            sp[i] = (float)Math.Sin(arg);
            sp[half + i] = (float)Math.Cos(arg);
        }
        int dOut = (int)_timeLin1W!.Shape[0];
        Tensor h1 = WhisperOps.ProjectLinear(backend, sinEmb, _timeLin1W!, _timeLin1B, 1, 1, dim, dOut);
        sinEmb.Dispose();
        backend.Silu(h1, h1);
        Tensor h2 = WhisperOps.ProjectLinear(backend, h1, _timeLin2W!, _timeLin2B, 1, 1, dOut, dOut);
        h1.Dispose();
        return h2;
    }

    /// <summary>CausalConv1d: left-pad (k-1), conv stride 1, no right pad.</summary>
    internal static Tensor CausalConv(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int inCh, int outCh, int t)
    {
        int k = (int)weight.Shape[2];
        Tensor outT = new(new TensorShape(1, outCh, t), DType.F32);
        backend.Conv1d(outT, input, weight, bias, stride: 1, padLeft: k - 1, padRight: 0, dilation: 1, groups: 1);
        return outT;
    }

    /// <summary>CausalBlock1D: CausalConv1d → LayerNorm (over channels) → Mish, all channels-first.</summary>
    internal static Tensor CausalBlock(IBackend backend, Tensor x, Tensor convW, Tensor convB, Tensor lnW, Tensor lnB, int inCh, int outCh, int t)
    {
        Tensor conv = CausalConv(backend, x, convW, convB, inCh, outCh, t);
        // LayerNorm over channels: transpose to [1, T, C], LN, transpose back.
        Tensor seq = new(new TensorShape(1, t, outCh), DType.F32);
        backend.Transpose2D(seq, conv, outCh, t);
        conv.Dispose();
        Tensor normed = new(seq.Shape, DType.F32);
        backend.LayerNorm(normed, seq, lnW, lnB, 1e-5f);
        seq.Dispose();
        Tensor outT = new(new TensorShape(1, outCh, t), DType.F32);
        backend.Transpose2D(outT, normed, t, outCh);
        normed.Dispose();
        backend.Mish(outT, outT);
        return outT;
    }

    /// <summary>Packs <c>[x, μ, spk-broadcast, cond]</c> into <c>[1, 4·mel, T]</c> on the device: the speaker
    /// vector <c>[1, mel, 1]</c> is time-broadcast via <see cref="IBackend.RepeatTime"/>, then all four
    /// channel-first blocks are concatenated with <see cref="IBackend.Concat"/> — no host copies.</summary>
    private static Tensor PackInput(IBackend backend, Tensor x, Tensor mu, Tensor spk, Tensor cond, int mel, int t)
    {
        Tensor spkBroad = new(new TensorShape(1, mel, t), DType.F32);
        backend.RepeatTime(spkBroad, spk, t);
        Tensor packed = new(new TensorShape(1, mel * 4, t), DType.F32);
        backend.Concat(packed, [x, mu, spkBroad, cond], dim: 1);
        spkBroad.Dispose();
        return packed;
    }

    private static Tensor ConcatChannels(IBackend backend, Tensor a, Tensor b, int aCh, int bCh, int t)
    {
        Tensor outT = new(new TensorShape(1, aCh + bCh, t), DType.F32);
        backend.Concat(outT, [a, b], dim: 1);
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core =
        [
            _timeLin1W, _timeLin1B, _timeLin2W, _timeLin2B, _downSampleW, _downSampleB, _upSampleW, _upSampleB,
            _finalBlockConvW, _finalBlockConvB, _finalBlockLnW, _finalBlockLnB, _finalProjW, _finalProjB,
        ];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (Tensor t in _downResnet.EnumerateWeights()) yield return t;
        foreach (MatchaTransformerBlock b in _downAttn) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        for (int i = 0; i < _midResnet.Length; i++)
        {
            foreach (Tensor t in _midResnet[i].EnumerateWeights()) yield return t;
            foreach (MatchaTransformerBlock b in _midAttn[i]) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        }
        foreach (Tensor t in _upResnet.EnumerateWeights()) yield return t;
        foreach (MatchaTransformerBlock b in _upAttn) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }
}

/// <summary>CausalResnetBlock1D: <c>h = block1(x); h += Mish(time_emb)·mlp; h = block2(h);
/// out = h + res_conv(x)</c>. block1/block2 are CausalConv1d → LayerNorm → Mish.</summary>
internal sealed unsafe class CausalResnet1D
{
    private readonly int _inCh, _outCh;
    private Tensor? _b1ConvW, _b1ConvB, _b1LnW, _b1LnB;
    private Tensor? _b2ConvW, _b2ConvB, _b2LnW, _b2LnB;
    private Tensor? _mlpW, _mlpB, _resConvW, _resConvB;

    public CausalResnet1D(int inCh, int outCh) { _inCh = inCh; _outCh = outCh; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _b1ConvW = WhisperOps.EnsureF32(w[$"{prefix}.block1.block.0.weight"]);
        _b1ConvB = WhisperOps.EnsureF32(w[$"{prefix}.block1.block.0.bias"]);
        _b1LnW = WhisperOps.EnsureF32(w[$"{prefix}.block1.block.2.weight"]);
        _b1LnB = WhisperOps.EnsureF32(w[$"{prefix}.block1.block.2.bias"]);
        _b2ConvW = WhisperOps.EnsureF32(w[$"{prefix}.block2.block.0.weight"]);
        _b2ConvB = WhisperOps.EnsureF32(w[$"{prefix}.block2.block.0.bias"]);
        _b2LnW = WhisperOps.EnsureF32(w[$"{prefix}.block2.block.2.weight"]);
        _b2LnB = WhisperOps.EnsureF32(w[$"{prefix}.block2.block.2.bias"]);
        _mlpW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.1.weight"]);
        _mlpB = WhisperOps.EnsureF32(w[$"{prefix}.mlp.1.bias"]);
        _resConvW = WhisperOps.EnsureF32(w[$"{prefix}.res_conv.weight"]);   // Conv1d k1
        _resConvB = WhisperOps.EnsureF32(w[$"{prefix}.res_conv.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor timeEmb)
    {
        int t = (int)x.Shape[2];
        Tensor h = CausalConditionalDecoder.CausalBlock(backend, x, _b1ConvW!, _b1ConvB!, _b1LnW!, _b1LnB!, _inCh, _outCh, t);

        // mlp = Mish(time_emb) → Linear(D→outCh); add broadcast over T (per-channel bias over the time axis).
        int dIn = (int)timeEmb.Shape[2];
        Tensor te = new(timeEmb.Shape, DType.F32);
        backend.Mish(te, timeEmb);
        Tensor tproj = WhisperOps.ProjectLinear(backend, te, _mlpW!, _mlpB, 1, 1, dIn, _outCh);
        te.Dispose();
        backend.BroadcastAdd(h, tproj, _outCh, t);
        tproj.Dispose();

        Tensor h2 = CausalConditionalDecoder.CausalBlock(backend, h, _b2ConvW!, _b2ConvB!, _b2LnW!, _b2LnB!, _outCh, _outCh, t);
        h.Dispose();

        // Residual: res_conv (1×1) on x.
        Tensor res = new(new TensorShape(1, _outCh, t), DType.F32);
        backend.Conv1d(res, x, _resConvW!, _resConvB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        backend.Add(h2, h2, res);
        res.Dispose();
        return h2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_b1ConvW, _b1ConvB, _b1LnW, _b1LnB, _b2ConvW, _b2ConvB, _b2LnW, _b2LnB, _mlpW, _mlpB, _resConvW, _resConvB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>Matcha/diffusers BasicTransformerBlock (self-attention only): pre-LN self-attn (no qkv bias,
/// out proj has bias, scale 1/√head_dim) + residual, then pre-LN GELU feed-forward + residual. Operates on
/// <c>[1, T, C]</c>.</summary>
internal sealed unsafe class MatchaTransformerBlock
{
    private readonly int _ch, _heads, _headDim, _inner;
    private Tensor? _norm1W, _norm1B, _qW, _kW, _vW, _oW, _oB;
    private Tensor? _norm3W, _norm3B, _ff0W, _ff0B, _ff2W, _ff2B;

    public MatchaTransformerBlock(int ch, int heads, int headDim)
    {
        _ch = ch; _heads = heads; _headDim = headDim; _inner = heads * headDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _norm1W = WhisperOps.EnsureF32(w[$"{prefix}.norm1.weight"]);
        _norm1B = WhisperOps.EnsureF32(w[$"{prefix}.norm1.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.attn1.to_q.weight"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.attn1.to_k.weight"]);
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.attn1.to_v.weight"]);
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.attn1.to_out.0.weight"]);
        _oB = WhisperOps.EnsureF32(w[$"{prefix}.attn1.to_out.0.bias"]);
        _norm3W = WhisperOps.EnsureF32(w[$"{prefix}.norm3.weight"]);
        _norm3B = WhisperOps.EnsureF32(w[$"{prefix}.norm3.bias"]);
        _ff0W = WhisperOps.EnsureF32(w[$"{prefix}.ff.net.0.proj.weight"]);
        _ff0B = WhisperOps.EnsureF32(w[$"{prefix}.ff.net.0.proj.bias"]);
        _ff2W = WhisperOps.EnsureF32(w[$"{prefix}.ff.net.2.weight"]);
        _ff2B = WhisperOps.EnsureF32(w[$"{prefix}.ff.net.2.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor seq, int t)
    {
        int c = _ch;
        // Self-attention sub-layer.
        Tensor normed = new(seq.Shape, DType.F32);
        backend.LayerNorm(normed, seq, _norm1W!, _norm1B!, 1e-5f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, null, 1, t, c, _inner);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, null, 1, t, c, _inner);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, null, 1, t, c, _inner);
        normed.Dispose();

        // [1, T, inner] → [1, H, T, d] for SDPA (contiguous inner = H·d, so this is a plain 0213 permute).
        Tensor qH = new(new TensorShape(1, _heads, t, _headDim), DType.F32); backend.Permute0213(qH, q, t, _heads, _headDim);
        Tensor kH = new(new TensorShape(1, _heads, t, _headDim), DType.F32); backend.Permute0213(kH, k, t, _heads, _headDim);
        Tensor vH = new(new TensorShape(1, _heads, t, _headDim), DType.F32); backend.Permute0213(vH, v, t, _heads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attn = new(new TensorShape(1, _heads, t, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, mask: null, 1f / MathF.Sqrt(_headDim));
        qH.Dispose(); kH.Dispose(); vH.Dispose();
        Tensor merged = new(new TensorShape(1, t, _inner), DType.F32);   // [1, H, T, d] → [1, T, inner]
        backend.Permute0213(merged, attn, _heads, t, _headDim);
        attn.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, merged, _oW!, _oB, 1, t, _inner, c);
        merged.Dispose();
        Tensor x = new(seq.Shape, DType.F32);
        backend.Add(x, seq, o); o.Dispose();

        // Feed-forward sub-layer (diffusers GELU = exact erf, not tanh-approx).
        Tensor n3 = new(x.Shape, DType.F32);
        backend.LayerNorm(n3, x, _norm3W!, _norm3B!, 1e-5f);
        int ffDim = (int)_ff0W!.Shape[0];
        Tensor f0 = WhisperOps.ProjectLinear(backend, n3, _ff0W!, _ff0B, 1, t, c, ffDim);
        n3.Dispose();
        backend.GeluErf(f0, f0);
        Tensor f2 = WhisperOps.ProjectLinear(backend, f0, _ff2W!, _ff2B, 1, t, ffDim, c);
        f0.Dispose();
        backend.Add(x, x, f2); f2.Dispose();
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1W, _norm1B, _qW, _kW, _vW, _oW, _oB, _norm3W, _norm3B, _ff0W, _ff0B, _ff2W, _ff2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
