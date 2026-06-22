using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Qwen3-TTS custom codec DECODER (net-new time-domain Snake/ConvNeXt causal-conv vocoder; NOT iSTFT).
/// Pipeline, channel-consistent so the chain hits 1920× total upsample:
/// <list type="number">
///   <item>Split-RVQ dequant: 1 semantic codebook (4096, dim 256) + 15 acoustic (2048, dim 512), residual-summed
///         and output-projected to the <c>latentDim</c> (512) sequence.</item>
///   <item>8-layer causal sliding-window transformer (hidden 512, 16 heads, head_dim 64, RoPE θ=10000,
///         window 72, RMSNorm, LayerScale 0.01).</item>
///   <item><c>pre_conv</c> (512→1024 k3 causal).</item>
///   <item>upsample [2,2]: each stage = causal transposed conv (stride 2) + <see cref="ConvNeXtBlock"/>.</item>
///   <item>decoder: causal conv 1024→1536 k7 + 4 <see cref="DecoderBlock"/>s over upsample_rates [8,5,4,3]
///         (SnakeBeta + transposed conv + 3 residual units).</item>
///   <item>final SnakeBeta + causal conv → 1 channel k7 + clamp.</item>
/// </list>
/// <c>2·2·8·5·4·3 = 1920</c> → 12.5 Hz frames to 24 kHz mono.
///
/// <para>Note: the research doc lists <c>pre_conv</c> before the transformer; the channel math (decoder input
/// 1024, transformer hidden 512) forces transformer → pre_conv ordering. The weight-key layout is the
/// checkpoint-gated reconcile item.</para></summary>
public sealed unsafe class Qwen3TtsVocoder : IDisposable
{
    private readonly Qwen3TtsVocoderConfig _cfg;

    // Split-RVQ.
    private Tensor? _semanticCodebook;          // [4096, 256]
    private Tensor? _semanticInProj;            // [latentDim, 256]
    private readonly Tensor?[] _acousticCodebook;   // [15] each [2048, 512]
    private readonly Tensor?[] _acousticInProj;     // [15] each [latentDim, 512]

    // Transformer.
    private readonly TransformerLayer[] _layers;
    private float[]? _cos, _sin;

    // pre_conv.
    private Tensor? _preConvW, _preConvB;

    // upsample [2,2].
    private readonly UpsampleStage[] _upsample;

    // decoder.
    private Tensor? _decInW, _decInB;
    private readonly DecoderBlock[] _decoderBlocks;
    private Tensor? _finalAlpha, _finalBeta, _finalConvW, _finalConvB;

    private int _disposed;

    public Qwen3TtsVocoderConfig Config => _cfg;
    public int SampleRate => _cfg.SampleRate;

    public Qwen3TtsVocoder(Qwen3TtsVocoderConfig cfg)
    {
        _cfg = cfg;
        _acousticCodebook = new Tensor?[cfg.AcousticCodebooks];
        _acousticInProj = new Tensor?[cfg.AcousticCodebooks];
        _layers = new TransformerLayer[cfg.TransformerLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new TransformerLayer(cfg);
        _upsample = new UpsampleStage[cfg.ConvNeXtUpsampleRates.Count];
        for (int i = 0; i < _upsample.Length; i++) _upsample[i] = new UpsampleStage(cfg.ConvNeXtDim, cfg.ConvNeXtUpsampleRates[i]);
        _decoderBlocks = new DecoderBlock[cfg.UpsampleRates.Count];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "vocoder")
    {
        _semanticCodebook = WhisperOps.EnsureF32(w[$"{prefix}.quantizer.semantic.codebook"]);
        _semanticInProj = WhisperOps.EnsureF32(w[$"{prefix}.quantizer.semantic.in_proj.weight"]);
        for (int k = 0; k < _cfg.AcousticCodebooks; k++)
        {
            _acousticCodebook[k] = WhisperOps.EnsureF32(w[$"{prefix}.quantizer.acoustic.{k}.codebook"]);
            _acousticInProj[k] = WhisperOps.EnsureF32(w[$"{prefix}.quantizer.acoustic.{k}.in_proj.weight"]);
        }
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.transformer.layers.{i}");
        (_cos, _sin) = Moonshine.RotaryEmbedding.GetTables(_cfg.TransformerHeadDim, _cfg.TransformerRopeTheta, 8_192);

        _preConvW = WhisperOps.EnsureF32(w[$"{prefix}.pre_conv.weight"]);
        _preConvB = TryGet(w, $"{prefix}.pre_conv.bias");

        for (int i = 0; i < _upsample.Length; i++) _upsample[i].LoadWeights(w, $"{prefix}.upsample.{i}");

        _decInW = WhisperOps.EnsureF32(w[$"{prefix}.decoder.in_conv.weight"]);
        _decInB = TryGet(w, $"{prefix}.decoder.in_conv.bias");
        int ch = _cfg.DecoderInChannels;
        for (int i = 0; i < _decoderBlocks.Length; i++)
        {
            int outCh = ch / 2;
            _decoderBlocks[i] = new DecoderBlock(ch, outCh, _cfg.UpsampleRates[i], _cfg.ResidualKernel, _cfg.ResidualDilations);
            _decoderBlocks[i].LoadWeights(w, $"{prefix}.decoder.blocks.{i}");
            ch = outCh;
        }
        _finalAlpha = WhisperOps.EnsureF32(w[$"{prefix}.decoder.final_snake.alpha"]);
        _finalBeta = TryGet(w, $"{prefix}.decoder.final_snake.beta");
        _finalConvW = WhisperOps.EnsureF32(w[$"{prefix}.decoder.out_conv.weight"]);
        _finalConvB = TryGet(w, $"{prefix}.decoder.out_conv.bias");
    }

    /// <summary>Decodes a <c>[16, T]</c> codebook grid (row 0 = semantic, rows 1..15 = acoustic) into 24 kHz
    /// mono PCM of length <c>T · 1920</c>.</summary>
    public float[] Decode(IBackend backend, int[,] codeGrid)
    {
        int t = codeGrid.GetLength(1);
        Tensor latent = Dequantize(backend, codeGrid, t);     // [1, latentDim, T]

        // Channels-last for the transformer.
        Tensor seq = new(new TensorShape(1, t, _cfg.TransformerDim), DType.F32);
        backend.Transpose2D(seq, latent, _cfg.TransformerDim, t);
        latent.Dispose();
        Tensor? mask = t > 1 ? BuildSlidingCausalMask(t, _cfg.SlidingWindow) : null;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, seq, t, mask, _cos!, _sin!);
            seq.Dispose();
            seq = next;
        }
        mask?.Dispose();
        Tensor afterTf = new(new TensorShape(1, _cfg.TransformerDim, t), DType.F32);
        backend.Transpose2D(afterTf, seq, t, _cfg.TransformerDim);
        seq.Dispose();

        // pre_conv 512 → 1024 (causal k3).
        Tensor pre = CausalConv(backend, afterTf, _preConvW!, _preConvB, _cfg.ConvNeXtDim, t, _cfg.PreConvKernel, 1);
        afterTf.Dispose();

        // upsample [2,2].
        Tensor up = pre;
        for (int i = 0; i < _upsample.Length; i++)
        {
            Tensor next = _upsample[i].Forward(backend, up);
            up.Dispose();
            up = next;
        }

        // decoder in_conv 1024 → 1536.
        int upT = (int)up.Shape[2];
        Tensor dec = CausalConv(backend, up, _decInW!, _decInB, _cfg.DecoderInChannels, upT, _cfg.DecoderInKernel, 1);
        up.Dispose();
        for (int i = 0; i < _decoderBlocks.Length; i++)
        {
            Tensor next = _decoderBlocks[i].Forward(backend, dec);
            dec.Dispose();
            dec = next;
        }

        // final SnakeBeta + out_conv → 1ch + clamp.
        Tensor act = new(dec.Shape, DType.F32);
        backend.Snake(act, dec, _finalAlpha!, _finalBeta);
        dec.Dispose();
        int finalT = (int)act.Shape[2];
        Tensor wave = CausalConv(backend, act, _finalConvW!, _finalConvB, 1, finalT, _cfg.OutKernel, 1);
        act.Dispose();
        Tensor clamped = new(wave.Shape, DType.F32);
        backend.Clamp(clamped, wave, -1f, 1f);
        wave.Dispose();

        int n = (int)clamped.Shape[2];
        float[] pcm = new float[n];
        float* cp = (float*)clamped.DataPointer;
        for (int i = 0; i < n; i++) pcm[i] = cp[i];
        clamped.Dispose();
        return pcm;
    }

    /// <summary>Split-RVQ dequant: semantic embed + projection, plus residual-summed acoustic embeds + projections,
    /// producing the <c>[1, latentDim, T]</c> latent.</summary>
    private Tensor Dequantize(IBackend backend, int[,] codeGrid, int t)
    {
        int latent = _cfg.LatentDim;
        Tensor outT = new(new TensorShape(1, latent, t), DType.F32);     // zero-initialized accumulator
        // Semantic (row 0): embed [256] then project [256→latent], accumulate channels-first.
        AddProjectedEmbed(backend, _semanticCodebook!, _semanticInProj!, _cfg.SemanticCodebookDim, latent, codeGrid, 0, t, outT);
        // Acoustic (rows 1..15).
        for (int k = 0; k < _cfg.AcousticCodebooks; k++)
            AddProjectedEmbed(backend, _acousticCodebook[k]!, _acousticInProj[k]!, _cfg.AcousticCodebookDim, latent, codeGrid, k + 1, t, outT);
        return outT;
    }

    private static void AddProjectedEmbed(IBackend backend, Tensor codebook, Tensor inProj, int codebookDim, int latent,
        int[,] codeGrid, int row, int t, Tensor accumCf)
    {
        // Gather per-frame embeddings into [1, t, codebookDim] (channels-last for the projection matmul).
        Tensor gathered = new(new TensorShape(1, t, codebookDim), DType.F32);
        float* gp = (float*)gathered.DataPointer;
        float* cb = (float*)codebook.DataPointer;
        int size = (int)codebook.Shape[0];
        for (int s = 0; s < t; s++)
        {
            int code = codeGrid[row, s];
            int clamped = (uint)code < (uint)size ? code : 0;
            Buffer.MemoryCopy(cb + (long)clamped * codebookDim, gp + (long)s * codebookDim, codebookDim * 4, codebookDim * 4);
        }
        Tensor projected = WhisperOps.ProjectLinear(backend, gathered, inProj, null, 1, t, codebookDim, latent);
        gathered.Dispose();

        // Add channels-last [1,t,latent] into channels-first accumulator [1,latent,t].
        float* pp = (float*)projected.DataPointer;
        float* ap = (float*)accumCf.DataPointer;
        for (int s = 0; s < t; s++)
            for (int c = 0; c < latent; c++)
                ap[(long)c * t + s] += pp[(long)s * latent + c];
        projected.Dispose();
    }

    /// <summary>Causal Conv1d: pads <c>(k-1)*dilation</c> on the left only, keeping <c>T_out == T_in</c>.</summary>
    internal static Tensor CausalConv(IBackend backend, Tensor x, Tensor weight, Tensor? bias, int outCh, int t, int k, int dilation)
    {
        int padL = (k - 1) * dilation;
        Tensor outT = new(new TensorShape(1, outCh, t), DType.F32);
        backend.Conv1d(outT, x, weight, bias, 1, padL, 0, dilation, 1);
        return outT;
    }

    /// <summary>Bidirectional sliding-window causal mask [1,1,T,T]: position q attends to k in
    /// <c>[q-window, q]</c> (causal + bounded look-back).</summary>
    internal static Tensor BuildSlidingCausalMask(int t, int window)
    {
        Tensor mask = new(new TensorShape(1, 1, t, t), DType.F32);
        float* p = (float*)mask.DataPointer;
        const float NegInf = -1e30f;
        for (int q = 0; q < t; q++)
            for (int k = 0; k < t; k++)
                p[(long)q * t + k] = (k <= q && q - k <= window) ? 0f : NegInf;
        return mask;
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_semanticCodebook, _semanticInProj, _preConvW, _preConvB, _decInW, _decInB, _finalAlpha, _finalBeta, _finalConvW, _finalConvB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        for (int k = 0; k < _cfg.AcousticCodebooks; k++)
        {
            if (_acousticCodebook[k] is not null) yield return _acousticCodebook[k]!;
            if (_acousticInProj[k] is not null) yield return _acousticInProj[k]!;
        }
        foreach (TransformerLayer l in _layers) foreach (Tensor tt in l.EnumerateWeights()) yield return tt;
        foreach (UpsampleStage u in _upsample) foreach (Tensor tt in u.EnumerateWeights()) yield return tt;
        foreach (DecoderBlock d in _decoderBlocks) foreach (Tensor tt in d.EnumerateWeights()) yield return tt;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    /// <summary>One causal sliding-window transformer layer: RMSNorm → MHA (RoPE, sliding-window mask) → LayerScale
    /// residual → RMSNorm → SwiGLU → LayerScale residual.</summary>
    private sealed class TransformerLayer
    {
        private readonly Qwen3TtsVocoderConfig _cfg;
        private Tensor? _inNorm, _postNorm, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _gateW, _upW, _downW, _ls1, _ls2;

        public TransformerLayer(Qwen3TtsVocoderConfig cfg) => _cfg = cfg;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _inNorm = WhisperOps.EnsureF32(w[$"{p}.input_layernorm.weight"]);
            _postNorm = WhisperOps.EnsureF32(w[$"{p}.post_attention_layernorm.weight"]);
            _qW = WhisperOps.EnsureF32(w[$"{p}.self_attn.q_proj.weight"]); _qB = TryGet(w, $"{p}.self_attn.q_proj.bias");
            _kW = WhisperOps.EnsureF32(w[$"{p}.self_attn.k_proj.weight"]); _kB = TryGet(w, $"{p}.self_attn.k_proj.bias");
            _vW = WhisperOps.EnsureF32(w[$"{p}.self_attn.v_proj.weight"]); _vB = TryGet(w, $"{p}.self_attn.v_proj.bias");
            _oW = WhisperOps.EnsureF32(w[$"{p}.self_attn.o_proj.weight"]); _oB = TryGet(w, $"{p}.self_attn.o_proj.bias");
            _gateW = WhisperOps.EnsureF32(w[$"{p}.mlp.gate_proj.weight"]);
            _upW = WhisperOps.EnsureF32(w[$"{p}.mlp.up_proj.weight"]);
            _downW = WhisperOps.EnsureF32(w[$"{p}.mlp.down_proj.weight"]);
            _ls1 = WhisperOps.EnsureF32(w[$"{p}.layer_scale_1.scale"]);
            _ls2 = WhisperOps.EnsureF32(w[$"{p}.layer_scale_2.scale"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t, Tensor? mask, float[] cos, float[] sin)
        {
            int dim = _cfg.TransformerDim, heads = _cfg.TransformerHeads, hd = _cfg.TransformerHeadDim;
            Tensor pre = new(new TensorShape(1, t, dim), DType.F32);
            backend.RmsNorm(pre, x, _inNorm!, _cfg.RmsNormEps);

            Tensor q = WhisperOps.ProjectLinear(backend, pre, _qW!, _qB, 1, t, dim, heads * hd);
            Tensor k = WhisperOps.ProjectLinear(backend, pre, _kW!, _kB, 1, t, dim, heads * hd);
            Tensor v = WhisperOps.ProjectLinear(backend, pre, _vW!, _vB, 1, t, dim, heads * hd);
            pre.Dispose();
            Tensor qM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Tensor kM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Tensor vM = new(new TensorShape(1, heads, t, hd), DType.F32);
            Dia.DiaHeads.FlatToHeads(qM, q, t, heads, hd); q.Dispose();
            Dia.DiaHeads.FlatToHeads(kM, k, t, heads, hd); k.Dispose();
            Dia.DiaHeads.FlatToHeads(vM, v, t, heads, hd); v.Dispose();
            Moonshine.RotaryEmbedding.ApplyInPlace(qM, heads, t, hd, hd, 0, cos, sin);
            Moonshine.RotaryEmbedding.ApplyInPlace(kM, heads, t, hd, hd, 0, cos, sin);

            Tensor attn = new(new TensorShape(1, heads, t, hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qM, kM, vM, mask, 1f / MathF.Sqrt(hd));
            qM.Dispose(); kM.Dispose(); vM.Dispose();
            Tensor flat = new(new TensorShape(1, t, heads * hd), DType.F32);
            Dia.DiaHeads.HeadsToFlat(flat, attn, t, heads, hd); attn.Dispose();
            Tensor attnOut = WhisperOps.ProjectLinear(backend, flat, _oW!, _oB, 1, t, heads * hd, dim); flat.Dispose();

            Tensor afterAttn = new(new TensorShape(1, t, dim), DType.F32);
            ScaledResidual(afterAttn, x, attnOut, _ls1!, t, dim); attnOut.Dispose();

            Tensor pre2 = new(new TensorShape(1, t, dim), DType.F32);
            backend.RmsNorm(pre2, afterAttn, _postNorm!, _cfg.RmsNormEps);
            Tensor gate = WhisperOps.ProjectLinear(backend, pre2, _gateW!, null, 1, t, dim, _cfg.TransformerFfnDim);
            Tensor up = WhisperOps.ProjectLinear(backend, pre2, _upW!, null, 1, t, dim, _cfg.TransformerFfnDim); pre2.Dispose();
            backend.Silu(gate, gate);
            Tensor act = new(gate.Shape, DType.F32); backend.Mul(act, gate, up); gate.Dispose(); up.Dispose();
            Tensor mlp = WhisperOps.ProjectLinear(backend, act, _downW!, null, 1, t, _cfg.TransformerFfnDim, dim); act.Dispose();
            Tensor outT = new(new TensorShape(1, t, dim), DType.F32);
            ScaledResidual(outT, afterAttn, mlp, _ls2!, t, dim); afterAttn.Dispose(); mlp.Dispose();
            return outT;
        }

        /// <summary>out = residual + layerScale ⊙ value, where layerScale is per-channel [dim].</summary>
        private static void ScaledResidual(Tensor outT, Tensor residual, Tensor value, Tensor scale, int t, int dim)
        {
            float* op = (float*)outT.DataPointer;
            float* rp = (float*)residual.DataPointer;
            float* vp = (float*)value.DataPointer;
            float* sp = (float*)scale.DataPointer;
            for (int s = 0; s < t; s++)
                for (int c = 0; c < dim; c++)
                {
                    long idx = (long)s * dim + c;
                    op[idx] = rp[idx] + sp[c] * vp[idx];
                }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_inNorm, _postNorm, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _gateW, _upW, _downW, _ls1, _ls2];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    /// <summary>Upsample stage: causal transposed conv (stride <c>rate</c>) followed by a <see cref="ConvNeXtBlock"/>.</summary>
    private sealed class UpsampleStage
    {
        private readonly int _channels;
        private readonly int _rate;
        private Tensor? _convW, _convB;
        private readonly ConvNeXtBlock _convnext = new();

        public UpsampleStage(int channels, int rate) { _channels = channels; _rate = rate; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _convW = WhisperOps.EnsureF32(w[$"{p}.conv.weight"]);
            _convB = TryGet(w, $"{p}.conv.bias");
            _convnext.LoadWeights(w, $"{p}.convnext");
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int tIn = (int)x.Shape[2];
            int k = (int)_convW!.Shape[2];
            // Causal transposed conv: trim trailing pad so output length == tIn*rate.
            int tOut = tIn * _rate;
            Tensor up = new(new TensorShape(1, _channels, tOut), DType.F32);
            backend.ConvTranspose1d(up, x, _convW!, _convB, _rate, 0, k - _rate, 1, 1);
            Tensor outT = _convnext.Forward(backend, up);
            up.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_convW is not null) yield return _convW;
            if (_convB is not null) yield return _convB;
            foreach (Tensor t in _convnext.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>ConvNeXt 1-D block: depthwise k7 causal conv → channel LayerNorm → pointwise expand → GELU →
    /// pointwise project → γ layer-scale → + residual.</summary>
    private sealed class ConvNeXtBlock
    {
        private Tensor? _dwW, _dwB, _normW, _normB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _dwW = WhisperOps.EnsureF32(w[$"{p}.dwconv.weight"]); _dwB = TryGet(w, $"{p}.dwconv.bias");
            _normW = WhisperOps.EnsureF32(w[$"{p}.norm.weight"]); _normB = TryGet(w, $"{p}.norm.bias");
            _pw1W = WhisperOps.EnsureF32(w[$"{p}.pwconv1.weight"]); _pw1B = TryGet(w, $"{p}.pwconv1.bias");
            _pw2W = WhisperOps.EnsureF32(w[$"{p}.pwconv2.weight"]); _pw2B = TryGet(w, $"{p}.pwconv2.bias");
            _gamma = TryGet(w, $"{p}.gamma");
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int c = (int)x.Shape[1], t = (int)x.Shape[2];
            int k = (int)_dwW!.Shape[2];
            // Depthwise causal conv: groups == channels, left-pad (k-1).
            Tensor h = new(new TensorShape(1, c, t), DType.F32);
            backend.Conv1d(h, x, _dwW!, _dwB, 1, k - 1, 0, 1, c);
            LayerNormChannels(h, _normW!, _normB, c, t);

            int hidden = (int)_pw1W!.Shape[0];
            Tensor rows = ChannelsFirstToRows(h, c, t); h.Dispose();
            Tensor mid = new(new TensorShape(t, hidden), DType.F32);
            backend.Linear(mid, rows, _pw1W!, _pw1B); rows.Dispose();
            Tensor act = new(mid.Shape, DType.F32); backend.Gelu(act, mid); mid.Dispose();
            Tensor outRows = new(new TensorShape(t, c), DType.F32);
            backend.Linear(outRows, act, _pw2W!, _pw2B); act.Dispose();

            Tensor result = RowsToChannelsFirst(outRows, c, t); outRows.Dispose();
            float* rp = (float*)result.DataPointer;
            float* xp = (float*)x.DataPointer;
            float* gp = _gamma is null ? null : (float*)_gamma.DataPointer;
            for (int ci = 0; ci < c; ci++)
            {
                float g = gp is null ? 1f : gp[ci];
                for (int i = 0; i < t; i++)
                {
                    long off = (long)ci * t + i;
                    rp[off] = xp[off] + g * rp[off];
                }
            }
            return result;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_dwW, _dwB, _normW, _normB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    /// <summary>DAC-style decoder block: SnakeBeta → causal transposed-conv upsample (stride <c>rate</c>) →
    /// 3 dilated <see cref="ResidualUnit"/>s.</summary>
    private sealed class DecoderBlock
    {
        private readonly int _inCh, _outCh, _rate, _kernel;
        private readonly int[] _dilations;
        private Tensor? _alpha, _beta, _convW, _convB;
        private readonly ResidualUnit[] _residuals;

        public DecoderBlock(int inCh, int outCh, int rate, int kernel, IReadOnlyList<int> dilations)
        {
            _inCh = inCh; _outCh = outCh; _rate = rate; _kernel = kernel;
            _dilations = [.. dilations];
            _residuals = new ResidualUnit[_dilations.Length];
            for (int i = 0; i < _dilations.Length; i++) _residuals[i] = new ResidualUnit(outCh, kernel, _dilations[i]);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _alpha = WhisperOps.EnsureF32(w[$"{p}.snake.alpha"]);
            _beta = TryGet(w, $"{p}.snake.beta");
            _convW = WhisperOps.EnsureF32(w[$"{p}.up_conv.weight"]);
            _convB = TryGet(w, $"{p}.up_conv.bias");
            for (int i = 0; i < _residuals.Length; i++) _residuals[i].LoadWeights(w, $"{p}.residuals.{i}");
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int tIn = (int)x.Shape[2];
            Tensor act = new(x.Shape, DType.F32);
            backend.Snake(act, x, _alpha!, _beta);
            int k = (int)_convW!.Shape[2];
            int tOut = tIn * _rate;
            Tensor up = new(new TensorShape(1, _outCh, tOut), DType.F32);
            backend.ConvTranspose1d(up, act, _convW!, _convB, _rate, 0, k - _rate, 1, 1);
            act.Dispose();
            Tensor cur = up;
            for (int i = 0; i < _residuals.Length; i++)
            {
                Tensor next = _residuals[i].Forward(backend, cur);
                cur.Dispose();
                cur = next;
            }
            return cur;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_alpha, _beta, _convW, _convB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
            foreach (ResidualUnit r in _residuals) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>DAC residual unit: SnakeBeta → dilated causal conv k → SnakeBeta → 1×1 causal conv → + residual.</summary>
    private sealed class ResidualUnit
    {
        private readonly int _channels, _kernel, _dilation;
        private Tensor? _alpha1, _beta1, _conv1W, _conv1B, _alpha2, _beta2, _conv2W, _conv2B;

        public ResidualUnit(int channels, int kernel, int dilation) { _channels = channels; _kernel = kernel; _dilation = dilation; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _alpha1 = WhisperOps.EnsureF32(w[$"{p}.snake1.alpha"]); _beta1 = TryGet(w, $"{p}.snake1.beta");
            _conv1W = WhisperOps.EnsureF32(w[$"{p}.conv1.weight"]); _conv1B = TryGet(w, $"{p}.conv1.bias");
            _alpha2 = WhisperOps.EnsureF32(w[$"{p}.snake2.alpha"]); _beta2 = TryGet(w, $"{p}.snake2.beta");
            _conv2W = WhisperOps.EnsureF32(w[$"{p}.conv2.weight"]); _conv2B = TryGet(w, $"{p}.conv2.bias");
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int t = (int)x.Shape[2];
            Tensor a1 = new(x.Shape, DType.F32); backend.Snake(a1, x, _alpha1!, _beta1);
            Tensor c1 = CausalConv(backend, a1, _conv1W!, _conv1B, _channels, t, _kernel, _dilation); a1.Dispose();
            Tensor a2 = new(c1.Shape, DType.F32); backend.Snake(a2, c1, _alpha2!, _beta2); c1.Dispose();
            Tensor c2 = CausalConv(backend, a2, _conv2W!, _conv2B, _channels, t, 1, 1); a2.Dispose();
            float* xp = (float*)x.DataPointer;
            float* cp = (float*)c2.DataPointer;
            long n = x.ElementCount;
            for (long i = 0; i < n; i++) cp[i] += xp[i];
            return c2;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_alpha1, _beta1, _conv1W, _conv1B, _alpha2, _beta2, _conv2W, _conv2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    // ── shared small helpers ────────────────────────────────────────────

    /// <summary>Per-(channel-vector) LayerNorm over the channel axis of a <c>[1, C, T]</c> tensor (ConvNeXt
    /// normalizes across channels at each time step), affine via <paramref name="weight"/>/<paramref name="bias"/> [C].</summary>
    private static void LayerNormChannels(Tensor x, Tensor weight, Tensor? bias, int c, int t)
    {
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        const float eps = 1e-6f;
        for (int s = 0; s < t; s++)
        {
            double mean = 0;
            for (int ci = 0; ci < c; ci++) mean += xp[(long)ci * t + s];
            mean /= c;
            double var = 0;
            for (int ci = 0; ci < c; ci++) { double d = xp[(long)ci * t + s] - mean; var += d * d; }
            var /= c;
            float inv = (float)(1.0 / Math.Sqrt(var + eps));
            for (int ci = 0; ci < c; ci++)
            {
                long idx = (long)ci * t + s;
                float norm = (float)((xp[idx] - mean) * inv) * wp[ci];
                if (bp != null) norm += bp[ci];
                xp[idx] = norm;
            }
        }
    }

    /// <summary>[1, C, T] → [T, C] row-major (one token per row) for pointwise Linear.</summary>
    private static Tensor ChannelsFirstToRows(Tensor x, int c, int t)
    {
        Tensor rows = new(new TensorShape(t, c), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* rp = (float*)rows.DataPointer;
        for (int s = 0; s < t; s++)
            for (int ci = 0; ci < c; ci++)
                rp[(long)s * c + ci] = xp[(long)ci * t + s];
        return rows;
    }

    /// <summary>[T, C] → [1, C, T].</summary>
    private static Tensor RowsToChannelsFirst(Tensor rows, int c, int t)
    {
        Tensor x = new(new TensorShape(1, c, t), DType.F32);
        float* rp = (float*)rows.DataPointer;
        float* xp = (float*)x.DataPointer;
        for (int s = 0; s < t; s++)
            for (int ci = 0; ci < c; ci++)
                xp[(long)ci * t + s] = rp[(long)s * c + ci];
        return x;
    }
}
