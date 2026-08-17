using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Shared per-denoise-step context for every <see cref="LtxVideo2Block"/>: the text-connector outputs, the
/// eight global AdaLN modulation vectors (one per <c>*_adaln_single</c> module, each <c>[numParams·dim]</c>), and the
/// four RoPE cos/sin tables (video/audio self + cross-modal). Computed once in the main transformer forward and
/// reused across all 48 blocks; each block adds its own per-layer <c>scale_shift_table</c>s on top.</summary>
public sealed class LtxVideo2BlockContext
{
    public required Tensor Encoder;            // [Lv, videoDim] video text features (connector)
    public required Tensor AudioEncoder;       // [La, audioDim] audio text features
    public Tensor? EncoderMask;                // additive cross-attn masks (null = full)
    public Tensor? AudioEncoderMask;

    public required Tensor TembVideo;          // [9·videoDim]
    public required Tensor TembAudio;          // [9·audioDim]
    public Tensor? TembPromptVideo;            // [2·videoDim] — null when the checkpoint has no prompt modulation
    public Tensor? TembPromptAudio;            // [2·audioDim]
    public required Tensor TembCaVideoScaleShift;   // [4·videoDim]
    public required Tensor TembCaAudioScaleShift;   // [4·audioDim]
    public required Tensor TembCaVideoGate;    // [1·videoDim]
    public required Tensor TembCaAudioGate;    // [1·audioDim]

    public required LtxVideo2Rope VideoRope;
    public required Tensor VideoCos, VideoSin;
    public required LtxVideo2Rope AudioRope;
    public required Tensor AudioCos, AudioSin;
    public required LtxVideo2Rope CaVideoRope;
    public required Tensor CaVideoCos, CaVideoSin;
    public required LtxVideo2Rope CaAudioRope;
    public required Tensor CaAudioCos, CaAudioSin;
}

/// <summary>One LTX-2.3 dual-stream transformer block (<c>LTX2VideoTransformerBlock</c>), single sample (B=1). Runs
/// the video <c>[Sv, 4096]</c> and audio <c>[Sa, 2048]</c> streams through four interleaved stages, each AdaLN-Single
/// modulated (per-block <c>scale_shift_table</c> + the global step temb, broadcast over the sequence):
/// (1) gated self-attention with RoPE, (2) gated cross-attention to the text features, (3) bidirectional
/// audio↔video cross-attention (a2v updates video, v2a updates audio), (4) gelu-approximate FFN. All pre-norms are
/// RMSNorm-no-affine (eps 1e-6); QK-norm is full-width RMSNorm. Reuses <see cref="LtxVideo2Attention"/>.</summary>
public sealed unsafe class LtxVideo2Block : IStreamingBlock
{
    private readonly int _vDim, _aDim;
    private readonly float _normEps;
    private readonly Tensor _onesV, _onesA;   // unit weights for the no-affine RMS pre-norms

    private readonly LtxVideo2Attention _attn1, _attn2, _aAttn1, _aAttn2, _a2v, _v2a;

    // Per-block AdaLN tables.
    private Tensor? _ssVideo;       // [9, vDim]
    private Tensor? _ssAudio;       // [9, aDim]
    private Tensor? _promptVideo;   // [2, vDim]
    private Tensor? _promptAudio;   // [2, aDim]
    private Tensor? _caVideo;       // [5, vDim]  ([:4]=scale/shift, [4:]=gate)
    private Tensor? _caAudio;       // [5, aDim]

    // FFNs.
    private Tensor? _ffW0, _ffB0, _ffW2, _ffB2, _aFfW0, _aFfB0, _aFfW2, _aFfB2;

    public LtxVideo2Block(LtxVideo2Config c)
    {
        _vDim = c.InnerDim;
        _aDim = c.AudioInnerDim;
        _normEps = c.NormEps;
        _onesV = Ones(_vDim);
        _onesA = Ones(_aDim);

        int vH = c.NumHeads, vHd = c.HeadDim, aH = c.AudioNumHeads, aHd = c.AudioHeadDim;
        float qk = c.QkNormEps;
        _attn1 = new LtxVideo2Attention(_vDim, _vDim, vH, vHd, _vDim, qk);            // video self
        _attn2 = new LtxVideo2Attention(_vDim, c.CrossAttentionDim, vH, vHd, _vDim, qk); // video-text cross
        _aAttn1 = new LtxVideo2Attention(_aDim, _aDim, aH, aHd, _aDim, qk);           // audio self
        _aAttn2 = new LtxVideo2Attention(_aDim, c.AudioCrossAttentionDim, aH, aHd, _aDim, qk); // audio-text cross
        // a2v: Q=video (vDim), KV=audio (aDim), attention at audio head dims, out→video.
        _a2v = new LtxVideo2Attention(_vDim, _aDim, aH, aHd, _vDim, qk);
        // v2a: Q=audio (aDim), KV=video (vDim), out→audio.
        _v2a = new LtxVideo2Attention(_aDim, _vDim, aH, aHd, _aDim, qk);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _attn1.LoadWeights(w, $"{p}.attn1");
        _attn2.LoadWeights(w, $"{p}.attn2");
        _aAttn1.LoadWeights(w, $"{p}.audio_attn1");
        _aAttn2.LoadWeights(w, $"{p}.audio_attn2");
        _a2v.LoadWeights(w, $"{p}.audio_to_video_attn");
        _v2a.LoadWeights(w, $"{p}.video_to_audio_attn");

        _ssVideo = LoadF32(w, $"{p}.scale_shift_table");
        _ssAudio = LoadF32(w, $"{p}.audio_scale_shift_table");
        // Prompt (text cross-attn KV) modulation is a 2.3-only feature; earlier LTX-2 (e.g. 19B) omits it — the
        // text features are then used unmodulated. Load only if present.
        _promptVideo = w.ContainsKey($"{p}.prompt_scale_shift_table") ? LoadF32(w, $"{p}.prompt_scale_shift_table") : null;
        _promptAudio = w.ContainsKey($"{p}.audio_prompt_scale_shift_table") ? LoadF32(w, $"{p}.audio_prompt_scale_shift_table") : null;
        _caVideo = LoadF32(w, $"{p}.scale_shift_table_a2v_ca_video");
        _caAudio = LoadF32(w, $"{p}.scale_shift_table_a2v_ca_audio");

        _ffW0 = w[$"{p}.ff.net.0.proj.weight"]; w.TryGetValue($"{p}.ff.net.0.proj.bias", out _ffB0);
        _ffW2 = w[$"{p}.ff.net.2.weight"]; w.TryGetValue($"{p}.ff.net.2.bias", out _ffB2);
        _aFfW0 = w[$"{p}.audio_ff.net.0.proj.weight"]; w.TryGetValue($"{p}.audio_ff.net.0.proj.bias", out _aFfB0);
        _aFfW2 = w[$"{p}.audio_ff.net.2.weight"]; w.TryGetValue($"{p}.audio_ff.net.2.bias", out _aFfB2);
    }

    /// <summary>Sum of this block's weight bytes — the streaming controller's budget heuristic. Computed on demand.</summary>
    /// <remarks>Goes through <see cref="DType.ComputeByteCount"/> because block-quantized dtypes carry
    /// <c>SizeInBytes == 0</c> (e.g. <c>Q4_K</c> is <c>("Q4_K", 0, true, 144, 256)</c>) — the naive
    /// <c>ElementCount * SizeInBytes</c> reports 0 for them, which would make the streaming budget see a
    /// weightless model and never engage on exactly the quantized checkpoints that need it.</remarks>
    public long EstimatedWeightBytes
    {
        get
        {
            long total = 0;
            foreach (Tensor t in EnumerateWeights()) total += t.DType.ComputeByteCount(t.ElementCount);
            return total;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (LtxVideo2Attention a in new[] { _attn1, _attn2, _aAttn1, _aAttn2, _a2v, _v2a })
            foreach (Tensor t in a.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _ssVideo, _ssAudio, _promptVideo, _promptAudio, _caVideo, _caAudio,
            _ffW0, _ffB0, _ffW2, _ffB2, _aFfW0, _aFfB0, _aFfW2, _aFfB2 })
            if (t is not null) yield return t;
        // HOST-built, so without this they are neither a preloaded weight nor any device op's output: the norms
        // read them 8x per block per CFG branch, and an unlisted tensor is a cache miss + re-upload on every read.
        yield return _onesV;
        yield return _onesA;
    }

    /// <summary>Forward both streams. Returns the updated (video, audio) hidden states; the inputs are consumed.</summary>
    public (Tensor Video, Tensor Audio) Forward(IBackend backend, Tensor hidden, Tensor audioHidden, LtxVideo2BlockContext ctx)
    {
        int sv = (int)hidden.Shape[0], sa = (int)audioHidden.Shape[0];

        // Modulation params (per-block table + global temb), broadcast over the sequence.
        Tensor[] vMod = Modulation(backend, _ssVideo!, 0, ctx.TembVideo, 9, _vDim);     // shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp,shift_tq,scale_tq,gate_tq
        Tensor[] aMod = Modulation(backend, _ssAudio!, 0, ctx.TembAudio, 9, _aDim);
        bool hasPrompt = _promptVideo is not null && ctx.TembPromptVideo is not null;
        Tensor[]? vPrompt = hasPrompt ? Modulation(backend, _promptVideo!, 0, ctx.TembPromptVideo!, 2, _vDim) : null;   // shift_kv,scale_kv
        Tensor[]? aPrompt = hasPrompt ? Modulation(backend, _promptAudio!, 0, ctx.TembPromptAudio!, 2, _aDim) : null;
        Tensor[] vCaSS = Modulation(backend, _caVideo!, 0, ctx.TembCaVideoScaleShift, 4, _vDim); // a2v_scale,a2v_shift,v2a_scale,v2a_shift
        Tensor[] aCaSS = Modulation(backend, _caAudio!, 0, ctx.TembCaAudioScaleShift, 4, _aDim);
        Tensor[] vCaGate = Modulation(backend, _caVideo!, 4, ctx.TembCaVideoGate, 1, _vDim);     // a2v_gate
        Tensor[] aCaGate = Modulation(backend, _caAudio!, 4, ctx.TembCaAudioGate, 1, _aDim);     // v2a_gate

        // ── 1. Self-attention ──
        Tensor n1 = NormModulate(backend, hidden, _onesV, vMod[1], vMod[0], _vDim);
        Tensor attn1 = _attn1.Forward(backend, n1, n1, ctx.VideoRope, ctx.VideoCos, ctx.VideoSin, ctx.VideoRope, ctx.VideoCos, ctx.VideoSin, null);
        n1.Dispose();
        hidden = GatedAddInto(backend, hidden, attn1, vMod[2]); attn1.Dispose();

        Tensor an1 = NormModulate(backend, audioHidden, _onesA, aMod[1], aMod[0], _aDim);
        Tensor aAttn1 = _aAttn1.Forward(backend, an1, an1, ctx.AudioRope, ctx.AudioCos, ctx.AudioSin, ctx.AudioRope, ctx.AudioCos, ctx.AudioSin, null);
        an1.Dispose();
        audioHidden = GatedAddInto(backend, audioHidden, aAttn1, aMod[2]); aAttn1.Dispose();

        // ── 2. Text cross-attention (Q: stream; K,V: text) ──
        Tensor n2 = NormModulate(backend, hidden, _onesV, vMod[7], vMod[6], _vDim);   // scale_text_q, shift_text_q
        Tensor encMod = vPrompt is not null ? ModulateRows(backend, ctx.Encoder, vPrompt[1], vPrompt[0], _vDim) : ctx.Encoder;   // enc·(1+scale_kv)+shift_kv
        Tensor attn2 = _attn2.Forward(backend, n2, encMod, null, null, null, null, null, null, ctx.EncoderMask);
        n2.Dispose(); if (vPrompt is not null) encMod.Dispose();
        hidden = GatedAddInto(backend, hidden, attn2, vMod[8]); attn2.Dispose();   // gate_text_q

        Tensor an2 = NormModulate(backend, audioHidden, _onesA, aMod[7], aMod[6], _aDim);
        Tensor aEncMod = aPrompt is not null ? ModulateRows(backend, ctx.AudioEncoder, aPrompt[1], aPrompt[0], _aDim) : ctx.AudioEncoder;
        Tensor aAttn2 = _aAttn2.Forward(backend, an2, aEncMod, null, null, null, null, null, null, ctx.AudioEncoderMask);
        an2.Dispose(); if (aPrompt is not null) aEncMod.Dispose();
        audioHidden = GatedAddInto(backend, audioHidden, aAttn2, aMod[8]); aAttn2.Dispose();

        // ── 3. Audio↔Video cross-attention ──
        Tensor ncaV = RmsNoAffine(backend, hidden, _onesV, _vDim);          // audio_to_video_norm(video)
        Tensor ncaA = RmsNoAffine(backend, audioHidden, _onesA, _aDim);     // video_to_audio_norm(audio)

        // a2v: Q=video, K,V=audio → update video.
        Tensor modV = ModulateRows(backend, ncaV, vCaSS[0], vCaSS[1], _vDim);   // a2v_scale, a2v_shift
        Tensor modA = ModulateRows(backend, ncaA, aCaSS[0], aCaSS[1], _aDim);
        Tensor a2v = _a2v.Forward(backend, modV, modA, ctx.CaVideoRope, ctx.CaVideoCos, ctx.CaVideoSin, ctx.CaAudioRope, ctx.CaAudioCos, ctx.CaAudioSin, null);
        modV.Dispose(); modA.Dispose();
        hidden = GatedAddInto(backend, hidden, a2v, vCaGate[0]); a2v.Dispose();

        // v2a: Q=audio, K,V=video → update audio.
        Tensor modV2 = ModulateRows(backend, ncaV, vCaSS[2], vCaSS[3], _vDim);  // v2a_scale, v2a_shift
        Tensor modA2 = ModulateRows(backend, ncaA, aCaSS[2], aCaSS[3], _aDim);
        Tensor v2a = _v2a.Forward(backend, modA2, modV2, ctx.CaAudioRope, ctx.CaAudioCos, ctx.CaAudioSin, ctx.CaVideoRope, ctx.CaVideoCos, ctx.CaVideoSin, null);
        modV2.Dispose(); modA2.Dispose();
        audioHidden = GatedAddInto(backend, audioHidden, v2a, aCaGate[0]); v2a.Dispose();
        ncaV.Dispose(); ncaA.Dispose();

        // ── 4. Feed-forward ──
        Tensor n3 = NormModulate(backend, hidden, _onesV, vMod[4], vMod[3], _vDim);   // scale_mlp, shift_mlp
        Tensor ff = Ffn(backend, n3, _ffW0!, _ffB0, _ffW2!, _ffB2, _vDim);
        n3.Dispose();
        hidden = GatedAddInto(backend, hidden, ff, vMod[5]); ff.Dispose();

        Tensor an3 = NormModulate(backend, audioHidden, _onesA, aMod[4], aMod[3], _aDim);
        Tensor aFf = Ffn(backend, an3, _aFfW0!, _aFfB0, _aFfW2!, _aFfB2, _aDim);
        an3.Dispose();
        audioHidden = GatedAddInto(backend, audioHidden, aFf, aMod[5]); aFf.Dispose();

        foreach (Tensor[]? arr in new[] { vMod, aMod, vPrompt, aPrompt, vCaSS, aCaSS, vCaGate, aCaGate })
            if (arr is not null) foreach (Tensor t in arr) t.Dispose();
        return (hidden, audioHidden);
    }

    /// <summary>AdaLN params on-device: rows <c>[row, row+n)</c> of <paramref name="table"/> plus <paramref name="temb"/>
    /// (<c>[n·dim]</c>), sliced into n <c>[dim]</c> vectors. Must stay on-device — a host loop here is a D2H drain of
    /// temb + per-vector H2D re-uploads on every block of every step (the Krea2/Chroma host-glue pathology).</summary>
    private static Tensor[] Modulation(IBackend backend, Tensor table, int row, Tensor temb, int n, int dim)
    {
        // Always slice the table rows first: elementwise Add sizes its launch from the FIRST operand, so the operand
        // must be exactly [n, dim] even when row==0 (the av-ca table is [5, dim] but only 4 or 1 rows participate).
        Tensor rows = new(new TensorShape(n, dim), DType.F32);
        backend.SliceRows(rows, table, row);
        Tensor sum = new(new TensorShape(n, dim), DType.F32);
        backend.Add(sum, rows, temb);
        rows.Dispose();
        Tensor[] outs = new Tensor[n];
        for (int m = 0; m < n; m++)
        {
            Tensor o = new(new TensorShape(dim), DType.F32);
            backend.SliceRows(o, sum, m);
            outs[m] = o;
        }
        sum.Dispose();
        return outs;
    }

    /// <summary>Affine-free RMS norm and the AdaLN shift/scale in ONE pass — the pair the block runs six times
    /// (video/audio x self-attn/cross-attn/FFN). The separate ops walked the activation twice.</summary>
    private Tensor NormModulate(IBackend backend, Tensor x, Tensor ones, Tensor scale, Tensor shift, int dim)
    {
        Tensor o = new(new TensorShape((int)x.Shape[0], dim), x.DType);
        backend.Ltx2RmsModulate(o, x, ones, scale, shift, _normEps);
        return o;
    }

    private Tensor RmsNoAffine(IBackend backend, Tensor x, Tensor ones, int dim)
    {
        // Activation dtype follows the stream (see LtxVideo2Attention.Forward); `ones` stays F32 — the F16
        // RmsNorm kernel takes an F16 activation with an F32 weight and rejects an F16 weight.
        Tensor o = new(new TensorShape((int)x.Shape[0], dim), x.DType);
        backend.RmsNorm(o, x, ones, _normEps);
        return o;
    }

    /// <summary>Out-of-place <c>x·(1+scale)+shift</c> (used for inputs we must not mutate, e.g. the shared
    /// connector encoder and the shared a2v/v2a norms).</summary>
    private static Tensor ModulateRows(IBackend backend, Tensor x, Tensor scale, Tensor shift, int dim)
    {
        Tensor scale1 = new(new TensorShape(dim), DType.F32);
        backend.AddScalar(scale1, scale, 1f);
        Tensor o = new(new TensorShape((int)x.Shape[0], dim), x.DType);
        backend.AffineBroadcastLastDim(o, x, scale1, shift);
        scale1.Dispose();
        return o;
    }

    /// <summary><c>residual + gate · value</c> into a fresh tensor; disposes the residual.</summary>
    private static Tensor GatedAddInto(IBackend backend, Tensor residual, Tensor value, Tensor gate)
    {
        // GatedResidualLastDim requires residual/value/output to share a dtype (gate stays F32); the value is an
        // attention or FFN output, which inherited the residual's dtype upstream.
        Tensor o = new(residual.Shape, residual.DType);
        backend.GatedResidualLastDim(o, residual, value, gate);
        residual.Dispose();
        return o;
    }

    private static Tensor Ffn(IBackend backend, Tensor x, Tensor w0, Tensor? b0, Tensor w2, Tensor? b2, int dim)
    {
        int s = (int)x.Shape[0];
        int inner = (int)w0.Shape[0];
        // The 4096->16384 GELU intermediate is the F16 overflow risk site (65504 ceiling). Verified empirically on
        // the real int8-convrot checkpoint rather than assumed: incoherent frames here would mean overflow, and the
        // fix would be to pin `proj`/`act` to F32 while the rest of the block stays F16.
        // gelu-approximate (tanh) folded into the up-projection's dequant epilogue where the backend supports
        // it — this intermediate is [tokens, 4*dim], the widest tensor in the block.
        Tensor act = new(new TensorShape(s, inner), x.DType);
        backend.LinearGelu(act, x, w0, b0);
        Tensor o = new(new TensorShape(s, dim), x.DType);
        backend.Linear(o, act, w2, b2);
        act.Dispose();
        return o;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}
