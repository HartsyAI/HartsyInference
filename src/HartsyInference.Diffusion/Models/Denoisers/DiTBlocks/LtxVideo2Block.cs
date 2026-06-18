using HartsyInference.Core.Backends;
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
    public required Tensor TembPromptVideo;    // [2·videoDim]
    public required Tensor TembPromptAudio;    // [2·audioDim]
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
public sealed unsafe class LtxVideo2Block
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
        _promptVideo = LoadF32(w, $"{p}.prompt_scale_shift_table");
        _promptAudio = LoadF32(w, $"{p}.audio_prompt_scale_shift_table");
        _caVideo = LoadF32(w, $"{p}.scale_shift_table_a2v_ca_video");
        _caAudio = LoadF32(w, $"{p}.scale_shift_table_a2v_ca_audio");

        _ffW0 = w[$"{p}.ff.net.0.proj.weight"]; w.TryGetValue($"{p}.ff.net.0.proj.bias", out _ffB0);
        _ffW2 = w[$"{p}.ff.net.2.weight"]; w.TryGetValue($"{p}.ff.net.2.bias", out _ffB2);
        _aFfW0 = w[$"{p}.audio_ff.net.0.proj.weight"]; w.TryGetValue($"{p}.audio_ff.net.0.proj.bias", out _aFfB0);
        _aFfW2 = w[$"{p}.audio_ff.net.2.weight"]; w.TryGetValue($"{p}.audio_ff.net.2.bias", out _aFfB2);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (LtxVideo2Attention a in new[] { _attn1, _attn2, _aAttn1, _aAttn2, _a2v, _v2a })
            foreach (Tensor t in a.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _ssVideo, _ssAudio, _promptVideo, _promptAudio, _caVideo, _caAudio,
            _ffW0, _ffB0, _ffW2, _ffB2, _aFfW0, _aFfB0, _aFfW2, _aFfB2 })
            if (t is not null) yield return t;
    }

    /// <summary>Forward both streams. Returns the updated (video, audio) hidden states; the inputs are consumed.</summary>
    public (Tensor Video, Tensor Audio) Forward(IBackend backend, Tensor hidden, Tensor audioHidden, LtxVideo2BlockContext ctx)
    {
        int sv = (int)hidden.Shape[0], sa = (int)audioHidden.Shape[0];

        // Modulation params (per-block table + global temb), broadcast over the sequence.
        Tensor[] vMod = Modulation(_ssVideo!, ctx.TembVideo, 9, _vDim);     // shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp,shift_tq,scale_tq,gate_tq
        Tensor[] aMod = Modulation(_ssAudio!, ctx.TembAudio, 9, _aDim);
        Tensor[] vPrompt = Modulation(_promptVideo!, ctx.TembPromptVideo, 2, _vDim);   // shift_kv,scale_kv
        Tensor[] aPrompt = Modulation(_promptAudio!, ctx.TembPromptAudio, 2, _aDim);
        Tensor[] vCaSS = Modulation(Slice(_caVideo!, 0, 4, _vDim), ctx.TembCaVideoScaleShift, 4, _vDim); // a2v_scale,a2v_shift,v2a_scale,v2a_shift
        Tensor[] aCaSS = Modulation(Slice(_caAudio!, 0, 4, _aDim), ctx.TembCaAudioScaleShift, 4, _aDim);
        Tensor[] vCaGate = Modulation(Slice(_caVideo!, 4, 5, _vDim), ctx.TembCaVideoGate, 1, _vDim);     // a2v_gate
        Tensor[] aCaGate = Modulation(Slice(_caAudio!, 4, 5, _aDim), ctx.TembCaAudioGate, 1, _aDim);     // v2a_gate

        // ── 1. Self-attention ──
        Tensor n1 = RmsNoAffine(backend, hidden, _onesV, _vDim);
        ApplyShiftScale(n1, vMod[1], vMod[0], _vDim);
        Tensor attn1 = _attn1.Forward(backend, n1, n1, ctx.VideoRope, ctx.VideoCos, ctx.VideoSin, ctx.VideoRope, ctx.VideoCos, ctx.VideoSin, null);
        n1.Dispose();
        hidden = GatedAddInto(hidden, attn1, vMod[2], _vDim); attn1.Dispose();

        Tensor an1 = RmsNoAffine(backend, audioHidden, _onesA, _aDim);
        ApplyShiftScale(an1, aMod[1], aMod[0], _aDim);
        Tensor aAttn1 = _aAttn1.Forward(backend, an1, an1, ctx.AudioRope, ctx.AudioCos, ctx.AudioSin, ctx.AudioRope, ctx.AudioCos, ctx.AudioSin, null);
        an1.Dispose();
        audioHidden = GatedAddInto(audioHidden, aAttn1, aMod[2], _aDim); aAttn1.Dispose();

        // ── 2. Text cross-attention (Q: stream; K,V: text) ──
        Tensor n2 = RmsNoAffine(backend, hidden, _onesV, _vDim);
        ApplyShiftScale(n2, vMod[7], vMod[6], _vDim);                       // scale_text_q, shift_text_q
        Tensor encMod = ModulateRows(backend, ctx.Encoder, vPrompt[1], vPrompt[0], _vDim);   // enc·(1+scale_kv)+shift_kv
        Tensor attn2 = _attn2.Forward(backend, n2, encMod, null, null, null, null, null, null, ctx.EncoderMask);
        n2.Dispose(); encMod.Dispose();
        hidden = GatedAddInto(hidden, attn2, vMod[8], _vDim); attn2.Dispose();   // gate_text_q

        Tensor an2 = RmsNoAffine(backend, audioHidden, _onesA, _aDim);
        ApplyShiftScale(an2, aMod[7], aMod[6], _aDim);
        Tensor aEncMod = ModulateRows(backend, ctx.AudioEncoder, aPrompt[1], aPrompt[0], _aDim);
        Tensor aAttn2 = _aAttn2.Forward(backend, an2, aEncMod, null, null, null, null, null, null, ctx.AudioEncoderMask);
        an2.Dispose(); aEncMod.Dispose();
        audioHidden = GatedAddInto(audioHidden, aAttn2, aMod[8], _aDim); aAttn2.Dispose();

        // ── 3. Audio↔Video cross-attention ──
        Tensor ncaV = RmsNoAffine(backend, hidden, _onesV, _vDim);          // audio_to_video_norm(video)
        Tensor ncaA = RmsNoAffine(backend, audioHidden, _onesA, _aDim);     // video_to_audio_norm(audio)

        // a2v: Q=video, K,V=audio → update video.
        Tensor modV = ModulateRows(backend, ncaV, vCaSS[0], vCaSS[1], _vDim);   // a2v_scale, a2v_shift
        Tensor modA = ModulateRows(backend, ncaA, aCaSS[0], aCaSS[1], _aDim);
        Tensor a2v = _a2v.Forward(backend, modV, modA, ctx.CaVideoRope, ctx.CaVideoCos, ctx.CaVideoSin, ctx.CaAudioRope, ctx.CaAudioCos, ctx.CaAudioSin, null);
        modV.Dispose(); modA.Dispose();
        hidden = GatedAddInto(hidden, a2v, vCaGate[0], _vDim); a2v.Dispose();

        // v2a: Q=audio, K,V=video → update audio.
        Tensor modV2 = ModulateRows(backend, ncaV, vCaSS[2], vCaSS[3], _vDim);  // v2a_scale, v2a_shift
        Tensor modA2 = ModulateRows(backend, ncaA, aCaSS[2], aCaSS[3], _aDim);
        Tensor v2a = _v2a.Forward(backend, modA2, modV2, ctx.CaAudioRope, ctx.CaAudioCos, ctx.CaAudioSin, ctx.CaVideoRope, ctx.CaVideoCos, ctx.CaVideoSin, null);
        modV2.Dispose(); modA2.Dispose();
        audioHidden = GatedAddInto(audioHidden, v2a, aCaGate[0], _aDim); v2a.Dispose();
        ncaV.Dispose(); ncaA.Dispose();

        // ── 4. Feed-forward ──
        Tensor n3 = RmsNoAffine(backend, hidden, _onesV, _vDim);
        ApplyShiftScale(n3, vMod[4], vMod[3], _vDim);                       // scale_mlp, shift_mlp
        Tensor ff = Ffn(backend, n3, _ffW0!, _ffB0, _ffW2!, _ffB2, _vDim);
        n3.Dispose();
        hidden = GatedAddInto(hidden, ff, vMod[5], _vDim); ff.Dispose();

        Tensor an3 = RmsNoAffine(backend, audioHidden, _onesA, _aDim);
        ApplyShiftScale(an3, aMod[4], aMod[3], _aDim);
        Tensor aFf = Ffn(backend, an3, _aFfW0!, _aFfB0, _aFfW2!, _aFfB2, _aDim);
        an3.Dispose();
        audioHidden = GatedAddInto(audioHidden, aFf, aMod[5], _aDim); aFf.Dispose();

        foreach (Tensor[] arr in new[] { vMod, aMod, vPrompt, aPrompt, vCaSS, aCaSS, vCaGate, aCaGate })
            foreach (Tensor t in arr) t.Dispose();
        return (hidden, audioHidden);
    }

    /// <summary>AdaLN params: <c>table[m] + temb[m]</c> for <paramref name="n"/> params of width <paramref name="dim"/>.
    /// <paramref name="temb"/> is laid out as <c>[n·dim]</c>; returns n vectors <c>[dim]</c>.</summary>
    private static Tensor[] Modulation(Tensor table, Tensor temb, int n, int dim)
    {
        float* ss = (float*)table.DataPointer;
        float* tb = (float*)temb.DataPointer;
        Tensor[] outs = new Tensor[n];
        for (int m = 0; m < n; m++)
        {
            Tensor o = new(new TensorShape(dim), DType.F32);
            float* op = (float*)o.DataPointer;
            for (int d = 0; d < dim; d++) op[d] = ss[(long)m * dim + d] + tb[(long)m * dim + d];
            outs[m] = o;
        }
        return outs;
    }

    /// <summary>View-copy of rows <c>[start, end)</c> from a <c>[N, dim]</c> table into a fresh <c>[end-start, dim]</c>
    /// tensor (the per-layer av-ca table is split into a scale/shift part and a gate part).</summary>
    private static Tensor Slice(Tensor table, int start, int end, int dim)
    {
        int rows = end - start;
        Tensor o = new(new TensorShape(rows, dim), DType.F32);
        Buffer.MemoryCopy((float*)table.DataPointer + (long)start * dim, (float*)o.DataPointer,
            (long)rows * dim * 4, (long)rows * dim * 4);
        return o;
    }

    private Tensor RmsNoAffine(IBackend backend, Tensor x, Tensor ones, int dim)
    {
        Tensor o = new(new TensorShape((int)x.Shape[0], dim), DType.F32);
        backend.RmsNorm(o, x, ones, _normEps);
        return o;
    }

    private static void ApplyShiftScale(Tensor x, Tensor scale, Tensor shift, int dim)
    {
        int s = (int)x.Shape[0];
        float* xp = (float*)x.DataPointer, sc = (float*)scale.DataPointer, sh = (float*)shift.DataPointer;
        for (int i = 0; i < s; i++)
            for (int d = 0; d < dim; d++)
                xp[(long)i * dim + d] = xp[(long)i * dim + d] * (1f + sc[d]) + sh[d];
    }

    /// <summary>Out-of-place <c>x·(1+scale)+shift</c> (used for inputs we must not mutate, e.g. the shared
    /// connector encoder and the shared a2v/v2a norms).</summary>
    private static Tensor ModulateRows(IBackend backend, Tensor x, Tensor scale, Tensor shift, int dim)
    {
        int s = (int)x.Shape[0];
        Tensor o = new(new TensorShape(s, dim), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer, sc = (float*)scale.DataPointer, sh = (float*)shift.DataPointer;
        for (int i = 0; i < s; i++)
            for (int d = 0; d < dim; d++)
                op[(long)i * dim + d] = xp[(long)i * dim + d] * (1f + sc[d]) + sh[d];
        return o;
    }

    /// <summary><c>residual + gate · value</c> into a fresh tensor; disposes the residual.</summary>
    private static Tensor GatedAddInto(Tensor residual, Tensor value, Tensor gate, int dim)
    {
        int s = (int)residual.Shape[0];
        Tensor o = new(new TensorShape(s, dim), DType.F32);
        float* rp = (float*)residual.DataPointer, vp = (float*)value.DataPointer, gp = (float*)gate.DataPointer, op = (float*)o.DataPointer;
        for (int i = 0; i < s; i++)
            for (int d = 0; d < dim; d++)
                op[(long)i * dim + d] = rp[(long)i * dim + d] + gp[d] * vp[(long)i * dim + d];
        residual.Dispose();
        return o;
    }

    private static Tensor Ffn(IBackend backend, Tensor x, Tensor w0, Tensor? b0, Tensor w2, Tensor? b2, int dim)
    {
        int s = (int)x.Shape[0];
        int inner = (int)w0.Shape[0];
        Tensor proj = new(new TensorShape(s, inner), DType.F32);
        backend.Linear(proj, x, w0, b0);
        Tensor act = new(proj.Shape, DType.F32);
        backend.Gelu(act, proj);   // gelu-approximate (tanh)
        proj.Dispose();
        Tensor o = new(new TensorShape(s, dim), DType.F32);
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
