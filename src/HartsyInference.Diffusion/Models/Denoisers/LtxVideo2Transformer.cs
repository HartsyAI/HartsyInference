using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>LTX-2.3 dual-stream audio+video DiT (<c>LTX2VideoTransformer3DModel</c>), ported from the vendored
/// diffusers <c>transformer_ltx2.py</c>. Single sample (B=1) over packed VAE-latent tokens: video <c>[Sv, 128]</c>
/// and audio <c>[Sa, 128]</c>. Flow: <c>proj_in</c>/<c>audio_proj_in</c> patchify projections → 48
/// <see cref="LtxVideo2Block"/>s (driven by a per-step <see cref="LtxVideo2BlockContext"/>) → per-stream AdaLN-Single
/// output layer (<c>scale_shift_table</c> + the time-embedding) → <c>proj_out</c>/<c>audio_proj_out</c>.
///
/// <para>The eight global modulation tables the blocks consume are produced by eight
/// <see cref="LTX2AdaLayerNormSingle"/> modules (<c>time_embed</c>, <c>audio_time_embed</c>, <c>prompt_adaln</c>,
/// <c>audio_prompt_adaln</c>, <c>av_cross_attn_video_scale_shift</c>, <c>av_cross_attn_audio_scale_shift</c>,
/// <c>av_cross_attn_video_a2v_gate</c>, <c>av_cross_attn_audio_v2a_gate</c>) — each a PixArt timestep embedder
/// (<c>emb.timestep_embedder.linear_1/2</c>) feeding a <c>linear</c> that emits <c>numParams·dim</c> modulation
/// values. LTX-2.3 uses per-modality text connectors, so <c>caption_projection</c> is absent (the encoder inputs are
/// already the connector outputs, video <c>[Lv, 4096]</c> / audio <c>[La, 2048]</c>).</para>
///
/// <para>RoPE: four flavors built per call from the latent grid — video self (3-axis, 4096), audio self (1-axis,
/// 2048), video-cross (temporal-only, 2048) and audio-cross (identical to audio self). See
/// <see cref="LtxVideo2Rope"/>. Numerics vs the real checkpoint are validation-pending.</para></summary>
public sealed unsafe class LtxVideo2Transformer : IDisposable
{
    private readonly LtxVideo2Config _config;
    private readonly LtxVideo2Block[] _blocks;
    private readonly LtxVideo2Rope _videoRope, _audioRope, _caVideoRope;
    private readonly float _gateScaleFactor;
    private int _disposed;

    private Tensor? _projInW, _projInB, _audioProjInW, _audioProjInB;
    private Tensor? _keyframesAbsPos;
    private Tensor? _keyframesOnes;
    private Tensor? _projOutW, _projOutB, _audioProjOutW, _audioProjOutB;
    private Tensor? _scaleShift, _audioScaleShift;          // [2, inner]

    private readonly AdaLnSingle _timeEmbed, _audioTimeEmbed;
    private readonly AdaLnSingle _promptAdaln, _audioPromptAdaln;
    private bool _hasPromptMod;
    private readonly AdaLnSingle _caVideoScaleShift, _caAudioScaleShift;
    private readonly AdaLnSingle _caVideoGate, _caAudioGate;

    // Per-generation RoPE cos/sin cache (grid-keyed; tables are step-invariant).
    private (int Frames, int Height, int Width, double Fps, int AudioFrames) _ropeKey = (-1, -1, -1, 0, -1);
    private Tensor? _vCosC, _vSinC, _aCosC, _aSinC, _cvCosC, _cvSinC;

    // ── Step-graph capture (HARTSY_DIT_GRAPH) — the whole dual-stream CFG pair (proj_in ×4 → 48 blocks interleaved
    // → 4 output layers) captured once and replayed. FULLY RESIDENT only (BeforeBlockForward null): a captured graph
    // bakes weight pointers, so the streamed-block path is incompatible — on 24 GB the 22B streams and this stays
    // eager; a bigger GPU (or HARTSY_LTX2_HEADROOM_MB=0 at a small geometry) holds all 48 resident and captures.
    // The per-step timestep tables (8 block-modulation + embV/embA) build OUTSIDE the capture into the fixed buffers
    // below; the latents live in the pipeline's device-Euler'd videoLat/audioLat; velocities land in fixed buffers
    // consumed by the pipeline's device CfgEulerStep (no host read → no snapshot needed). ──
    private Tensor? _gTVideo, _gTAudio, _gTPromptV, _gTPromptA, _gTCaVss, _gTCaAss, _gTCaVGate, _gTCaAGate, _gEmbV, _gEmbA;
    private Tensor? _gVCondV, _gVCondA, _gVUncondV, _gVUncondA;   // the 4 fixed velocity outputs
    private long _graphSig = long.MinValue;
    private int _graphSigCalls, _graphSigFlips;
    private bool _graphDead, _graphPinned;
    private const int GraphCaptureCall = 3;

    /// <summary>True once the CFG-pair step graph PATH is running (from its first call — <c>_graphSig</c> set — until
    /// self-disable). The pipeline keeps the four velocity buffers (transformer-owned fixed buffers, rewritten next
    /// step) instead of disposing them while this holds — including the warm calls, which already return the fixed
    /// buffers. (LTX-2.3 has no mid-loop FreeActivations, so unlike Kandinsky this need not wait for the capture.)</summary>
    public bool StepGraphActive => DiTBlocks.DitStepGraph.Enabled && !_graphDead && !StepGraphSuspended
        && _graphSig != long.MinValue;

    /// <summary>Makes forwards run eager and stay INVISIBLE to the graph state machine — no signature compare, no
    /// flip counted, no capture. The two-stage pipeline's refine stage sets this: it runs at a different grid, and
    /// letting that grid reach the signature would both burn two of the eight allowed flips per generation (after
    /// which the graph is disabled for the session) and capture on its own last step, never replaying.</summary>
    public bool StepGraphSuspended { get; set; }

    /// <summary>Invalidates the captured graph — MUST run before the DiT weights are freed (the graph bakes weight
    /// pointers). The next generation re-warms and re-captures.</summary>
    public void InvalidateStepGraph(IBackend backend)
    {
        backend.StepGraphReset();
        if (ReferenceEquals(backend.StepGraphOwner, this)) backend.StepGraphOwner = null;
        _graphSig = long.MinValue;
        _graphSigCalls = 0;
        _graphPinned = false;
    }

    // Diagnostic (HARTSY_LTX2_PROBE=1): per-block residual-stream absmax/rms on the FIRST forward only — the
    // F16-activation headroom probe (streams riding >60k overflow F16). Host sync per probe, debug only.
    // Reads whichever dtype the stream is in: since the block path went F16 this must decode halves, and the
    // old raw-float read would have reported nonsense at exactly the moment the probe is worth running.
    private static readonly bool Ltx2Probe = Environment.GetEnvironmentVariable("HARTSY_LTX2_PROBE") == "1";
    private static bool _probedOnce;

    private static void Probe(string label, Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float* p = (float*)f32.DataPointer;
        float mx = 0f;
        double ss = 0;
        long n = f32.ElementCount;
        for (long e = 0; e < n; e++) { float av = MathF.Abs(p[e]); if (av > mx) mx = av; ss += (double)p[e] * p[e]; }
        HartsyInference.Core.Logging.Logs.Info($"[ltx2-probe] {label}: absmax={mx:F2} rms={Math.Sqrt(ss / n):F4} ({t.DType})");
        if (!ReferenceEquals(f32, t)) f32.Dispose();
    }

    public LtxVideo2Transformer(LtxVideo2Config config)
    {
        _config = config;
        _blocks = new LtxVideo2Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new LtxVideo2Block(config);

        int[] videoScale = [config.VaeTemporalCompression, config.VaeSpatialCompression, config.VaeSpatialCompression];
        // rope_type (split for LTX-2.3) + the per-rope head layout drive the split apply (pairs the two halves within
        // each head). Video self uses the video heads; the audio and both cross-modal ropes use the audio heads
        // (the a2v query, v2a query and audio self all run at the 32×64 audio head config, inner 2048).
        _videoRope = LtxVideo2Rope.ForVideoSelf(config.InnerDim, config.RopeTheta, config.RopeBaseNumFrames,
            config.RopeBaseHeight, config.RopeBaseWidth, videoScale, config.CausalOffset,
            config.RopeType, config.NumHeads, config.HeadDim);
        // Cross-modal video rope uses the audio-cross width (= audio inner dim), the temporal axis only, and the
        // a2v attention's (audio) head layout.
        _caVideoRope = LtxVideo2Rope.ForVideoCross(config.AudioCrossAttentionDim, config.RopeTheta,
            config.RopeBaseNumFrames, config.RopeBaseHeight, config.RopeBaseWidth, videoScale, config.CausalOffset,
            config.RopeType, config.AudioNumHeads, config.AudioHeadDim);
        // Audio self and audio-cross share coordinates and width (2048) — one rope serves both.
        _audioRope = LtxVideo2Rope.ForAudio(config.AudioInnerDim, config.RopeTheta, config.AudioPosEmbedMaxPos,
            config.AudioScaleFactor, config.CausalOffset, config.AudioSamplingRate, config.AudioHopLength,
            config.RopeType, config.AudioNumHeads, config.AudioHeadDim);

        _gateScaleFactor = (float)config.CrossAttnTimestepScaleMultiplier / config.TimestepScaleMultiplier;

        int v = config.InnerDim, a = config.AudioInnerDim;
        _timeEmbed = new AdaLnSingle(v, config.SelfAttnModParams);   // 9
        _audioTimeEmbed = new AdaLnSingle(a, config.SelfAttnModParams);
        _promptAdaln = new AdaLnSingle(v, 2);
        _audioPromptAdaln = new AdaLnSingle(a, 2);
        _caVideoScaleShift = new AdaLnSingle(v, 4);
        _caAudioScaleShift = new AdaLnSingle(a, 4);
        _caVideoGate = new AdaLnSingle(v, 1);
        _caAudioGate = new AdaLnSingle(a, 1);
    }

    public LtxVideo2Config Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _projInW = w["proj_in.weight"]; w.TryGetValue("proj_in.bias", out _projInB);
        _audioProjInW = w["audio_proj_in.weight"]; w.TryGetValue("audio_proj_in.bias", out _audioProjInB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _audioProjOutW = w["audio_proj_out.weight"]; w.TryGetValue("audio_proj_out.bias", out _audioProjOutB);
        _scaleShift = LoadF32(w, "scale_shift_table");
        _audioScaleShift = LoadF32(w, "audio_scale_shift_table");

        _timeEmbed.LoadWeights(w, "time_embed");
        _audioTimeEmbed.LoadWeights(w, "audio_time_embed");
        // Prompt (text cross-attn KV) modulation is a 2.3-only feature; earlier LTX-2 (e.g. 19B) omits it.
        _hasPromptMod = w.ContainsKey("prompt_adaln.emb.timestep_embedder.linear_1.weight");
        if (_hasPromptMod)
        {
            _promptAdaln.LoadWeights(w, "prompt_adaln");
            _audioPromptAdaln.LoadWeights(w, "audio_prompt_adaln");
        }
        _caVideoScaleShift.LoadWeights(w, "av_cross_attn_video_scale_shift");
        _caAudioScaleShift.LoadWeights(w, "av_cross_attn_audio_scale_shift");
        _caVideoGate.LoadWeights(w, "av_cross_attn_video_a2v_gate");
        _caAudioGate.LoadWeights(w, "av_cross_attn_audio_v2a_gate");

        // LTX-2.5's absolute marker for the tokens at temporal position 0.
        if (w.TryGetValue(LtxVideo2VariantDetector.KeyframesEmbeddingKey, out Tensor? keyframes))
        {
            _keyframesAbsPos = keyframes.DType == DType.F32 ? keyframes : keyframes.CastTo(DType.F32);
            _keyframesOnes = OnesRow(_config.InnerDim);
        }

        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");

        VerifyVariant(w);
    }

    /// <summary>Fails a load whose weights contradict the detected variant, so a mis-detected checkpoint stops here
    /// instead of generating quietly wrong output.</summary>
    private void VerifyVariant(IReadOnlyDictionary<string, Tensor> w)
    {
        bool hasKeyframes = _keyframesAbsPos is not null;
        if (hasKeyframes != _config.UseKeyframesAbsPosEmbedding)
        {
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"LTX-2 config/checkpoint mismatch: UseKeyframesAbsPosEmbedding={_config.UseKeyframesAbsPosEmbedding} " +
                $"but '{LtxVideo2VariantDetector.KeyframesEmbeddingKey}' is {(hasKeyframes ? "present" : "absent")}.");
        }

        bool hasFfBias = w.ContainsKey("transformer_blocks.0.ff.net.0.proj.bias");
        if (hasFfBias != _config.FfBias)
        {
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"LTX-2 config/checkpoint mismatch: FfBias={_config.FfBias} but the video FFN bias is " +
                $"{(hasFfBias ? "present" : "absent")}.");
        }
    }

    private static Tensor OnesRow(int n)
    {
        Tensor t = new Tensor(new TensorShape(1, n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }

    /// <summary>Adds the LTX-2.5 keyframe marker to the tokens whose temporal position starts at 0. Video tokens are
    /// laid out <c>(frame, height, width)</c>-major and <see cref="DiTBlocks.LtxVideo2Rope"/> clamps the causal
    /// temporal start at 0, which only the first latent frame reaches — so those tokens are the leading
    /// <c>height·width</c> rows. Revisit alongside any work that appends i2v guide tokens, which the reference
    /// excludes from this mask.</summary>
    private void ApplyKeyframesAbsPos(IBackend backend, Tensor hidden, (int Frames, int Height, int Width) grid)
    {
        if (_keyframesAbsPos is null) return;
        int rows = grid.Height * grid.Width;
        // A grid that can't address the first latent frame is a caller bug, and skipping the marker would generate
        // silently un-marked video — the exact failure VerifyVariant exists to stop.
        if (rows <= 0 || rows > (int)hidden.Shape[0])
        {
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException(
                $"LTX-2.5 keyframe marker: grid {grid.Height}x{grid.Width} needs {rows} token rows but the video " +
                $"stream has {hidden.Shape[0]}.");
        }
        // Slice/affine/scatter rather than aliasing hidden's HOST buffer: reading DataPointer drains the whole
        // [tokens, dim] stream D2H and frees its device copy, and the affine's result then lives only in the alias
        // tensor's own device cache — which Dispose frees WITHOUT a write-back, so the marker never reached the
        // hidden that the next op re-uploads. On the CPU backend the aliasing worked, which is why this survived.
        using Tensor firstFrame = new Tensor(new TensorShape(rows, (int)hidden.Shape[1]), hidden.DType);
        backend.SliceRowsGeneric(firstFrame, hidden, 0);
        backend.AffineBroadcastLastDim(firstFrame, firstFrame, _keyframesOnes!, _keyframesAbsPos);
        backend.ScatterRowsGeneric(hidden, firstFrame, 0);
    }

    /// <summary>Always-resident (non-block) weights — proj_in/out, the global AdaLN-Single modulation tables. Touched
    /// every step regardless of the executing block, so the streaming controller doesn't manage them; preload eagerly.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        // The keyframe marker and its companion ones-row are read on every step and by the captured step graph,
        // which bakes pointers — so they belong with the eagerly-preloaded set, not the streamed blocks.
        foreach (Tensor? t in new[] { _projInW, _projInB, _audioProjInW, _audioProjInB,
            _projOutW, _projOutB, _audioProjOutW, _audioProjOutB, _scaleShift, _audioScaleShift,
            _keyframesAbsPos, _keyframesOnes })
            if (t is not null) yield return t;
        AdaLnSingle[] adalns = _hasPromptMod
            ? new[] { _timeEmbed, _audioTimeEmbed, _promptAdaln, _audioPromptAdaln, _caVideoScaleShift, _caAudioScaleShift, _caVideoGate, _caAudioGate }
            : new[] { _timeEmbed, _audioTimeEmbed, _caVideoScaleShift, _caAudioScaleShift, _caVideoGate, _caAudioGate };
        foreach (AdaLnSingle m in adalns)
            foreach (Tensor t in m.EnumerateWeights()) yield return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in EnumerateSharedWeights()) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
    }

    /// <summary>Number of streamable transformer blocks.</summary>
    public int BlockCount => _blocks.Length;

    /// <summary>The streamable block at <paramref name="idx"/> (implements <see cref="IStreamingBlock"/>).</summary>
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <summary>Optional hook invoked immediately before each block's forward pass — pipelines plug a
    /// <c>BlockStreamingController</c> here to drive prefetch/eviction so the 22B fp8 fits in 24 GB. Null = all resident.</summary>
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>Debug/parity hook invoked with the current video + audio hidden states after <c>proj_in</c>
    /// (index <c>-1</c>) and after each transformer block (index <c>0..BlockCount-1</c>). Used by the numerical parity
    /// harness to localize where the C# port diverges from the diffusers reference; leave null in production. On a GPU
    /// backend the tensors are device-resident, so read them from a CPU-backend run (or sync) when dumping.</summary>
    public Action<int, Tensor, Tensor>? OnBlockOutput { get; set; }

    /// <summary>Velocity prediction over both streams. <paramref name="videoTokens"/> is <c>[Sv, inChannels]</c>
    /// (f,h,w order); <paramref name="audioTokens"/> is <c>[Sa, audioInChannels]</c>; <paramref name="encoderVideo"/>
    /// /<paramref name="encoderAudio"/> are the per-modality text-connector outputs (<c>[Lv, 4096]</c>/<c>[La,
    /// 2048]</c>); <paramref name="timestep"/> is the scheduler timestep (≈0..1000). Returns the (video, audio)
    /// velocities <c>[Sv, outChannels]</c>/<c>[Sa, audioOutChannels]</c>; the inputs are consumed.</summary>
    public (Tensor Video, Tensor Audio) Forward(IBackend backend, Tensor videoTokens, Tensor audioTokens,
        Tensor encoderVideo, Tensor encoderAudio, float timestep,
        (int Frames, int Height, int Width) grid, int audioFrames, double fps,
        Tensor? encoderVideoMask, Tensor? encoderAudioMask)
    {
        int sv = (int)videoTokens.Shape[0], sa = (int)audioTokens.Shape[0];
        int v = _config.InnerDim, a = _config.AudioInnerDim;

        // proj_in patchify projections. Same activation-dtype choice as ForwardCfgPair — see the note there.
        DType act = backend.SupportsF16Activations ? DiTBlocks.DitDtype.Act : DType.F32;
        Tensor hidden = new Tensor(new TensorShape(sv, v), act);
        backend.Linear(hidden, videoTokens, _projInW!, _projInB);
        ApplyKeyframesAbsPos(backend, hidden, grid);
        Tensor audioHidden = new Tensor(new TensorShape(sa, a), act);
        backend.Linear(audioHidden, audioTokens, _audioProjInW!, _audioProjInB);

        // Global modulation tables + the two embedded-timesteps used by the output layer.
        (Tensor tVideo, Tensor embV) = _timeEmbed.Forward(backend, timestep);
        (Tensor tAudio, Tensor embA) = _audioTimeEmbed.Forward(backend, timestep);
        Tensor? tPromptV = null, tPromptA = null;
        if (_hasPromptMod)
        {
            // Same ×1000-scaled timestep as every other modulator; the reference feeds compute_prompt_timestep
            // its timestep_scaled. The raw sigma lands ~1000x off along the sinusoidal embedding and leaves the
            // text cross-attn unable to discriminate between prompts.
            (tPromptV, Tensor _pv) = _promptAdaln.Forward(backend, timestep); _pv.Dispose();
            (tPromptA, Tensor _pa) = _audioPromptAdaln.Forward(backend, timestep); _pa.Dispose();
        }
        (Tensor tCaVss, Tensor _cv) = _caVideoScaleShift.Forward(backend, timestep); _cv.Dispose();
        (Tensor tCaAss, Tensor _ca) = _caAudioScaleShift.Forward(backend, timestep); _ca.Dispose();
        (Tensor tCaVGate, Tensor _gv) = _caVideoGate.Forward(backend, timestep * _gateScaleFactor); _gv.Dispose();
        (Tensor tCaAGate, Tensor _ga) = _caAudioGate.Forward(backend, timestep * _gateScaleFactor); _ga.Dispose();

        // RoPE tables depend only on (grid, fps, audioFrames) — fixed for a whole generation — so build them once
        // and reuse across every step/CFG forward (was 3 host builds + uploads ×2 forwards ×steps per gen).
        (int, int, int, double, int) ropeKey = (grid.Frames, grid.Height, grid.Width, fps, audioFrames);
        if (_ropeKey != ropeKey)
        {
            BuildRopeTables(backend, grid, audioFrames, fps);
            _ropeKey = ropeKey;
        }
        (Tensor vCos, Tensor vSin) = (_vCosC!, _vSinC!);
        (Tensor aCos, Tensor aSin) = (_aCosC!, _aSinC!);
        (Tensor cvCos, Tensor cvSin) = (_cvCosC!, _cvSinC!);

        LtxVideo2BlockContext ctx = new()
        {
            Encoder = encoderVideo,
            AudioEncoder = encoderAudio,
            EncoderMask = encoderVideoMask,
            AudioEncoderMask = encoderAudioMask,
            TembVideo = tVideo,
            TembAudio = tAudio,
            TembPromptVideo = tPromptV,
            TembPromptAudio = tPromptA,
            TembCaVideoScaleShift = tCaVss,
            TembCaAudioScaleShift = tCaAss,
            TembCaVideoGate = tCaVGate,
            TembCaAudioGate = tCaAGate,
            VideoRope = _videoRope, VideoCos = vCos, VideoSin = vSin,
            AudioRope = _audioRope, AudioCos = aCos, AudioSin = aSin,
            CaVideoRope = _caVideoRope, CaVideoCos = cvCos, CaVideoSin = cvSin,
            // Audio-cross shares the audio self rope + coordinates.
            CaAudioRope = _audioRope, CaAudioCos = aCos, CaAudioSin = aSin,
        };

        OnBlockOutput?.Invoke(-1, hidden, audioHidden);   // post-proj_in state (parity harness)
        for (int i = 0; i < _blocks.Length; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (hidden, audioHidden) = _blocks[i].Forward(backend, hidden, audioHidden, ctx);
            OnBlockOutput?.Invoke(i, hidden, audioHidden);
        }

        foreach (Tensor? t in new[] { tVideo, tAudio, tPromptV, tPromptA, tCaVss, tCaAss, tCaVGate, tCaAGate })
            t?.Dispose();

        Tensor video = OutputLayer(backend, hidden, embV, _scaleShift!, _projOutW!, _projOutB, sv, v, _config.OutChannels);
        Tensor audio = OutputLayer(backend, audioHidden, embA, _audioScaleShift!, _audioProjOutW!, _audioProjOutB,
            sa, a, _config.AudioOutChannels);
        hidden.Dispose(); audioHidden.Dispose(); embV.Dispose(); embA.Dispose();
        return (video, audio);
    }

    /// <summary>CFG-paired velocity prediction: runs the conditional and unconditional branches through each block
    /// back-to-back, so under block streaming every block's weights upload ONCE per step instead of once per branch —
    /// halving the dominant 19 GB/forward weight traffic. The branches share the latent inputs (proj_in computed once,
    /// device-copied), all timestep modulation, and the RoPE tables; they differ only in the text-connector features.
    /// Numerically identical to two <see cref="Forward"/> calls. <see cref="OnBlockOutput"/> fires for the conditional
    /// branch only.</summary>
    public ((Tensor Video, Tensor Audio) Cond, (Tensor Video, Tensor Audio) Uncond) ForwardCfgPair(
        IBackend backend, Tensor videoTokens, Tensor audioTokens,
        Tensor encoderVideoCond, Tensor encoderAudioCond, Tensor encoderVideoUncond, Tensor encoderAudioUncond,
        float timestep, (int Frames, int Height, int Width) grid, int audioFrames, double fps)
    {
        int sv = (int)videoTokens.Shape[0], sa = (int)audioTokens.Shape[0];
        int v = _config.InnerDim, a = _config.AudioInnerDim;

        // Step-graph fast path (opt-in): capture the whole CFG pair once and replay it. FULLY RESIDENT only —
        // BeforeBlockForward null means no block streaming (a captured graph can't span re-pointered weights).
        // Diagnostic hooks off (they inject host reads). Self-disables to eager on capture failure.
        if (DiTBlocks.DitStepGraph.Enabled && backend.StepGraphSupported && !_graphDead && !StepGraphSuspended
            && BeforeBlockForward is null && OnBlockOutput is null)
        {
            return ForwardCfgPairGraph(backend, videoTokens, audioTokens,
                encoderVideoCond, encoderAudioCond, encoderVideoUncond, encoderAudioUncond,
                timestep, grid, audioFrames, fps, sv, sa, v, a);
        }

        // proj_in is where the block stream's activation dtype is CHOSEN; every block/attention helper below
        // inherits it from its input, and OutputLayer converts back to F32 for the pipeline's F32-only Euler step.
        // The latent in, the connector encoder, the AdaLN modulation vectors and every norm/rope parameter stay F32.
        // Gated on the backend: the CPU backend's Linear is F32-only, so it keeps the F32 path.
        DType act = backend.SupportsF16Activations ? DiTBlocks.DitDtype.Act : DType.F32;
        Tensor hidC = new Tensor(new TensorShape(sv, v), act);
        backend.Linear(hidC, videoTokens, _projInW!, _projInB);
        ApplyKeyframesAbsPos(backend, hidC, grid);
        Tensor audC = new Tensor(new TensorShape(sa, a), act);
        backend.Linear(audC, audioTokens, _audioProjInW!, _audioProjInB);
        Tensor hidU = new Tensor(new TensorShape(sv, v), act);
        backend.CopyInto(hidU, hidC);
        Tensor audU = new Tensor(new TensorShape(sa, a), act);
        backend.CopyInto(audU, audC);

        (Tensor tVideo, Tensor embV) = _timeEmbed.Forward(backend, timestep);
        (Tensor tAudio, Tensor embA) = _audioTimeEmbed.Forward(backend, timestep);
        Tensor? tPromptV = null, tPromptA = null;
        if (_hasPromptMod)
        {
            (tPromptV, Tensor _pv) = _promptAdaln.Forward(backend, timestep); _pv.Dispose();
            (tPromptA, Tensor _pa) = _audioPromptAdaln.Forward(backend, timestep); _pa.Dispose();
        }
        (Tensor tCaVss, Tensor _cv) = _caVideoScaleShift.Forward(backend, timestep); _cv.Dispose();
        (Tensor tCaAss, Tensor _ca) = _caAudioScaleShift.Forward(backend, timestep); _ca.Dispose();
        (Tensor tCaVGate, Tensor _gv) = _caVideoGate.Forward(backend, timestep * _gateScaleFactor); _gv.Dispose();
        (Tensor tCaAGate, Tensor _ga) = _caAudioGate.Forward(backend, timestep * _gateScaleFactor); _ga.Dispose();

        (int, int, int, double, int) ropeKey = (grid.Frames, grid.Height, grid.Width, fps, audioFrames);
        if (_ropeKey != ropeKey)
        {
            BuildRopeTables(backend, grid, audioFrames, fps);
            _ropeKey = ropeKey;
        }

        LtxVideo2BlockContext MakeCtx(Tensor encV, Tensor encA) => new()
        {
            Encoder = encV,
            AudioEncoder = encA,
            EncoderMask = null,
            AudioEncoderMask = null,
            TembVideo = tVideo,
            TembAudio = tAudio,
            TembPromptVideo = tPromptV,
            TembPromptAudio = tPromptA,
            TembCaVideoScaleShift = tCaVss,
            TembCaAudioScaleShift = tCaAss,
            TembCaVideoGate = tCaVGate,
            TembCaAudioGate = tCaAGate,
            VideoRope = _videoRope, VideoCos = _vCosC!, VideoSin = _vSinC!,
            AudioRope = _audioRope, AudioCos = _aCosC!, AudioSin = _aSinC!,
            CaVideoRope = _caVideoRope, CaVideoCos = _cvCosC!, CaVideoSin = _cvSinC!,
            CaAudioRope = _audioRope, CaAudioCos = _aCosC!, CaAudioSin = _aSinC!,
        };
        LtxVideo2BlockContext ctxC = MakeCtx(encoderVideoCond, encoderAudioCond);
        LtxVideo2BlockContext ctxU = MakeCtx(encoderVideoUncond, encoderAudioUncond);

        OnBlockOutput?.Invoke(-1, hidC, audC);
        bool probeThisForward = Ltx2Probe && !_probedOnce;
        if (probeThisForward) { Probe("proj_in video", hidC); Probe("proj_in audio", audC); }
        for (int i = 0; i < _blocks.Length; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (hidC, audC) = _blocks[i].Forward(backend, hidC, audC, ctxC);
            (hidU, audU) = _blocks[i].Forward(backend, hidU, audU, ctxU);
            OnBlockOutput?.Invoke(i, hidC, audC);
            if (probeThisForward) { Probe($"block{i} video", hidC); Probe($"block{i} audio", audC); }
        }
        if (probeThisForward) _probedOnce = true;

        foreach (Tensor? t in new[] { tVideo, tAudio, tPromptV, tPromptA, tCaVss, tCaAss, tCaVGate, tCaAGate })
            t?.Dispose();

        Tensor videoC = OutputLayer(backend, hidC, embV, _scaleShift!, _projOutW!, _projOutB, sv, v, _config.OutChannels);
        Tensor audioC = OutputLayer(backend, audC, embA, _audioScaleShift!, _audioProjOutW!, _audioProjOutB, sa, a, _config.AudioOutChannels);
        Tensor videoU = OutputLayer(backend, hidU, embV, _scaleShift!, _projOutW!, _projOutB, sv, v, _config.OutChannels);
        Tensor audioU = OutputLayer(backend, audU, embA, _audioScaleShift!, _audioProjOutW!, _audioProjOutB, sa, a, _config.AudioOutChannels);
        hidC.Dispose(); audC.Dispose(); hidU.Dispose(); audU.Dispose(); embV.Dispose(); embA.Dispose();
        return ((videoC, audioC), (videoU, audioU));
    }

    /// <summary>Step-graph CFG-pair forward (fully-resident path). Builds the per-step timestep tables OUTSIDE the
    /// capture into fixed buffers, then captures/replays proj_in ×4 → 48 blocks (cond+uncond interleaved) → 4 output
    /// layers, landing the four velocities in fixed buffers the pipeline's device Euler consumes (no host read).
    /// Warm calls 1..2 run the body eagerly through the fixed buffers; call <see cref="GraphCaptureCall"/> records it.</summary>
    private ((Tensor Video, Tensor Audio) Cond, (Tensor Video, Tensor Audio) Uncond) ForwardCfgPairGraph(
        IBackend backend, Tensor videoTokens, Tensor audioTokens,
        Tensor encVCond, Tensor encACond, Tensor encVUncond, Tensor encAUncond,
        float timestep, (int Frames, int Height, int Width) grid, int audioFrames, double fps,
        int sv, int sa, int v, int a)
    {
        // ── Build the per-step timestep tables OUTSIDE the capture; land them in the fixed buffers ──
        (Tensor tVideo, Tensor embV) = _timeEmbed.Forward(backend, timestep);
        (Tensor tAudio, Tensor embA) = _audioTimeEmbed.Forward(backend, timestep);
        RefreshFixed(backend, ref _gTVideo, tVideo);
        RefreshFixed(backend, ref _gTAudio, tAudio);
        RefreshFixed(backend, ref _gEmbV, embV);
        RefreshFixed(backend, ref _gEmbA, embA);
        if (_hasPromptMod)
        {
            (Tensor tPromptV, Tensor _pv) = _promptAdaln.Forward(backend, timestep); _pv.Dispose();
            (Tensor tPromptA, Tensor _pa) = _audioPromptAdaln.Forward(backend, timestep); _pa.Dispose();
            RefreshFixed(backend, ref _gTPromptV, tPromptV);
            RefreshFixed(backend, ref _gTPromptA, tPromptA);
        }
        (Tensor tCaVss, Tensor _cv) = _caVideoScaleShift.Forward(backend, timestep); _cv.Dispose();
        (Tensor tCaAss, Tensor _ca) = _caAudioScaleShift.Forward(backend, timestep); _ca.Dispose();
        (Tensor tCaVGate, Tensor _gv) = _caVideoGate.Forward(backend, timestep * _gateScaleFactor); _gv.Dispose();
        (Tensor tCaAGate, Tensor _ga) = _caAudioGate.Forward(backend, timestep * _gateScaleFactor); _ga.Dispose();
        RefreshFixed(backend, ref _gTCaVss, tCaVss);
        RefreshFixed(backend, ref _gTCaAss, tCaAss);
        RefreshFixed(backend, ref _gTCaVGate, tCaVGate);
        RefreshFixed(backend, ref _gTCaAGate, tCaAGate);

        // ── RoPE tables (cached per grid; step-invariant) ──
        (int, int, int, double, int) ropeKey = (grid.Frames, grid.Height, grid.Width, fps, audioFrames);
        if (_ropeKey != ropeKey)
        {
            BuildRopeTables(backend, grid, audioFrames, fps);
            _ropeKey = ropeKey;
            _graphPinned = false;
        }

        // ── Signature / owner management ──
        long sig = ((long)grid.Frames * 73856093L) ^ ((long)grid.Height * 19349663L)
            ^ ((long)grid.Width * 83492791L) ^ ((long)audioFrames * 2654435761L);
        bool ownerLost = !ReferenceEquals(backend.StepGraphOwner, this);
        if (sig != _graphSig || ownerLost)
        {
            backend.StepGraphReset();
            if (!ownerLost && _graphSig != long.MinValue && ++_graphSigFlips > 8)
            {
                _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning("[LTX-2 graph] signature flip storm — step-graph disabled for this session.");
            }
            _graphSig = sig; _graphSigCalls = 0; backend.StepGraphOwner = this; _graphPinned = false;
        }
        _graphSigCalls++;

        if (_graphSigCalls == 1 && !_graphPinned)
        {
            // Pin the step-invariant capture inputs (the 4 connector encoders + the RoPE tables) so a pageable
            // re-upload can't invalidate the recording.
            List<Tensor> pins = new()
            {
                encVCond, encACond, encVUncond, encAUncond,
                _vCosC!, _vSinC!, _aCosC!, _aSinC!, _cvCosC!, _cvSinC!,
            };
            backend.PreloadWeights(pins);
            _graphPinned = true;
        }

        _gVCondV ??= new Tensor(new TensorShape(sv, _config.OutChannels), DType.F32);
        _gVCondA ??= new Tensor(new TensorShape(sa, _config.AudioOutChannels), DType.F32);
        _gVUncondV ??= new Tensor(new TensorShape(sv, _config.OutChannels), DType.F32);
        _gVUncondA ??= new Tensor(new TensorShape(sa, _config.AudioOutChannels), DType.F32);

        LtxVideo2BlockContext ctxC = MakeGraphCtx(encVCond, encACond);
        LtxVideo2BlockContext ctxU = MakeGraphCtx(encVUncond, encAUncond);

        if (backend.StepGraphReady && _graphSigCalls > GraphCaptureCall)
        {
            backend.StepGraphLaunch();
            return ((_gVCondV!, _gVCondA!), (_gVUncondV!, _gVUncondA!));
        }

        bool capture = _graphSigCalls == GraphCaptureCall;
        if (capture) { backend.TrimMemoryPool(); backend.StepGraphBegin(); }
        try
        {
            RunCfgPairIntoFixed(backend, videoTokens, audioTokens, ctxC, ctxU, sv, sa, v, a, grid);
        }
        catch (Exception ex) when (capture)
        {
            backend.StepGraphReset(); _graphDead = true;
            HartsyInference.Core.Logging.Logs.Warning($"[LTX-2 graph] capture invalidated — falling back to eager: {ex.Message}");
            RunCfgPairIntoFixed(backend, videoTokens, audioTokens, ctxC, ctxU, sv, sa, v, a, grid);
            return ((_gVCondV!, _gVCondA!), (_gVUncondV!, _gVUncondA!));
        }
        if (capture)
        {
            try
            {
                backend.StepGraphEndAndLaunch();
                HartsyInference.Core.Logging.Logs.Info("[LTX-2 graph] dual-stream CFG-pair denoise step captured; replaying via cuGraphLaunch.");
            }
            catch (Exception ex)
            {
                backend.StepGraphReset(); _graphDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[LTX-2 graph] capture failed — falling back to eager: {ex.Message}");
                RunCfgPairIntoFixed(backend, videoTokens, audioTokens, ctxC, ctxU, sv, sa, v, a, grid);
            }
        }
        return ((_gVCondV!, _gVCondA!), (_gVUncondV!, _gVUncondA!));
    }

    /// <summary>Lands a freshly built per-step tensor in its fixed device buffer (device copy) and disposes the
    /// fresh one — the fixed buffer keeps a stable address the capture bakes.</summary>
    private static void RefreshFixed(IBackend backend, ref Tensor? fixedBuf, Tensor fresh)
    {
        fixedBuf ??= new Tensor(fresh.Shape, DType.F32);
        backend.CopyInto(fixedBuf, fresh);
        fresh.Dispose();
    }

    /// <summary>Block context wired to the FIXED per-step timestep buffers + cached RoPE (mirrors the eager MakeCtx).</summary>
    private LtxVideo2BlockContext MakeGraphCtx(Tensor encV, Tensor encA) => new()
    {
        Encoder = encV, AudioEncoder = encA, EncoderMask = null, AudioEncoderMask = null,
        TembVideo = _gTVideo!, TembAudio = _gTAudio!,
        TembPromptVideo = _gTPromptV, TembPromptAudio = _gTPromptA,
        TembCaVideoScaleShift = _gTCaVss!, TembCaAudioScaleShift = _gTCaAss!,
        TembCaVideoGate = _gTCaVGate!, TembCaAudioGate = _gTCaAGate!,
        VideoRope = _videoRope, VideoCos = _vCosC!, VideoSin = _vSinC!,
        AudioRope = _audioRope, AudioCos = _aCosC!, AudioSin = _aSinC!,
        CaVideoRope = _caVideoRope, CaVideoCos = _cvCosC!, CaVideoSin = _cvSinC!,
        CaAudioRope = _audioRope, CaAudioCos = _aCosC!, CaAudioSin = _aSinC!,
    };

    /// <summary>The captured (or capture-warming) CFG-pair body: proj_in ×4 → blocks (cond then uncond per block)
    /// → 4 output layers, landing the velocities in the fixed buffers. All device ops; reads only fixed/pinned/
    /// resident tensors.</summary>
    private void RunCfgPairIntoFixed(IBackend backend, Tensor videoTokens, Tensor audioTokens,
        LtxVideo2BlockContext ctxC, LtxVideo2BlockContext ctxU, int sv, int sa, int v, int a,
        (int Frames, int Height, int Width) grid)
    {
        DType act = backend.SupportsF16Activations ? DiTBlocks.DitDtype.Act : DType.F32;
        Tensor hidC = new Tensor(new TensorShape(sv, v), act);
        backend.Linear(hidC, videoTokens, _projInW!, _projInB);
        ApplyKeyframesAbsPos(backend, hidC, grid);
        Tensor audC = new Tensor(new TensorShape(sa, a), act);
        backend.Linear(audC, audioTokens, _audioProjInW!, _audioProjInB);
        Tensor hidU = new Tensor(new TensorShape(sv, v), act);
        backend.CopyInto(hidU, hidC);
        Tensor audU = new Tensor(new TensorShape(sa, a), act);
        backend.CopyInto(audU, audC);

        for (int i = 0; i < _blocks.Length; i++)
        {
            (hidC, audC) = _blocks[i].Forward(backend, hidC, audC, ctxC);
            (hidU, audU) = _blocks[i].Forward(backend, hidU, audU, ctxU);
        }

        Tensor videoC = OutputLayer(backend, hidC, _gEmbV!, _scaleShift!, _projOutW!, _projOutB, sv, v, _config.OutChannels);
        Tensor audioC = OutputLayer(backend, audC, _gEmbA!, _audioScaleShift!, _audioProjOutW!, _audioProjOutB, sa, a, _config.AudioOutChannels);
        Tensor videoU = OutputLayer(backend, hidU, _gEmbV!, _scaleShift!, _projOutW!, _projOutB, sv, v, _config.OutChannels);
        Tensor audioU = OutputLayer(backend, audU, _gEmbA!, _audioScaleShift!, _audioProjOutW!, _audioProjOutB, sa, a, _config.AudioOutChannels);
        hidC.Dispose(); audC.Dispose(); hidU.Dispose(); audU.Dispose();

        backend.CopyInto(_gVCondV!, videoC); backend.CopyInto(_gVCondA!, audioC);
        backend.CopyInto(_gVUncondV!, videoU); backend.CopyInto(_gVUncondA!, audioU);
        videoC.Dispose(); audioC.Dispose(); videoU.Dispose(); audioU.Dispose();
    }

    /// <summary>AdaLN-Single output, fully device-resident: <c>shift/scale = scale_shift_table + embedded</c> ([dim],
    /// broadcast over the sequence; table rows are [shift; scale] — keep that order), LayerNorm-no-affine,
    /// <c>·(1+scale)+shift</c>, then <c>proj_out</c>. Was a host loop draining the full hidden state, 2×/forward.</summary>
    private static Tensor OutputLayer(IBackend backend, Tensor hidden, Tensor embedded, Tensor scaleShift,
        Tensor projW, Tensor? projB, int s, int dim, int outChannels)
    {
        Tensor ssShift = new Tensor(new TensorShape(dim), DType.F32);
        backend.SliceRows(ssShift, scaleShift, 0);
        Tensor ssScale = new Tensor(new TensorShape(dim), DType.F32);
        backend.SliceRows(ssScale, scaleShift, 1);
        Tensor shift = new Tensor(new TensorShape(dim), DType.F32);
        backend.Add(shift, ssShift, embedded);
        ssShift.Dispose();
        Tensor scale1 = new Tensor(new TensorShape(dim), DType.F32);
        backend.Add(scale1, ssScale, embedded);
        ssScale.Dispose();
        Tensor scale1p = new Tensor(new TensorShape(dim), DType.F32);
        backend.AddScalar(scale1p, scale1, 1f);
        scale1.Dispose();
        // LayerNormNoAffine and AffineBroadcastLastDim both reject a mixed input/output pair, so these follow the
        // stream; the final proj_out writes F32 because the pipeline's CfgEulerStep is F32-only.
        Tensor normed = new Tensor(new TensorShape(s, dim), hidden.DType);
        backend.LayerNormNoAffine(normed, hidden, 1e-6f);
        Tensor modded = new Tensor(new TensorShape(s, dim), hidden.DType);
        backend.AffineBroadcastLastDim(modded, normed, scale1p, shift);
        normed.Dispose(); scale1p.Dispose(); shift.Dispose();
        Tensor outVel = new Tensor(new TensorShape(s, outChannels), DType.F32);
        backend.Linear(outVel, modded, projW, projB);
        modded.Dispose();
        return outVel;
    }

    /// <summary>Rebuilds and pins the per-generation RoPE tables. They depend only on (grid, fps, audioFrames), so
    /// every forward flavor shares this one site rather than keeping three copies that can drift.</summary>
    private void BuildRopeTables(IBackend backend, (int Frames, int Height, int Width) grid, int audioFrames, double fps)
    {
        DisposeRopeCache(backend);
        // F16 cos/sin halve the fused rope kernel's traffic, of which the tables are about half. Only the fused
        // SPLIT path reads them, and an F16 activation implies the backend that serves it.
        DType tables = LtxVideo2Rope.F16Tables && backend.SupportsF16Activations && DiTBlocks.DitDtype.Act == DType.F16
            && _videoRope.Flavor == LtxVideo2Rope.RopeType.Split ? DType.F16 : DType.F32;
        _videoRope.TableDType = tables;
        _audioRope.TableDType = tables;
        _caVideoRope.TableDType = tables;
        (_vCosC, _vSinC) = _videoRope.BuildVideo(grid.Frames, grid.Height, grid.Width, fps);
        (_aCosC, _aSinC) = _audioRope.BuildAudio(audioFrames);
        (_cvCosC, _cvSinC) = _caVideoRope.BuildVideo(grid.Frames, grid.Height, grid.Width, fps);
        // PIN them on the device. These are built on the HOST, so they are neither a preloaded weight nor any
        // device op's output — every attention that consumes one takes a cache miss and re-uploads it, then
        // frees it again. The big video tables are large enough for auto-promote to rescue; the audio pair
        // ([Sa, audioInner/2], ~0.4 MB) is not, and it was being re-uploaded 1536 times per step — ~620 MB of
        // pure PCIe traffic for tables that never change within a generation.
        backend.PreloadWeights(EnumerateRopeTables());
    }

    /// <summary>The per-generation RoPE tables, for pinning as resident weights (they are grid-invariant for the
    /// whole generation) and for releasing on the next re-size.</summary>
    private IEnumerable<Tensor> EnumerateRopeTables()
    {
        foreach (Tensor? t in new[] { _vCosC, _vSinC, _aCosC, _aSinC, _cvCosC, _cvSinC })
            if (t is not null) yield return t;
    }

    /// <summary>Releases the per-generation RoPE tables. <paramref name="backend"/> must be supplied whenever they
    /// were pinned (every path that built them), or their weight-cache entries outlive the tensors.</summary>
    private void DisposeRopeCache(IBackend? backend = null)
    {
        if (backend is not null) backend.FreeWeights(EnumerateRopeTables());
        foreach (Tensor? t in new[] { _vCosC, _vSinC, _aCosC, _aSinC, _cvCosC, _cvSinC })
            t?.Dispose();
        _vCosC = _vSinC = _aCosC = _aSinC = _cvCosC = _cvSinC = null;
        _ropeKey = (-1, -1, -1, 0, -1);
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        return t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Null-only (no Tensor.Dispose): transformer disposal may run AFTER the backend's CudaContext is torn
            // down, where disposing a GPU-promoted tensor throws ObjectDisposedException. The finalizer drain path
            // reclaims them; actual disposal happens on mid-session grid changes in DisposeRopeCache.
            _vCosC = _vSinC = _aCosC = _aSinC = _cvCosC = _cvSinC = null;
            _projInW = _projInB = _audioProjInW = _audioProjInB = null;
            _projOutW = _projOutB = _audioProjOutW = _audioProjOutB = _scaleShift = _audioScaleShift = null;
            _keyframesAbsPos = _keyframesOnes = null;
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>One <c>LTX2AdaLayerNormSingle</c>: a PixArt timestep embedder (<c>emb.timestep_embedder.linear_1/2</c>:
    /// sinusoidal-256 → SiLU → <c>dim</c>) feeding <c>linear</c> (SiLU → <c>numParams·dim</c>). Returns the flat
    /// <c>[numParams·dim]</c> modulation vector and the <c>[dim]</c> embedded timestep (used by the output layer).</summary>
    private sealed class AdaLnSingle
    {
        private readonly int _dim, _numParams;
        private Tensor? _emb1W, _emb1B, _emb2W, _emb2B, _linW, _linB;

        public AdaLnSingle(int dim, int numParams)
        {
            _dim = dim;
            _numParams = numParams;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _emb1W = w[$"{p}.emb.timestep_embedder.linear_1.weight"]; w.TryGetValue($"{p}.emb.timestep_embedder.linear_1.bias", out _emb1B);
            _emb2W = w[$"{p}.emb.timestep_embedder.linear_2.weight"]; w.TryGetValue($"{p}.emb.timestep_embedder.linear_2.bias", out _emb2B);
            _linW = w[$"{p}.linear.weight"]; w.TryGetValue($"{p}.linear.bias", out _linB);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _emb1W, _emb1B, _emb2W, _emb2B, _linW, _linB })
                if (t is not null) yield return t;
        }

        public (Tensor Mod, Tensor Embedded) Forward(IBackend backend, float timestep)
        {
            Tensor sin = new Tensor(new TensorShape(1, 256), DType.F32);
            DiTUtils.SinusoidalTimestepEmbedding(sin, timestep, 1, 256, 10000f);
            Tensor e1 = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.Linear(e1, sin, _emb1W!, _emb1B); sin.Dispose();
            Tensor e1a = new Tensor(e1.Shape, DType.F32); backend.Silu(e1a, e1); e1.Dispose();
            Tensor embedded2d = new Tensor(new TensorShape(1, _dim), DType.F32);
            backend.Linear(embedded2d, e1a, _emb2W!, _emb2B); e1a.Dispose();

            Tensor sil = new Tensor(new TensorShape(1, _dim), DType.F32); backend.Silu(sil, embedded2d);
            Tensor modFlat = new Tensor(new TensorShape(1, _numParams * _dim), DType.F32);
            backend.Linear(modFlat, sil, _linW!, _linB); sil.Dispose();

            // Returned as the [1, n·dim]/[1, dim] Linear outputs directly — every consumer is an element-count
            // device op. The old host Buffer.MemoryCopy repack was a D2H drain of both tensors per modulator
            // (8 modulators × 2 CFG forwards per step — the Kandinsky temb stream-drain pathology).
            return (modFlat, embedded2d);
        }
    }
}
