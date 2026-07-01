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
    private Tensor? _projOutW, _projOutB, _audioProjOutW, _audioProjOutB;
    private Tensor? _scaleShift, _audioScaleShift;          // [2, inner]

    private readonly AdaLnSingle _timeEmbed, _audioTimeEmbed;
    private readonly AdaLnSingle _promptAdaln, _audioPromptAdaln;
    private bool _hasPromptMod;
    private readonly AdaLnSingle _caVideoScaleShift, _caAudioScaleShift;
    private readonly AdaLnSingle _caVideoGate, _caAudioGate;

    public LtxVideo2Transformer(LtxVideo2Config config)
    {
        _config = config;
        _blocks = new LtxVideo2Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new LtxVideo2Block(config);

        int[] videoScale = [config.VaeTemporalCompression, config.VaeSpatialCompression, config.VaeSpatialCompression];
        _videoRope = LtxVideo2Rope.ForVideoSelf(config.InnerDim, config.RopeTheta, config.RopeBaseNumFrames,
            config.RopeBaseHeight, config.RopeBaseWidth, videoScale, config.CausalOffset);
        // Cross-modal video rope uses the audio-cross width (= audio inner dim) and the temporal axis only.
        _caVideoRope = LtxVideo2Rope.ForVideoCross(config.AudioCrossAttentionDim, config.RopeTheta,
            config.RopeBaseNumFrames, config.RopeBaseHeight, config.RopeBaseWidth, videoScale, config.CausalOffset);
        // Audio self and audio-cross share coordinates and width (2048) — one rope serves both.
        _audioRope = LtxVideo2Rope.ForAudio(config.AudioInnerDim, config.RopeTheta, config.AudioPosEmbedMaxPos,
            config.AudioScaleFactor, config.CausalOffset, config.AudioSamplingRate, config.AudioHopLength);

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

        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");
    }

    /// <summary>Always-resident (non-block) weights — proj_in/out, the global AdaLN-Single modulation tables. Touched
    /// every step regardless of the executing block, so the streaming controller doesn't manage them; preload eagerly.</summary>
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        foreach (Tensor? t in new[] { _projInW, _projInB, _audioProjInW, _audioProjInB,
            _projOutW, _projOutB, _audioProjOutW, _audioProjOutB, _scaleShift, _audioScaleShift })
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

        // proj_in patchify projections.
        Tensor hidden = new Tensor(new TensorShape(sv, v), DType.F32);
        backend.Linear(hidden, videoTokens, _projInW!, _projInB);
        Tensor audioHidden = new Tensor(new TensorShape(sa, a), DType.F32);
        backend.Linear(audioHidden, audioTokens, _audioProjInW!, _audioProjInB);

        // Global modulation tables + the two embedded-timesteps used by the output layer.
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

        (Tensor vCos, Tensor vSin) = _videoRope.BuildVideo(grid.Frames, grid.Height, grid.Width, fps);
        (Tensor aCos, Tensor aSin) = _audioRope.BuildAudio(audioFrames);
        (Tensor cvCos, Tensor cvSin) = _caVideoRope.BuildVideo(grid.Frames, grid.Height, grid.Width, fps);

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

        for (int i = 0; i < _blocks.Length; i++)
        {
            BeforeBlockForward?.Invoke(i);
            (hidden, audioHidden) = _blocks[i].Forward(backend, hidden, audioHidden, ctx);
        }

        foreach (Tensor? t in new[] { tVideo, tAudio, tPromptV, tPromptA, tCaVss, tCaAss, tCaVGate, tCaAGate,
            vCos, vSin, aCos, aSin, cvCos, cvSin })
            t?.Dispose();

        Tensor video = OutputLayer(backend, hidden, embV, _scaleShift!, _projOutW!, _projOutB, sv, v, _config.OutChannels);
        Tensor audio = OutputLayer(backend, audioHidden, embA, _audioScaleShift!, _audioProjOutW!, _audioProjOutB,
            sa, a, _config.AudioOutChannels);
        hidden.Dispose(); audioHidden.Dispose(); embV.Dispose(); embA.Dispose();
        return (video, audio);
    }

    /// <summary>AdaLN-Single output: <c>shift/scale = scale_shift_table + embedded</c> ([dim], broadcast over the
    /// sequence), LayerNorm-no-affine, <c>·(1+scale)+shift</c>, then <c>proj_out</c>.</summary>
    private static Tensor OutputLayer(IBackend backend, Tensor hidden, Tensor embedded, Tensor scaleShift,
        Tensor projW, Tensor? projB, int s, int dim, int outChannels)
    {
        float* ss = (float*)scaleShift.DataPointer;
        float* em = (float*)embedded.DataPointer;
        Tensor normed = new Tensor(new TensorShape(s, dim), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, hidden, 1, s, dim, 1e-6f);
        float* np = (float*)normed.DataPointer;
        for (int i = 0; i < s; i++)
            for (int d = 0; d < dim; d++)
            {
                float shift = ss[d] + em[d];
                float scale = ss[dim + d] + em[d];
                np[(long)i * dim + d] = np[(long)i * dim + d] * (1f + scale) + shift;
            }
        Tensor outVel = new Tensor(new TensorShape(s, outChannels), DType.F32);
        backend.Linear(outVel, normed, projW, projB);
        normed.Dispose();
        return outVel;
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
            _projInW = _projInB = _audioProjInW = _audioProjInB = null;
            _projOutW = _projOutB = _audioProjOutW = _audioProjOutB = _scaleShift = _audioScaleShift = null;
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

            Tensor mod = new Tensor(new TensorShape(_numParams * _dim), DType.F32);
            Buffer.MemoryCopy((float*)modFlat.DataPointer, (float*)mod.DataPointer,
                (long)_numParams * _dim * 4, (long)_numParams * _dim * 4);
            modFlat.Dispose();

            Tensor embedded = new Tensor(new TensorShape(_dim), DType.F32);
            Buffer.MemoryCopy((float*)embedded2d.DataPointer, (float*)embedded.DataPointer, (long)_dim * 4, (long)_dim * 4);
            embedded2d.Dispose();
            return (mod, embedded);
        }
    }
}
