using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Port of the official <c>AudioGenerationNetwork</c> (<c>audio_model.py</c>) inference path: input projections
/// (V1 SELU / V2 SiLU), CLIP + visual fusion, Synchformer position embedding and nearest-exact resampling, timestep
/// embedding, joint blocks, fused blocks and the adaLN final convolution, with classifier-free guidance in
/// <see cref="OdeWrapper"/>. The REPA head and the pooled multimodal embedding are training-only outputs and are not
/// loaded. All tensors are F32 <c>[B, T, C]</c>.</summary>
public sealed class ControlFoleyNetwork : IDisposable
{
    private const float RopeTheta = 10000f;

    private sealed record Projection(Tensor First, Tensor? FirstBias, Func<Tensor, bool>? Activation, Tensor W1, Tensor W2, Tensor W3);

    private ControlFoleyNetworkConfig _cfg;
    private ControlFoleyRope _latentRope, _clipRope;
    private ControlFoleyJointBlock[] _joint = [];
    private ControlFoleyBlock[] _fused = [];
    private Projection? _audioIn, _clipIn, _visualIn, _syncIn, _textIn, _clapIn, _timbreIn;
    private Tensor? _clipCondW, _clipCondB, _textCondW, _textCondB, _timbreCondW, _timbreCondB;
    private Tensor? _globalW1, _globalW2, _globalW3, _syncPos, _tW0, _tB0, _tW2, _tB2, _finalAdaW, _finalAdaB, _finalConvW, _finalConvB;
    private Tensor? _latentMean, _latentStd, _emptyString, _emptyClip, _emptyVisual, _emptySync, _emptyAudio, _emptyTimbre;
    private float[] _timeFreqs = [];
    private readonly List<Tensor> _owned = [];
    private int _disposed;

    /// <summary>Creates an unloaded network; call <see cref="LoadWeights"/> before use.</summary>
    public ControlFoleyNetwork(ControlFoleyNetworkConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _cfg = config;
        (_latentRope, _clipRope) = BuildRopes(config);
    }

    /// <summary>Active hyper-parameters (sequence lengths follow <see cref="UpdateSequenceLengths(ControlFoleyNetworkConfig)"/>).</summary>
    public ControlFoleyNetworkConfig Config => _cfg;

    /// <summary>Rotary table applied to the latent stream.</summary>
    public ControlFoleyRope LatentRope => _latentRope;

    /// <summary>Rotary table applied to the CLIP stream (frequencies scaled to the latent rate).</summary>
    public ControlFoleyRope ClipRope => _clipRope;

    private static (ControlFoleyRope Latent, ControlFoleyRope Clip) BuildRopes(ControlFoleyNetworkConfig c)
    {
        int headDim = c.HiddenDim / c.NumHeads;
        return (ControlFoleyRope.Compute(c.LatentSeqLen, headDim, RopeTheta, 1f),
            ControlFoleyRope.Compute(c.ClipSeqLen, headDim, RopeTheta, 1f * c.LatentSeqLen / c.ClipSeqLen));
    }

    /// <summary>Port of <c>update_seq_lengths</c>: changes the clip duration the network is run at.</summary>
    public void UpdateSequenceLengths(ControlFoleyNetworkConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        if (config.HiddenDim != _cfg.HiddenDim || config.Depth != _cfg.Depth || config.TextSeqLen != _cfg.TextSeqLen)
        {
            throw new ArgumentException("Only the latent, CLIP, visual and sync lengths may change.");
        }

        _cfg = config;
        (_latentRope, _clipRope) = BuildRopes(config);
    }

    /// <summary>Resizes for another clip duration using the official temporal configuration.</summary>
    public void UpdateSequenceLengths(ControlFoleyTemporalConfig temporal) => UpdateSequenceLengths(_cfg.WithSequenceLengths(temporal));

    /// <summary>Binds the checkpoint tensors (official key names, F32; other dtypes are cast). Tensors that are already
    /// F32 stay owned by the caller and must outlive the network.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _audioIn = LoadProjection(w, "audio_input_proj");
        _clipIn = LoadProjection(w, "clip_input_proj");
        _visualIn = LoadProjection(w, "visual_input_proj");
        _syncIn = LoadProjection(w, "sync_input_proj");
        _textIn = LoadProjection(w, "text_input_proj");
        _clapIn = LoadProjection(w, "clap_input_proj");
        _timbreIn = LoadProjection(w, "timbre_input_proj");
        _clipCondW = T(w, "clip_cond_proj.weight");
        _clipCondB = T(w, "clip_cond_proj.bias");
        _textCondW = T(w, "text_cond_proj.weight");
        _textCondB = T(w, "text_cond_proj.bias");
        _timbreCondW = T(w, "timbre_cond_proj.weight");
        _timbreCondB = T(w, "timbre_cond_proj.bias");
        _globalW1 = T(w, "global_cond_mlp.w1.weight");
        _globalW2 = T(w, "global_cond_mlp.w2.weight");
        _globalW3 = T(w, "global_cond_mlp.w3.weight");
        _syncPos = T(w, "sync_pos_emb");
        _tW0 = T(w, "t_embed.mlp.0.weight");
        _tB0 = T(w, "t_embed.mlp.0.bias");
        _tW2 = T(w, "t_embed.mlp.2.weight");
        _tB2 = T(w, "t_embed.mlp.2.bias");
        _finalAdaW = T(w, "final_layer.adaLN_modulation.1.weight");
        _finalAdaB = T(w, "final_layer.adaLN_modulation.1.bias");
        _finalConvW = T(w, "final_layer.conv.weight");
        _finalConvB = T(w, "final_layer.conv.bias");
        _latentMean = T(w, "latent_mean");
        _latentStd = T(w, "latent_std");
        _emptyString = T(w, "empty_string_feat");
        _emptyClip = T(w, "empty_clip_feat");
        _emptyVisual = T(w, "empty_visual_feat");
        _emptySync = T(w, "empty_sync_feat");
        _emptyAudio = T(w, "empty_audio_feat");
        _emptyTimbre = T(w, "empty_timbre_feat");

        int freqEmbed = (int)_tW0.Shape[1];
        float maxPeriod = _cfg.V2 ? 1f : 10000f;
        _timeFreqs = new float[freqEmbed / 2];
        for (int i = 0; i < _timeFreqs.Length; i++)
        {
            _timeFreqs[i] = 1f / MathF.Pow(10000f, 2f * i / freqEmbed);
            _timeFreqs[i] *= 10000f / maxPeriod;
        }

        _joint = new ControlFoleyJointBlock[_cfg.JointDepth];
        for (int i = 0; i < _joint.Length; i++)
        {
            _joint[i] = ControlFoleyJointBlock.Load(w, $"joint_blocks.{i}", _cfg.NumHeads, preOnly: i == _joint.Length - 1);
        }

        _fused = new ControlFoleyBlock[_cfg.FusedDepth];
        for (int i = 0; i < _fused.Length; i++)
        {
            _fused[i] = ControlFoleyBlock.Load(w, $"fused_blocks.{i}", _cfg.NumHeads, preOnly: false);
        }

        Validate();
    }

    private void Validate()
    {
        if (_latentMean!.ElementCount != _cfg.LatentDim || _emptyString!.Shape[0] != _cfg.TextSeqLen ||
            _syncPos!.ElementCount != 8L * _cfg.SyncDim || _finalConvW!.Shape[0] != _cfg.LatentDim ||
            _emptyClip!.Shape[1] != _cfg.ClipDim || _emptyTimbre!.Shape[1] != _cfg.TimbreDim)
        {
            throw new InvalidDataException("ControlFoley checkpoint does not match the network configuration.");
        }
    }

    private Tensor T(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        if (!w.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"Missing ControlFoley weight '{key}'.");
        }

        return ControlFoleyOps.PrepareWeight(t, _owned);
    }

    private Projection LoadProjection(IReadOnlyDictionary<string, Tensor> w, string name)
    {
        bool selu = !_cfg.V2 && name is "audio_input_proj" or "sync_input_proj";
        bool silu = _cfg.V2;
        int mlp = selu || silu ? 2 : 1;
        Func<Tensor, bool>? act = selu ? x => { ControlFoleyOps.Selu(x); return true; } : null;
        return new Projection(T(w, $"{name}.0.weight"), T(w, $"{name}.0.bias"), act, T(w, $"{name}.{mlp}.w1.weight"),
            T(w, $"{name}.{mlp}.w2.weight"), T(w, $"{name}.{mlp}.w3.weight"));
    }

    private Tensor Apply(IBackend backend, Projection p, Tensor x)
    {
        Tensor first = ControlFoleyOps.Project(backend, x, p.First, p.FirstBias);
        if (_cfg.V2)
        {
            backend.Silu(first, first);
        }
        else
        {
            p.Activation?.Invoke(first);
        }

        Tensor o = ControlFoleyOps.SwiGlu(backend, first, p.W1, p.W2, p.W3);
        first.Dispose();
        return o;
    }

    // ---- conditions -----------------------------------------------------------------------------------------

    /// <summary>Projects the six raw condition features (<c>[B, T, C]</c>, lengths per <see cref="Config"/>) once per generation.</summary>
    public ControlFoleyConditions PreprocessConditions(IBackend backend, Tensor clipF, Tensor visualF, Tensor syncF, Tensor textF,
        Tensor audioF, Tensor timbreF)
    {
        CheckShape(clipF, _cfg.ClipSeqLen, _cfg.ClipDim, "clip");
        CheckShape(visualF, _cfg.VisualSeqLen, _cfg.VisualDim, "visual");
        CheckShape(syncF, _cfg.SyncSeqLen, _cfg.SyncDim, "sync");
        CheckShape(textF, _cfg.TextSeqLen, _cfg.TextDim, "text");
        CheckShape(audioF, _cfg.AudioSeqLen, _cfg.AudioDim, "audio");
        CheckShape(timbreF, _cfg.TimbreSeqLen, _cfg.TimbreDim, "timbre");
        long bs = clipF.Shape[0];
        if (visualF.Shape[0] != bs || syncF.Shape[0] != bs || textF.Shape[0] != bs || audioF.Shape[0] != bs || timbreF.Shape[0] != bs)
        {
            throw new ArgumentException("All condition features must share the batch size.");
        }

        Tensor clipFC, textFC, timbre;
        Tensor clipTokens = FuseClipVisual(backend, clipF, visualF);
        Tensor sync = ProjectSync(backend, syncF);
        (Tensor text, textFC) = ProjectText(backend, textF);
        Tensor audio = Apply(backend, _clapIn!, audioF);
        using (Tensor timbreTokens = Apply(backend, _timbreIn!, timbreF))
        using (Tensor pooled = ControlFoleyOps.MeanOverTime(timbreTokens))
        {
            timbre = ControlFoleyOps.Linear(backend, pooled, _timbreCondW!, _timbreCondB);
        }

        using (Tensor pooled = ControlFoleyOps.MeanOverTime(clipTokens))
        {
            clipFC = ControlFoleyOps.Linear(backend, pooled, _clipCondW!, _clipCondB);
        }

        return new ControlFoleyConditions(clipTokens, sync, text, audio, timbre, clipFC, textFC);
    }

    private (Tensor Text, Tensor Pooled) ProjectText(IBackend backend, Tensor textF)
    {
        Tensor text = Apply(backend, _textIn!, textF);
        using Tensor pooled = ControlFoleyOps.MeanOverTime(text);
        return (text, ControlFoleyOps.Linear(backend, pooled, _textCondW!, _textCondB));
    }

    private Tensor FuseClipVisual(IBackend backend, Tensor clipF, Tensor visualF)
    {
        Tensor clip = Apply(backend, _clipIn!, clipF);
        using Tensor visual = Apply(backend, _visualIn!, visualF);
        if (visual.Shape[1] == clip.Shape[1])
        {
            ControlFoleyOps.AddBroadcastTime(clip, visual);
            return clip;
        }

        using Tensor resized = ControlFoleyOps.InterpolateLinear(visual, (int)clip.Shape[1]);
        ControlFoleyOps.AddBroadcastTime(clip, resized);
        return clip;
    }

    private Tensor ProjectSync(IBackend backend, Tensor syncF)
    {
        using Tensor withPos = ControlFoleyOps.Clone(syncF);
        int b = (int)syncF.Shape[0], t = (int)syncF.Shape[1], d = (int)syncF.Shape[2];
        unsafe
        {
            float* p = (float*)withPos.DataPointer, pos = (float*)_syncPos!.DataPointer;
            for (int bi = 0; bi < b; bi++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    float* row = p + ((long)bi * t + ti) * d, e = pos + (ti % 8) * d;
                    for (int c = 0; c < d; c++)
                    {
                        row[c] += e[c];
                    }
                }
            }
        }

        using Tensor projected = Apply(backend, _syncIn!, withPos);
        return ControlFoleyOps.InterpolateNearestExact(projected, _cfg.LatentSeqLen);
    }

    private void CheckShape(Tensor x, int length, int dim, string name)
    {
        if (x.DType != DType.F32 || x.Shape.Rank != 3 || x.Shape[1] != length || x.Shape[2] != dim)
        {
            throw new ArgumentException($"{name} features must be F32 [B, {length}, {dim}], got {x.DType} {x.Shape}.");
        }
    }

    /// <summary>Learned "no text" sequence, <c>[bs, TextSeqLen, TextDim]</c>.</summary>
    public unsafe Tensor GetEmptyStringSequence(int bs)
    {
        Tensor o = ControlFoleyOps.New(bs, _emptyString!.Shape[0], _emptyString.Shape[1]);
        long bytes = _emptyString.ElementCount * 4;
        for (int b = 0; b < bs; b++)
        {
            Buffer.MemoryCopy((void*)_emptyString.DataPointer, (byte*)o.DataPointer + b * bytes, bytes, bytes);
        }

        return o;
    }

    /// <summary>Learned "no CLIP" sequence, <c>[bs, ClipSeqLen, ClipDim]</c>.</summary>
    public Tensor GetEmptyClipSequence(int bs) => Repeat(_emptyClip!, bs, _cfg.ClipSeqLen);

    /// <summary>Learned "no visual" sequence, <c>[bs, VisualSeqLen, VisualDim]</c>.</summary>
    public Tensor GetEmptyVisualSequence(int bs) => Repeat(_emptyVisual!, bs, _cfg.VisualSeqLen);

    /// <summary>Learned "no sync" sequence, <c>[bs, SyncSeqLen, SyncDim]</c>.</summary>
    public Tensor GetEmptySyncSequence(int bs) => Repeat(_emptySync!, bs, _cfg.SyncSeqLen);

    /// <summary>Learned "no reference audio" sequence, <c>[bs, AudioSeqLen, AudioDim]</c>.</summary>
    public Tensor GetEmptyAudioSequence(int bs) => Repeat(_emptyAudio!, bs, _cfg.AudioSeqLen);

    /// <summary>Learned "no timbre" sequence, <c>[bs, TimbreSeqLen, TimbreDim]</c>.</summary>
    public Tensor GetEmptyTimbreSequence(int bs) => Repeat(_emptyTimbre!, bs, _cfg.TimbreSeqLen);

    private static unsafe Tensor Repeat(Tensor row, int bs, int length)
    {
        int d = (int)row.Shape[1];
        Tensor o = ControlFoleyOps.New(bs, length, d);
        float* op = (float*)o.DataPointer;
        for (long i = 0; i < (long)bs * length; i++)
        {
            Buffer.MemoryCopy((void*)row.DataPointer, op + i * d, d * 4L, d * 4L);
        }

        return o;
    }

    /// <summary>The unconditional branch of classifier-free guidance: every modality replaced by its learned empty
    /// feature (text optionally by <paramref name="negativeText"/>, <c>[bs, TextSeqLen, TextDim]</c>).</summary>
    public ControlFoleyConditions GetEmptyConditions(IBackend backend, int bs, Tensor? negativeText = null)
    {
        using Tensor text = GetEmptyStringSequence(1);
        using Tensor clip = GetEmptyClipSequence(1), visual = GetEmptyVisualSequence(1), sync = GetEmptySyncSequence(1);
        using Tensor audio = GetEmptyAudioSequence(1), timbre = GetEmptyTimbreSequence(1);
        using ControlFoleyConditions one = PreprocessConditions(backend, clip, visual, sync, text, audio, timbre);
        if (negativeText is null)
        {
            return new ControlFoleyConditions(ControlFoleyOps.ExpandBatch(one.ClipF, bs), ControlFoleyOps.ExpandBatch(one.SyncF, bs),
                ControlFoleyOps.ExpandBatch(one.TextF, bs), ControlFoleyOps.ExpandBatch(one.AudioF, bs),
                ControlFoleyOps.ExpandBatch(one.TimbreF, bs), ControlFoleyOps.ExpandBatch(one.ClipFC, bs),
                ControlFoleyOps.ExpandBatch(one.TextFC, bs));
        }

        CheckShape(negativeText, _cfg.TextSeqLen, _cfg.TextDim, "negative text");
        if (negativeText.Shape[0] != bs)
        {
            throw new ArgumentException("Negative text features must have one row per batch element.");
        }

        (Tensor negText, Tensor negPooled) = ProjectText(backend, negativeText);
        return new ControlFoleyConditions(ControlFoleyOps.ExpandBatch(one.ClipF, bs), ControlFoleyOps.ExpandBatch(one.SyncF, bs), negText,
            ControlFoleyOps.ExpandBatch(one.AudioF, bs), ControlFoleyOps.ExpandBatch(one.TimbreF, bs),
            ControlFoleyOps.ExpandBatch(one.ClipFC, bs), negPooled);
    }

    // ---- forward --------------------------------------------------------------------------------------------

    /// <summary>Velocity <c>[B, LatentSeqLen, LatentDim]</c> for a latent at per-row timesteps <paramref name="t"/>.
    /// <paramref name="multimodal"/> receives the pooled global condition <c>[B, 1, hidden]</c> (a diagnostic output).</summary>
    public Tensor PredictFlow(IBackend backend, Tensor latent, ReadOnlySpan<float> t, ControlFoleyConditions cond, out Tensor multimodal)
    {
        CheckShape(latent, _cfg.LatentSeqLen, _cfg.LatentDim, "latent");
        int bs = (int)latent.Shape[0];
        if (t.Length != bs || cond.Batch != bs)
        {
            throw new ArgumentException("Timesteps and conditions must match the latent batch.");
        }

        using Tensor sum = ControlFoleyOps.Clone(cond.ClipFC);
        ControlFoleyOps.AddBroadcastTime(sum, cond.TextFC);
        ControlFoleyOps.AddBroadcastTime(sum, cond.TimbreF);
        multimodal = ControlFoleyOps.SwiGlu(backend, sum, _globalW1!, _globalW2!, _globalW3!);
        using Tensor globalC = TimestepEmbedding(backend, t);
        ControlFoleyOps.AddBroadcastTime(globalC, multimodal);
        using Tensor extendedC = ControlFoleyOps.Clone(cond.SyncF);
        ControlFoleyOps.AddBroadcastTime(extendedC, globalC);

        Tensor x = Apply(backend, _audioIn!, latent);
        using Tensor clip = ControlFoleyOps.Clone(cond.ClipF), text = ControlFoleyOps.Clone(cond.TextF), audio = ControlFoleyOps.Clone(cond.AudioF);
        foreach (ControlFoleyJointBlock block in _joint)
        {
            block.Forward(backend, x, clip, audio, text, globalC, extendedC, _latentRope, _clipRope);
        }

        foreach (ControlFoleyBlock block in _fused)
        {
            block.Forward(backend, x, extendedC, _latentRope);
        }

        using (x)
        using (Tensor activated = ControlFoleyOps.Silu(backend, extendedC))
        using (Tensor modulation = ControlFoleyOps.Linear(backend, activated, _finalAdaW!, _finalAdaB))
        using (Tensor modulated = ControlFoleyOps.NormModulate(backend, x, ControlFoleyOps.ChunkOf(modulation, 0, 2),
            ControlFoleyOps.ChunkOf(modulation, 1, 2)))
        {
            return ControlFoleyOps.ConvLastChannel(backend, modulated, _finalConvW!, _finalConvB);
        }
    }

    private unsafe Tensor TimestepEmbedding(IBackend backend, ReadOnlySpan<float> t)
    {
        int half = _timeFreqs.Length;
        using Tensor emb = ControlFoleyOps.New(t.Length, 1, 2 * half);
        float* p = (float*)emb.DataPointer;
        for (int b = 0; b < t.Length; b++)
        {
            for (int i = 0; i < half; i++)
            {
                float arg = t[b] * _timeFreqs[i];
                p[b * 2 * half + i] = MathF.Cos(arg);
                p[b * 2 * half + half + i] = MathF.Sin(arg);
            }
        }

        Tensor h = ControlFoleyOps.Linear(backend, emb, _tW0!, _tB0);
        backend.Silu(h, h);
        Tensor o = ControlFoleyOps.Linear(backend, h, _tW2!, _tB2);
        h.Dispose();
        return o;
    }

    /// <summary>Port of <c>ode_wrapper</c>: the velocity at time <paramref name="t"/>, guided as
    /// <c>cfg * v + (1 - cfg) * v_empty</c> when <paramref name="cfgStrength"/> is at least 1.</summary>
    public unsafe Tensor OdeWrapper(IBackend backend, float t, Tensor latent, ControlFoleyConditions cond, ControlFoleyConditions empty,
        float cfgStrength)
    {
        float[] ts = new float[latent.Shape[0]];
        Array.Fill(ts, t);
        Tensor flow = PredictFlow(backend, latent, ts, cond, out Tensor multimodal);
        multimodal.Dispose();
        if (cfgStrength < 1f)
        {
            return flow;
        }

        using Tensor flowEmpty = PredictFlow(backend, latent, ts, empty, out Tensor emptyMultimodal);
        emptyMultimodal.Dispose();
        float* f = (float*)flow.DataPointer, e = (float*)flowEmpty.DataPointer;
        for (long i = 0; i < flow.ElementCount; i++)
        {
            f[i] = cfgStrength * f[i] + (1f - cfgStrength) * e[i];
        }

        return flow;
    }

    /// <summary>In-place <c>(x - mean) / std</c> of a <c>[B, N, LatentDim]</c> latent.</summary>
    public unsafe void Normalize(Tensor x) => ScaleShift(x, false);

    /// <summary>In-place <c>x * std + mean</c> of a <c>[B, N, LatentDim]</c> latent (the step before the VAE decode).</summary>
    public unsafe void Unnormalize(Tensor x) => ScaleShift(x, true);

    private unsafe void ScaleShift(Tensor x, bool inverse)
    {
        int c = _cfg.LatentDim;
        if (x.Shape[2] != c)
        {
            throw new ArgumentException($"Expected {c} latent channels, got {x.Shape}.");
        }

        float* p = (float*)x.DataPointer, mean = (float*)_latentMean!.DataPointer, std = (float*)_latentStd!.DataPointer;
        for (long i = 0; i < x.ElementCount; i++)
        {
            int ch = (int)(i % c);
            p[i] = inverse ? p[i] * std[ch] + mean[ch] : (p[i] - mean[ch]) / std[ch];
        }
    }

    /// <summary>Every bound weight tensor (for residency planning).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Projection? p in new[] { _audioIn, _clipIn, _visualIn, _syncIn, _textIn, _clapIn, _timbreIn })
        {
            if (p is not null)
            {
                foreach (Tensor t in new[] { p.First, p.W1, p.W2, p.W3 })
                {
                    yield return t;
                }
            }
        }

        foreach (ControlFoleyJointBlock b in _joint)
        {
            foreach (Tensor t in b.Weights())
            {
                yield return t;
            }
        }

        foreach (ControlFoleyBlock b in _fused)
        {
            foreach (Tensor t in b.Weights())
            {
                yield return t;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (Tensor t in _owned.Concat(_joint.SelectMany(j => j.OwnedTensors)).Concat(_fused.SelectMany(f => f.OwnedTensors)))
        {
            t.Dispose();
        }
    }
}
