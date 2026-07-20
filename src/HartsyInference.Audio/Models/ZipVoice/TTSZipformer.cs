using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>The Zipformer backbone (<c>TTSZipformer</c>), used for BOTH of ZipVoice's two backbones:
/// <c>fm_decoder</c> (the flow-matching velocity network, time-conditioned, 5 stages,
/// <c>[1,2,4,2,1]</c> downsampling — U-Net-shaped frame-rate curve) and <c>text_encoder</c> (unconditional on
/// the FM timestep, single stage, no downsampling). Sequential pipeline:
/// <code>
/// h = in_proj(x)
/// timeEmb = useTimeEmb ? time_embed(sinusoidal(t)) : null     // Linear→SwooshR→Linear, shared across stages
/// for each of the 5 (or 1) stages: h = stage(h, timeEmb)      // each stage re-projects timeEmb itself
/// out = out_proj(h)
/// </code>
/// Batch is always 1 — this port has no padding-mask/attention-mask support (inference-only, single
/// utterance).</summary>
internal sealed unsafe class TTSZipformer
{
    private readonly int _inDim, _dim, _outDim, _timeEmbedDim;
    private readonly bool _useTimeEmb;
    private readonly ZipformerEncoderStage[] _stages;
    private Tensor? _inProjW, _inProjB, _outProjW, _outProjB;
    private Tensor? _timeEmbed0W, _timeEmbed0B, _timeEmbed2W, _timeEmbed2B;

    public TTSZipformer(int inDim, int dim, int outDim, int timeEmbedDim, bool useTimeEmb,
        int[] downsamplingFactor, int[] numLayers, int[] cnnKernel,
        int numHeads, int queryHeadDim, int posHeadDim, int posDim, int valueHeadDim, int feedforwardDim)
    {
        _inDim = inDim; _dim = dim; _outDim = outDim; _timeEmbedDim = timeEmbedDim; _useTimeEmb = useTimeEmb;

        int numStages = downsamplingFactor.Length;
        _stages = new ZipformerEncoderStage[numStages];
        for (int i = 0; i < numStages; i++)
            _stages[i] = new ZipformerEncoderStage(dim, numLayers[i], downsamplingFactor[i], cnnKernel[i],
                numHeads, queryHeadDim, posHeadDim, posDim, valueHeadDim, feedforwardDim, timeEmbedDim, useTimeEmb);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _inProjW = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.weight"]);
        _inProjB = WhisperOps.EnsureF32(w[$"{prefix}.in_proj.bias"]);
        _outProjW = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.weight"]);
        _outProjB = WhisperOps.EnsureF32(w[$"{prefix}.out_proj.bias"]);

        for (int i = 0; i < _stages.Length; i++)
            _stages[i].LoadWeights(w, $"{prefix}.encoders.{i}");

        if (_useTimeEmb)
        {
            _timeEmbed0W = WhisperOps.EnsureF32(w[$"{prefix}.time_embed.0.weight"]);
            _timeEmbed0B = WhisperOps.EnsureF32(w[$"{prefix}.time_embed.0.bias"]);
            _timeEmbed2W = WhisperOps.EnsureF32(w[$"{prefix}.time_embed.2.weight"]);
            _timeEmbed2B = WhisperOps.EnsureF32(w[$"{prefix}.time_embed.2.bias"]);
        }
    }

    /// <summary><paramref name="x"/> is <c>[1, T, inDim]</c>. <paramref name="timestep"/> is required iff this
    /// instance was constructed with <c>useTimeEmb=true</c> (fm_decoder), ignored otherwise (text_encoder).
    /// Returns <c>[1, T, outDim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t, float? timestep)
    {
        Tensor h = WhisperOps.ProjectLinear(backend, x, _inProjW!, _inProjB, 1, t, _inDim, _dim);

        Tensor? timeEmb = null;
        if (_useTimeEmb)
        {
            if (timestep is null) throw new ArgumentException("timestep required when useTimeEmb=true.");
            timeEmb = ComputeTimeEmbed(backend, timestep.Value);
        }

        for (int i = 0; i < _stages.Length; i++)
        {
            Tensor next = _stages[i].Forward(backend, h, timeEmb, t);
            h.Dispose();
            h = next;
        }
        timeEmb?.Dispose();

        Tensor output = WhisperOps.ProjectLinear(backend, h, _outProjW!, _outProjB, 1, t, _dim, _outDim);
        h.Dispose();
        return output;
    }

    /// <summary>sinusoidal(t) [dim=timeEmbedDim] → Linear(dim, 2dim) → SwooshR → Linear(2dim, dim).
    /// Returns <c>[1, 1, timeEmbedDim]</c>.</summary>
    private Tensor ComputeTimeEmbed(IBackend backend, float timestep)
    {
        Tensor sin = new(new TensorShape(1, 1, _timeEmbedDim), DType.F32);
        float* sp = (float*)sin.DataPointer;
        int half = _timeEmbedDim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / half);
            float angle = timestep * freq;
            sp[i] = MathF.Cos(angle);
            sp[half + i] = MathF.Sin(angle);
        }

        Tensor hidden = WhisperOps.ProjectLinear(backend, sin, _timeEmbed0W!, _timeEmbed0B, 1, 1, _timeEmbedDim, _timeEmbedDim * 2);
        sin.Dispose();
        ZipformerScaling.SwooshRInPlace(hidden);
        Tensor output = WhisperOps.ProjectLinear(backend, hidden, _timeEmbed2W!, _timeEmbed2B, 1, 1, _timeEmbedDim * 2, _timeEmbedDim);
        hidden.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_inProjW, _inProjB, _outProjW, _outProjB, _timeEmbed0W, _timeEmbed0B, _timeEmbed2W, _timeEmbed2B];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (ZipformerEncoderStage s in _stages) foreach (Tensor t in s.EnumerateWeights()) yield return t;
    }
}
