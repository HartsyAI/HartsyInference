using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>ZipVoice's text-conditioning + flow-matching-velocity machinery (<c>zipvoice/models/zipvoice.py</c>),
/// everything outside the two <see cref="TTSZipformer"/> backbones themselves. There is no trained duration
/// predictor — token→frame alignment is "average upsampling": each token is assumed to span
/// <c>totalFrames / tokenCount</c> frames (floor division), with the leftover remainder frames at the tail
/// mapped to an extra PAD token appended to the sequence before text-encoding (so those frames pick up the
/// pad embedding, not a repeat of the last real token — an easy-to-miss upstream quirk, preserved here
/// exactly). Reference-audio conditioning is a hard prompt-prefix mask (kept verbatim for the prompt region,
/// zeroed beyond it), concatenated with the text condition and the noisy flow-matching state along the
/// feature axis: <c>[xt | text_condition | speech_condition]</c>, each 100-dim, feeding the 300-dim
/// <c>fm_decoder.in_proj</c>.</summary>
public sealed class ZipVoiceModel : IDisposable
{
    private readonly ZipVoiceConfig _cfg;
    private readonly TTSZipformer _textEncoder;
    private readonly TTSZipformer _fmDecoder;
    private Tensor? _embedTable;
    private int _disposed;

    public ZipVoiceModel(ZipVoiceConfig cfg)
    {
        _cfg = cfg;
        _textEncoder = new TTSZipformer(
            inDim: cfg.TextEmbedDim, dim: cfg.TextEncoderDim, outDim: cfg.FeatDim,
            timeEmbedDim: cfg.TimeEmbedDim, useTimeEmb: false,
            downsamplingFactor: [1], numLayers: [cfg.TextEncoderNumLayers], cnnKernel: [cfg.TextEncoderCnnModuleKernel],
            numHeads: cfg.TextEncoderNumHeads, queryHeadDim: cfg.QueryHeadDim, posHeadDim: cfg.PosHeadDim,
            posDim: cfg.PosDim, valueHeadDim: cfg.ValueHeadDim, feedforwardDim: cfg.TextEncoderFeedforwardDim);

        _fmDecoder = new TTSZipformer(
            inDim: cfg.FeatDim * 3, dim: cfg.FmDecoderDim, outDim: cfg.FeatDim,
            timeEmbedDim: cfg.TimeEmbedDim, useTimeEmb: true,
            downsamplingFactor: cfg.FmDecoderDownsamplingFactor, numLayers: cfg.FmDecoderNumLayers,
            cnnKernel: cfg.FmDecoderCnnModuleKernel, numHeads: cfg.FmDecoderNumHeads, queryHeadDim: cfg.QueryHeadDim,
            posHeadDim: cfg.PosHeadDim, posDim: cfg.PosDim, valueHeadDim: cfg.ValueHeadDim,
            feedforwardDim: cfg.FmDecoderFeedforwardDim);
    }

    public unsafe void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _embedTable = WhisperOps.EnsureF32(w["embed.weight"]);
        _textEncoder.LoadWeights(w, "text_encoder");
        _fmDecoder.LoadWeights(w, "fm_decoder");
    }

    /// <summary>Predicted TARGET-region frame count (upstream <c>duration="predict"</c>): the reference audio's
    /// measured frames-per-token rate, scaled by the target token count and <paramref name="speed"/>.</summary>
    public static int PredictTargetFrames(int promptFrames, int promptTokenCount, int targetTokenCount, float speed)
        => (int)Math.Ceiling((double)promptFrames / promptTokenCount * targetTokenCount / Math.Max(1e-6f, speed));

    /// <summary>Embeds the joint (prompt+target) token sequence with an appended pad token, then runs the
    /// text_encoder. Returns <c>[1, L+1, feat_dim]</c> per-token hidden states (index <c>L</c> is the pad
    /// token's embedding, used by leftover tail frames in <see cref="BuildFrameToTokenIndex"/>).</summary>
    public unsafe Tensor EmbedAndEncodeText(IBackend backend, int[] jointTokenIds, int padId)
    {
        int l = jointTokenIds.Length;
        int textEmbedDim = _cfg.TextEmbedDim;
        Tensor embedded = new(new TensorShape(1, l + 1, textEmbedDim), DType.F32);
        float* ep = (float*)embedded.DataPointer;
        float* table = (float*)_embedTable!.DataPointer;
        for (int i = 0; i < l; i++)
            Buffer.MemoryCopy(table + (long)jointTokenIds[i] * textEmbedDim, ep + (long)i * textEmbedDim,
                textEmbedDim * sizeof(float), textEmbedDim * sizeof(float));
        Buffer.MemoryCopy(table + (long)padId * textEmbedDim, ep + (long)l * textEmbedDim,
            textEmbedDim * sizeof(float), textEmbedDim * sizeof(float));

        Tensor hidden = _textEncoder.Forward(backend, embedded, l + 1, timestep: null);
        embedded.Dispose();
        return hidden;
    }

    /// <summary>"Average upsampling": frame <c>t</c> maps to token index <c>t / avg</c> for the first
    /// <c>avg*jointTokenCount</c> frames (<c>avg = totalFrames / jointTokenCount</c>, floor division); any
    /// leftover tail frames map to index <c>jointTokenCount</c> (the appended pad token from
    /// <see cref="EmbedAndEncodeText"/>).</summary>
    public static int[] BuildFrameToTokenIndex(int jointTokenCount, int totalFrames)
    {
        int[] idx = new int[totalFrames];
        int avg = jointTokenCount > 0 ? totalFrames / jointTokenCount : 0;
        int cur = 0;
        if (avg > 0)
            for (int i = 0; i < jointTokenCount; i++)
                for (int k = 0; k < avg && cur < totalFrames; k++)
                    idx[cur++] = i;
        for (; cur < totalFrames; cur++) idx[cur] = jointTokenCount;
        return idx;
    }

    /// <summary>Gathers <paramref name="textHidden"/> (<c>[1, L+1, feat_dim]</c>) per <paramref
    /// name="frameToTokenIndex"/> into <c>[1, totalFrames, feat_dim]</c>.</summary>
    public unsafe Tensor GatherTextCondition(Tensor textHidden, int[] frameToTokenIndex)
    {
        int totalFrames = frameToTokenIndex.Length;
        int featDim = _cfg.FeatDim;
        Tensor output = new(new TensorShape(1, totalFrames, featDim), DType.F32);
        float* op = (float*)output.DataPointer;
        float* hp = (float*)textHidden.DataPointer;
        for (int t = 0; t < totalFrames; t++)
            Buffer.MemoryCopy(hp + (long)frameToTokenIndex[t] * featDim, op + (long)t * featDim,
                featDim * sizeof(float), featDim * sizeof(float));
        return output;
    }

    /// <summary>One flow-matching velocity forward: concatenates <c>[xt | textCondition | speechCondition]</c>
    /// (each <c>[1, T, feat_dim]</c>) along the feature axis and runs <c>fm_decoder</c>. Returns
    /// <c>[1, T, feat_dim]</c>.</summary>
    public unsafe Tensor ForwardVelocity(IBackend backend, Tensor xt, Tensor textCondition, Tensor speechCondition, int t, float timestep)
    {
        int featDim = _cfg.FeatDim;
        Tensor concat = new(new TensorShape(1, t, featDim * 3), DType.F32);
        float* cp = (float*)concat.DataPointer;
        float* xp = (float*)xt.DataPointer;
        float* tp = (float*)textCondition.DataPointer;
        float* sp = (float*)speechCondition.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long dst = (long)i * featDim * 3;
            long src = (long)i * featDim;
            Buffer.MemoryCopy(xp + src, cp + dst, featDim * sizeof(float), featDim * sizeof(float));
            Buffer.MemoryCopy(tp + src, cp + dst + featDim, featDim * sizeof(float), featDim * sizeof(float));
            Buffer.MemoryCopy(sp + src, cp + dst + 2 * featDim, featDim * sizeof(float), featDim * sizeof(float));
        }

        Tensor velocity = _fmDecoder.Forward(backend, concat, t, timestep);
        concat.Dispose();
        return velocity;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedTable is not null) yield return _embedTable;
        foreach (Tensor t in _textEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _fmDecoder.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
