using System.Diagnostics;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using MimiModel = HartsyInference.Audio.Models.Codecs.Mimi.Mimi;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Kyutai STT pipeline (delayed-streams): silence-pads the waveform, Mimi-encodes it to 32-codebook
/// codes, then runs the Helium backbone one frame at a time — summing the previous text token's embedding
/// with the 32 audio-code embeddings — and greedily emits one text token per frame. Returns the PAD-stripped
/// text token ids (caller SentencePiece-decodes; the Audio package carries no text-vocab dependency).</summary>
public sealed unsafe class KyutaiSttPipeline : IDisposable
{
    private readonly KyutaiSttConfig _cfg;
    private readonly KyutaiSttModel _model;
    private readonly MimiModel _codec;
    private int _disposed;

    public KyutaiSttPipeline(KyutaiSttConfig cfg)
    {
        _cfg = cfg;
        _model = new KyutaiSttModel(cfg);
        _codec = new MimiModel(cfg.Codec);
    }

    /// <summary>Loads the Helium backbone + shared embedding and the Mimi codec weights.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> backbone, IReadOnlyDictionary<string, Tensor> mimi,
        string backbonePrefix = "model")
    {
        _model.LoadWeights(backbone, backbonePrefix);
        _codec.LoadWeights(mimi);
    }

    /// <summary>Transcribes 24 kHz mono PCM to text token ids (PAD stripped). Input is resampled by the
    /// caller; this method only applies the silence prefix/delay padding the model expects.</summary>
    public int[] Transcribe(IBackend backend, ReadOnlySpan<float> pcm24k, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int sr = _cfg.Codec.SampleRate;
        int prefix = (int)MathF.Round(_cfg.AudioSilencePrefixSeconds * sr);
        int suffix = (int)MathF.Round(_cfg.AudioDelaySeconds * sr);
        int tPcm = prefix + pcm24k.Length + suffix;

        Tensor pcm = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* pp = (float*)pcm.DataPointer;
        pcm24k.CopyTo(new Span<float>(pp + prefix, pcm24k.Length));

        Tensor codes = _codec.Encode(backend, pcm, batch: 1, tPcm: tPcm);
        pcm.Dispose();
        int nQ = (int)codes.Shape[0];
        int tFrames = (int)codes.Shape[2];
        int* cp = (int*)codes.DataPointer;

        int cacheCap = Math.Min(_cfg.Helium.MaxPositionEmbeddings, tFrames + 4);
        using StreamingKvCache cache = new(_cfg.Helium.NumHiddenLayers, batch: 1,
            _cfg.Helium.NumKeyValueHeads, cacheCap, _cfg.Helium.HeadDim);

        int prevText = _cfg.TextPad;
        int kUse = Math.Min(nQ, _cfg.NumCodebooks);
        Span<int> frameCodes = stackalloc int[kUse];
        List<int> tokens = new(tFrames);

        for (int t = 0; t < tFrames && t < cacheCap - 1; t++)
        {
            for (int k = 0; k < kUse; k++) frameCodes[k] = cp[(long)k * tFrames + t];
            Tensor embed = _model.EmbedFrame(prevText, frameCodes);
            Tensor hidden = _model.Step(backend, embed, posStart: t, cache);
            embed.Dispose();
            Tensor logits = _model.ProjectText(backend, hidden);
            hidden.Dispose();

            int next = ArgMax(new Span<float>((void*)logits.DataPointer, _cfg.TextVocab));
            logits.Dispose();
            if (next != _cfg.TextPad) tokens.Add(next);
            prevText = next;
            if (progress != null && (t & 63) == 0) progress(new(t, tFrames, sw.Elapsed.TotalMilliseconds));
        }
        codes.Dispose();
        sw.Stop();
        Logs.Info($"Kyutai STT: {tFrames} frames → {tokens.Count} text tokens in {sw.ElapsedMilliseconds}ms.");
        return tokens.ToArray();
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _model.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codec.EnumerateWeights()) yield return t;
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0;
        float bv = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > bv) { bv = v[i]; best = i; }
        return best;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(KyutaiSttPipeline));
    }
}
