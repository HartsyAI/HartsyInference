using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Everything IndexTTS-2 derives from the speaker's reference clip alone — the w2v-bert feature, the S2Mel
/// reference mel and CAM++ style, the length-regulated prompt condition, the GPT's speaker conditioning and the base
/// emotion vector. Built once by <see cref="IndexTts2Pipeline.PrepareReference"/> and reused for any number of
/// syntheses, mirroring the reference implementation's <c>cache_spk_cond</c>/<c>cache_s2mel_style</c>/
/// <c>cache_s2mel_prompt</c>/<c>cache_mel</c> (the 24-layer w2v-bert pass over the clip is the single most expensive
/// front-end step, and it depends on nothing but the clip). Tied to the pipeline that created it.</summary>
public sealed class IndexTts2Reference : IDisposable
{
    internal IndexTts2Pipeline Owner { get; }
    internal Tensor SpkCondEmb { get; }
    internal int TSpk { get; }
    internal Tensor RefMel { get; }
    internal int TRef { get; }
    internal Tensor Style { get; }
    internal Tensor PromptCondition { get; }
    internal Tensor SpeakerConditioning { get; }
    internal Tensor BaseEmoVec { get; }
    internal float[] Audio16k { get; }
    private int _disposed;

    internal IndexTts2Reference(IndexTts2Pipeline owner, Tensor spkCondEmb, int tSpk, Tensor refMel, int tRef, Tensor style,
        Tensor promptCondition, Tensor speakerConditioning, Tensor baseEmoVec, float[] audio16k)
    {
        Owner = owner;
        SpkCondEmb = spkCondEmb;
        TSpk = tSpk;
        RefMel = refMel;
        TRef = tRef;
        Style = style;
        PromptCondition = promptCondition;
        SpeakerConditioning = speakerConditioning;
        BaseEmoVec = baseEmoVec;
        Audio16k = audio16k;
    }

    /// <summary>Seconds of reference audio the clip contributed (after the 15 s cap).</summary>
    public double Seconds => Audio16k.Length / 16_000.0;

    internal void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(IndexTts2Reference));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        SpkCondEmb.Dispose();
        RefMel.Dispose();
        Style.Dispose();
        PromptCondition.Dispose();
        SpeakerConditioning.Dispose();
        BaseEmoVec.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>Optional per-stage wall-clock sink for <see cref="IndexTts2Options.Timings"/>; every value accumulates in
/// milliseconds across the segments of one synthesis.</summary>
public sealed class IndexTts2Timings
{
    /// <summary>Reference-clip front end (w2v-bert, mel, CAM++, conditioning); zero when a prepared
    /// <see cref="IndexTts2Reference"/> was supplied.</summary>
    public double ReferenceMs { get; set; }
    public double EmotionMs { get; set; }
    /// <summary>The GPT's autoregressive code generation.</summary>
    public double GptMs { get; set; }
    /// <summary>2.0's second GPT pass + <c>gpt_layer</c> + codebook embedding (2.5: the codec decoder).</summary>
    public double SemanticMs { get; set; }
    /// <summary>Length regulator + the S2Mel flow-matching solve.</summary>
    public double FlowMs { get; set; }
    public double VocoderMs { get; set; }
    public int CodeCount { get; set; }
    public int SegmentCount { get; set; }
    /// <summary>Milliseconds from the call until the first segment's audio was ready (time-to-first-audio).</summary>
    public double FirstAudioMs { get; set; }
    public double AudioSeconds { get; set; }

    public double TotalMs => ReferenceMs + EmotionMs + GptMs + SemanticMs + FlowMs + VocoderMs;

    /// <summary>Real-time factor over the generation stages (below 1 is faster than real time).</summary>
    public double Rtf => AudioSeconds > 0 ? TotalMs / 1000.0 / AudioSeconds : 0;

    public override string ToString() =>
        $"ref {ReferenceMs:F0} ms, emo {EmotionMs:F0} ms, gpt {GptMs:F0} ms ({CodeCount} codes), semantic {SemanticMs:F0} ms, " +
        $"flow {FlowMs:F0} ms, vocoder {VocoderMs:F0} ms | first audio {FirstAudioMs:F0} ms | {AudioSeconds:F2} s audio, RTF {Rtf:F2}";
}
