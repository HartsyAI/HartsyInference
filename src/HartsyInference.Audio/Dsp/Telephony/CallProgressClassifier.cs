namespace HartsyInference.Audio.Dsp.Telephony;

/// <summary>Classifies what the far end of a phone call is sending (ringing, busy, a beep, a recording, a person, hold
/// music, silence) from 16 kHz mono audio, as a stream of <see cref="CallProgressEvent"/> signals with confidences.</summary>
/// <remarks>
/// <para><b>What is reliable.</b> Tone cadences (440+480 Hz ringback, 480+620 Hz busy and fast busy, the European
/// 425 Hz tones, dial tone, the SIT triple) and single beeps are found with Goertzel filters on 40 ms blocks and a
/// burst on/off state machine; those are deterministic DSP and are the dependable part.</para>
/// <para><b>What is heuristic.</b> Whether speech came from a machine or a person, and whether non-speech sound is hold
/// music, cannot be decided from audio alone. The classifier turns the host's per-frame speech probability (a VAD such
/// as Silero) and whether the local side is speaking into timing evidence: a beep after a long greeting (strong), long
/// uninterrupted far-end speech with no turn-taking (weak), a short utterance followed by waiting or speech that answers
/// the local side (weak, human). They are events with a confidence, never a verdict; the host or the language model
/// decides.</para>
/// <para>Feed the raw call audio, before any denoiser. Allocation-free after construction; not thread-safe. Events are
/// queued (up to <see cref="QueueCapacity"/>) and read with <see cref="TryDequeue"/>.</para></remarks>
public sealed partial class CallProgressClassifier
{
    /// <summary>Sample rate the classifier runs at.</summary>
    public const int SampleRate = 16_000;

    /// <summary>Samples per analysis block (40 ms); every duration the classifier measures is a multiple of it.</summary>
    public const int BlockSamples = 640;

    /// <summary>Events queued before the oldest unread one is dropped.</summary>
    public const int QueueCapacity = 16;

    private const int BlockMs = BlockSamples * 1000 / SampleRate;
    private const int GridStartHz = 700;
    private const int GridStepHz = 25;
    private const int GridCount = 61;
    private const int FixedBins = 5;
    private const int Bin350 = 0;
    private const int Bin425 = 1;
    private const int Bin440 = 2;
    private const int Bin480 = 3;
    private const int Bin620 = 4;
    private const int ReemitBlocks = 250;
    private const int KindCount = 10;

    private readonly GoertzelBank _bank;
    private readonly float[] _hann = new float[BlockSamples];
    private readonly float[] _block = new float[BlockSamples];
    private readonly float[] _windowed = new float[BlockSamples];
    private readonly float[] _power = new float[FixedBins + GridCount];
    private readonly CallProgressEvent[] _queue = new CallProgressEvent[QueueCapacity];
    private readonly long[] _lastBlock = new long[KindCount];
    private readonly float[] _lastConfidence = new float[KindCount];
    private readonly float _silenceRms;
    private readonly float _speechProbability;
    private readonly int _longGreetingBlocks;
    private readonly int _gapBlocks;
    private readonly int _holdBlocks;
    private readonly int _silenceBlocks;
    private int _fill;
    private long _blockIndex;
    private long _consumed;
    private int _queueHead;
    private int _queueCount;
    private long _enqueued;
    private long _dropped;

    /// <summary>A classifier with the default thresholds.</summary>
    public CallProgressClassifier() : this(new CallProgressOptions())
    {
    }

    /// <summary>A classifier with <paramref name="options"/>.</summary>
    public CallProgressClassifier(CallProgressOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        double[] frequencies = new double[FixedBins + GridCount];
        frequencies[Bin350] = 350;
        frequencies[Bin425] = 425;
        frequencies[Bin440] = 440;
        frequencies[Bin480] = 480;
        frequencies[Bin620] = 620;
        for (int i = 0; i < GridCount; i++)
        {
            frequencies[FixedBins + i] = GridStartHz + i * GridStepHz;
        }
        _bank = new GoertzelBank(SampleRate, frequencies);
        for (int i = 0; i < BlockSamples; i++)
        {
            _hann[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / BlockSamples);
        }
        _silenceRms = MathF.Pow(10f, options.SilenceDbfs / 20f);
        _speechProbability = options.SpeechProbability;
        _longGreetingBlocks = options.LongGreetingMs / BlockMs;
        _gapBlocks = Math.Max(1, options.UtteranceGapMs / BlockMs);
        _holdBlocks = options.HoldMusicMs / BlockMs;
        _silenceBlocks = options.PromptSilenceMs / BlockMs;
        ResetState();
    }

    /// <summary>Samples pushed since construction or the last <see cref="Reset"/>.</summary>
    public long ConsumedSamples => _consumed;

    /// <summary>Events dropped because the queue was full.</summary>
    public long DroppedEvents => _dropped;

    /// <summary>Events waiting in the queue.</summary>
    public int QueuedEvents => _queueCount;

    /// <summary>Feeds samples (16 kHz mono, ±1; any length). <paramref name="speechProbability"/> is the host VAD's
    /// speech probability for them (0-1), <paramref name="localSpeaking"/> whether the local side is talking, which
    /// pauses the far-end speech heuristics (what the far end says over our own voice is echo or interruption, not a
    /// greeting). Returns how many events this call queued.</summary>
    public int Process(ReadOnlySpan<float> samples, float speechProbability, bool localSpeaking)
    {
        long enqueuedBefore = _enqueued;
        while (!samples.IsEmpty)
        {
            int take = Math.Min(BlockSamples - _fill, samples.Length);
            samples[..take].CopyTo(_block.AsSpan(_fill));
            _fill += take;
            _consumed += take;
            samples = samples[take..];
            if (_fill == BlockSamples)
            {
                _fill = 0;
                AnalyzeBlock(speechProbability, localSpeaking);
                _blockIndex++;
            }
        }
        return (int)(_enqueued - enqueuedBefore);
    }

    /// <summary>Takes the oldest queued event.</summary>
    public bool TryDequeue(out CallProgressEvent item)
    {
        if (_queueCount == 0)
        {
            item = default;
            return false;
        }
        item = _queue[_queueHead];
        _queueHead = (_queueHead + 1) % QueueCapacity;
        _queueCount--;
        return true;
    }

    /// <summary>Clears all state and the sample counter.</summary>
    public void Reset()
    {
        _fill = 0;
        _blockIndex = 0;
        _consumed = 0;
        _queueHead = 0;
        _queueCount = 0;
        ResetState();
    }

    private enum BlockKind
    {
        Silent,
        Speech,
        Sound,
        Tone,
    }

    private void AnalyzeBlock(float speechProbability, bool local)
    {
        float energy = 0f;
        foreach (float x in _block)
        {
            energy += x * x;
        }
        float rms = MathF.Sqrt(energy / BlockSamples);
        BlockKind kind;
        int tone = ToneNone;
        float beepHz = 0f;
        if (rms < _silenceRms)
        {
            kind = BlockKind.Silent;
        }
        else
        {
            float windowedEnergy = 0f;
            for (int i = 0; i < BlockSamples; i++)
            {
                float w = _block[i] * _hann[i];
                _windowed[i] = w;
                windowedEnergy += w * w;
            }
            _bank.Compute(_windowed, _power);
            // 3·P/(N·E) is 1 for a pure tone on a bin's frequency under the Hann window.
            float norm = 3f / (BlockSamples * windowedEnergy);
            tone = FindTone(norm, out beepHz);
            kind = tone != ToneNone ? BlockKind.Tone : speechProbability >= _speechProbability ? BlockKind.Speech : BlockKind.Sound;
        }
        UpdateTones(tone == ToneBeep ? ToneNone : tone);
        UpdateBeep(tone == ToneBeep, beepHz);
        UpdateSpeech(kind, local);
        UpdateSoundAndSilence(kind, local);
    }

    private int FindTone(float norm, out float beepHz)
    {
        float f350 = _power[Bin350] * norm;
        float f425 = _power[Bin425] * norm;
        float f440 = _power[Bin440] * norm;
        float f480 = _power[Bin480] * norm;
        float f620 = _power[Bin620] * norm;
        beepHz = 0f;
        if (f440 >= 0.2f && f480 >= 0.2f && f440 + f480 >= 0.7f && f620 < 0.1f && f350 < 0.1f)
        {
            return ToneRing;
        }
        if (f480 >= 0.2f && f620 >= 0.2f && f480 + f620 >= 0.7f && f440 < 0.1f)
        {
            return ToneBusyPair;
        }
        if (f350 >= 0.2f && f440 >= 0.2f && f350 + f440 >= 0.7f && f480 < 0.1f)
        {
            return ToneDialPair;
        }
        if (f425 >= 0.7f && f350 < 0.1f && f480 < 0.1f && f620 < 0.1f)
        {
            return Tone425;
        }
        int best = 0;
        float peak = 0f;
        for (int i = 0; i < GridCount; i++)
        {
            float p = _power[FixedBins + i];
            if (p > peak)
            {
                peak = p;
                best = i;
            }
        }
        if (peak * norm >= 0.55f)
        {
            beepHz = GridStartHz + best * GridStepHz;
            return ToneBeep;
        }
        return ToneNone;
    }

    private void Emit(CallProgressKind kind, float confidence, CallProgressReason reason, long startBlock, float frequencyHz, int durationMs)
    {
        int k = (int)kind;
        if (_blockIndex - _lastBlock[k] < ReemitBlocks && confidence < _lastConfidence[k] + 0.1f)
        {
            return;
        }
        _lastBlock[k] = _blockIndex;
        _lastConfidence[k] = confidence;
        CallProgressEvent item = new(kind, confidence, reason, startBlock * BlockSamples, (_blockIndex + 1) * BlockSamples, frequencyHz, durationMs);
        if (_queueCount == QueueCapacity)
        {
            _queueHead = (_queueHead + 1) % QueueCapacity;
            _queueCount--;
            _dropped++;
        }
        _queue[(_queueHead + _queueCount) % QueueCapacity] = item;
        _queueCount++;
        _enqueued++;
    }
}
