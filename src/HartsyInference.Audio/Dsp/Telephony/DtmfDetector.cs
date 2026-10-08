namespace HartsyInference.Audio.Dsp.Telephony;

/// <summary>Detects in-band DTMF tones in 16 kHz mono audio: Goertzel over the eight DTMF frequencies and the
/// second harmonics of both groups, on 20 ms blocks every 10 ms, with the amplitude, in-group dominance, twist,
/// tonal-fraction and second-harmonic checks that keep speech from reading as keys.</summary>
/// <remarks>Reports one <see cref="DtmfEvent"/> per tone, when it has lasted <see cref="DtmfDetectorOptions.MinToneMs"/>;
/// the tone then stays "held" until it is absent for <see cref="DtmfDetectorOptions.MinPauseMs"/>, so a dropout shorter
/// than that does not split a key into two, and a key pressed again after a real pause is a new event. Feed it the raw
/// audio, before any denoiser: a noise suppressor attenuates tones. Allocation-free after construction; not
/// thread-safe. Events are queued (up to <see cref="QueueCapacity"/>) and read with <see cref="TryDequeue"/>.</remarks>
public sealed class DtmfDetector
{
    /// <summary>Sample rate the detector runs at.</summary>
    public const int SampleRate = 16_000;

    /// <summary>Samples per analysis block (20 ms).</summary>
    public const int BlockSamples = 320;

    /// <summary>Samples between blocks (10 ms).</summary>
    public const int HopSamples = 160;

    /// <summary>Events queued before the oldest unread one is dropped.</summary>
    public const int QueueCapacity = 16;

    private static readonly double[] LowHz = [697, 770, 852, 941];
    private static readonly double[] HighHz = [1209, 1336, 1477, 1633];
    private static readonly char[] Keys = ['1', '2', '3', 'A', '4', '5', '6', 'B', '7', '8', '9', 'C', '*', '0', '#', 'D'];

    private readonly GoertzelBank _bank;
    private readonly float[] _block = new float[BlockSamples];
    private readonly float[] _power = new float[16];
    private readonly DtmfEvent[] _queue = new DtmfEvent[QueueCapacity];
    private readonly float _minAmplitudeEnergy;
    private readonly float _minTonalFraction;
    private readonly float _dominance;
    private readonly float _twistForward;
    private readonly float _twistReverse;
    private readonly float _harmonicLow;
    private readonly float _harmonicHigh;
    private readonly int _minBlocks;
    private readonly int _releaseBlocks;
    private int _fill;
    private long _blockStart;
    private long _consumed;
    private char _pending;
    private int _pendingRun;
    private long _pendingStart;
    private char _held;
    private int _absent;
    private int _queueHead;
    private int _queueCount;
    private long _dropped;
    private long _enqueued;

    /// <summary>A detector with the default thresholds.</summary>
    public DtmfDetector() : this(new DtmfDetectorOptions())
    {
    }

    /// <summary>A detector with <paramref name="options"/>.</summary>
    public DtmfDetector(DtmfDetectorOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        double[] all = new double[16];
        for (int i = 0; i < 4; i++)
        {
            all[i] = LowHz[i];
            all[4 + i] = HighHz[i];
            all[8 + i] = 2 * LowHz[i];
            all[12 + i] = 2 * HighHz[i];
        }
        _bank = new GoertzelBank(SampleRate, all);
        // A component of amplitude A over N samples carries A²·N/2 of energy.
        _minAmplitudeEnergy = options.MinToneAmplitude * options.MinToneAmplitude * BlockSamples / 2f;
        _minTonalFraction = options.MinTonalFraction;
        _dominance = MathF.Pow(10f, options.GroupDominanceDb / 10f);
        _twistForward = MathF.Pow(10f, options.MaxForwardTwistDb / 10f);
        _twistReverse = MathF.Pow(10f, options.MaxReverseTwistDb / 10f);
        _harmonicLow = options.MaxSecondHarmonicRatioLow;
        _harmonicHigh = options.MaxSecondHarmonicRatioHigh;
        // The first of the blocks that carry a tone starts one block-length before the confirmation, so min-duration
        // N ms needs enough blocks that the hop-spaced run spans it.
        _minBlocks = Math.Max(2, (int)Math.Ceiling((options.MinToneMs * (SampleRate / 1000.0) - BlockSamples) / HopSamples) + 1);
        _releaseBlocks = Math.Max(1, (int)Math.Ceiling(options.MinPauseMs * (SampleRate / 1000.0) / HopSamples));
    }

    /// <summary>Samples pushed since construction or the last <see cref="Reset"/>.</summary>
    public long ConsumedSamples => _consumed;

    /// <summary>The key whose tone is being held, or <c>'\0'</c> when none.</summary>
    public char HeldDigit => _held;

    /// <summary>Events dropped because the queue was full.</summary>
    public long DroppedEvents => _dropped;

    /// <summary>Events waiting in the queue.</summary>
    public int QueuedEvents => _queueCount;

    /// <summary>Feeds samples (16 kHz mono, ±1; any length). Returns how many events this call queued.</summary>
    public int Process(ReadOnlySpan<float> samples)
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
                AnalyzeBlock();
                // Slide by one hop: the newest overlap becomes the oldest part of the next block.
                _block.AsSpan(HopSamples).CopyTo(_block);
                _fill = BlockSamples - HopSamples;
                _blockStart += HopSamples;
            }
        }
        return (int)(_enqueued - enqueuedBefore);
    }

    /// <summary>Takes the oldest queued event.</summary>
    public bool TryDequeue(out DtmfEvent item)
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
        _blockStart = 0;
        _consumed = 0;
        _pending = '\0';
        _pendingRun = 0;
        _held = '\0';
        _absent = 0;
        _queueHead = 0;
        _queueCount = 0;
    }

    private void AnalyzeBlock()
    {
        float energy = 0f;
        foreach (float x in _block)
        {
            energy += x * x;
        }
        char key = '\0';
        if (energy > 0f)
        {
            _bank.Compute(_block, _power);
            key = Decide(energy);
        }
        Advance(key);
    }

    private char Decide(float energy)
    {
        float scale = 2f / BlockSamples;
        int low = Peak(0, out float lowE, out float lowSecond);
        int high = Peak(4, out float highE, out float highSecond);
        lowE *= scale;
        highE *= scale;
        if (lowE < _minAmplitudeEnergy || highE < _minAmplitudeEnergy)
        {
            return '\0';
        }
        if ((lowE + highE) < _minTonalFraction * energy)
        {
            return '\0';
        }
        if (lowE < _dominance * lowSecond * scale || highE < _dominance * highSecond * scale)
        {
            return '\0';
        }
        float twist = highE / lowE;
        if (twist > _twistReverse || twist < 1f / _twistForward)
        {
            return '\0';
        }
        if (_power[8 + low] * scale > _harmonicLow * lowE || _power[12 + high] * scale > _harmonicHigh * highE)
        {
            return '\0';
        }
        return Keys[low * 4 + high];
    }

    private int Peak(int start, out float best, out float second)
    {
        int index = 0;
        best = _power[start];
        second = 0f;
        for (int i = 1; i < 4; i++)
        {
            float p = _power[start + i];
            if (p > best)
            {
                second = best;
                best = p;
                index = i;
            }
            else if (p > second)
            {
                second = p;
            }
        }
        return index;
    }

    private void Advance(char key)
    {
        if (_held != '\0')
        {
            if (key == _held)
            {
                _absent = 0;
                _pending = '\0';
                return;
            }
            if (key == '\0')
            {
                if (++_absent >= _releaseBlocks)
                {
                    _held = '\0';
                    _absent = 0;
                }
                return;
            }
            // A different key with no pause: it has to prove itself as a new tone below.
            _absent = 0;
        }
        if (key == '\0')
        {
            _pending = '\0';
            _pendingRun = 0;
            return;
        }
        if (key != _pending)
        {
            _pending = key;
            _pendingRun = 1;
            _pendingStart = _blockStart;
            return;
        }
        if (++_pendingRun >= _minBlocks)
        {
            Enqueue(new DtmfEvent(key, _pendingStart, _blockStart + BlockSamples));
            _held = key;
            _absent = 0;
            _pending = '\0';
            _pendingRun = 0;
        }
    }

    private void Enqueue(DtmfEvent item)
    {
        if (_queueCount == QueueCapacity)
        {
            _queueHead = (_queueHead + 1) % QueueCapacity;
            _queueCount--;
            _dropped++;
        }
        _queue[(_queueHead + _queueCount) % QueueCapacity] = item;
        _enqueued++;
        _queueCount++;
    }
}
