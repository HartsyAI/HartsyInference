namespace HartsyInference.Audio.Dsp.Telephony;

public sealed partial class CallProgressClassifier
{
    private const int ToneNone = 0;
    private const int ToneRing = 1;      // 440+480 Hz
    private const int ToneBusyPair = 2;  // 480+620 Hz
    private const int Tone425 = 3;       // 425 Hz single (European)
    private const int ToneDialPair = 4;  // 350+440 Hz
    private const int ToneBeep = 5;      // single tone 700-2200 Hz
    private const int MaxOffBlocks = 150;

    private const int CadenceNone = 0;
    private const int CadenceSlow = 1;
    private const int CadenceBusy = 2;
    private const int CadenceFast = 3;

    // Cadence of the current tone family.
    private int _burstTone;
    private int _onBlocks;
    private int _gapInBurst;
    private int _lastTone;
    private int _lastOnBlocks;
    private int _offBlocks;
    private long _burstStart;
    private long _sequenceStart;
    private int _cycleClass;
    private int _cycleRun;

    // Beep runs and the SIT triple.
    private int _beepRun;
    private int _beepGap;
    private float _beepHz;
    private long _beepStart;
    private int _sitStage;
    private long _sitLastEnd;
    private long _sitStart;

    private void ResetTones()
    {
        _burstTone = 0;
        _onBlocks = 0;
        _gapInBurst = 0;
        _lastTone = 0;
        _lastOnBlocks = 0;
        _offBlocks = 0;
        _burstStart = 0;
        _sequenceStart = 0;
        _cycleClass = CadenceNone;
        _cycleRun = 0;
        _beepRun = 0;
        _beepGap = 0;
        _beepHz = 0f;
        _beepStart = 0;
        _sitStage = 0;
        _sitLastEnd = 0;
        _sitStart = 0;
    }

    private void UpdateTones(int tone)
    {
        if (_burstTone != 0)
        {
            if (tone == _burstTone)
            {
                _onBlocks += 1 + _gapInBurst;
                _gapInBurst = 0;
                OnSteadyTone();
                return;
            }
            // One stray block inside a burst (a noisy block, a transition) does not end it.
            if (++_gapInBurst <= 1)
            {
                return;
            }
            EndBurst();
            _offBlocks = _gapInBurst;
            _gapInBurst = 0;
            _lastTone = _burstTone;
            _burstTone = 0;
        }
        if (tone != 0)
        {
            StartBurst(tone);
        }
        else if (_lastTone != 0 && ++_offBlocks > MaxOffBlocks)
        {
            _lastTone = 0;
            _cycleClass = CadenceNone;
            _cycleRun = 0;
        }
    }

    private void StartBurst(int tone)
    {
        if (_lastTone == tone && _offBlocks > 0)
        {
            int cadence = ClassifyCycle(_lastOnBlocks * BlockMs, _offBlocks * BlockMs);
            if (cadence != CadenceNone && cadence == _cycleClass)
            {
                _cycleRun++;
            }
            else
            {
                _cycleClass = cadence;
                _cycleRun = cadence == CadenceNone ? 0 : 1;
            }
            if (cadence != CadenceNone)
            {
                OnCycle(tone, cadence, _cycleRun);
            }
        }
        else
        {
            _sequenceStart = _blockIndex;
            _cycleClass = CadenceNone;
            _cycleRun = 0;
        }
        _burstTone = tone;
        _burstStart = _blockIndex;
        _onBlocks = 1;
        _gapInBurst = 0;
        _lastTone = 0;
        _offBlocks = 0;
    }

    private void EndBurst()
    {
        _lastOnBlocks = _onBlocks;
        int onMs = _onBlocks * BlockMs;
        // One European ring burst, heard once: 1 s on is a ring, 0.5 s on is a busy cycle that needs its pair.
        if (_burstTone == Tone425 && onMs is >= 800 and <= 1300)
        {
            Emit(CallProgressKind.RingbackTone, 0.6f, CallProgressReason.ToneCadence, _burstStart, 0f, onMs);
        }
    }

    private static int ClassifyCycle(int onMs, int offMs)
    {
        if (onMs is >= 300 and <= 700 && offMs is >= 300 and <= 700)
        {
            return CadenceBusy;
        }
        if (onMs is >= 120 and <= 380 && offMs is >= 120 and <= 380)
        {
            return CadenceFast;
        }
        if (onMs is >= 800 and <= 2600 && offMs >= 1500)
        {
            return CadenceSlow;
        }
        return CadenceNone;
    }

    private void OnCycle(int tone, int cadence, int run)
    {
        float confidence = run switch { 1 => 0.75f, 2 => 0.9f, _ => 0.95f };
        int span = (int)((_blockIndex - _sequenceStart) * BlockMs);
        switch (tone, cadence)
        {
            case (ToneRing, CadenceSlow):
            case (Tone425, CadenceSlow):
                Emit(CallProgressKind.RingbackTone, run == 1 ? 0.85f : 0.95f, CallProgressReason.ToneCadence, _sequenceStart, 0f, span);
                break;
            case (ToneBusyPair, CadenceBusy):
            case (Tone425, CadenceBusy):
                Emit(CallProgressKind.BusyTone, confidence, CallProgressReason.ToneCadence, _sequenceStart, 0f, span);
                break;
            case (ToneBusyPair, CadenceFast):
            case (Tone425, CadenceFast):
                Emit(CallProgressKind.FastBusyTone, confidence, CallProgressReason.ToneCadence, _sequenceStart, 0f, span);
                break;
        }
    }

    private void OnSteadyTone()
    {
        int onMs = _onBlocks * BlockMs;
        switch (_burstTone)
        {
            case ToneRing when onMs >= 880:
                Emit(CallProgressKind.RingbackTone, 0.6f, CallProgressReason.ToneCadence, _burstStart, 0f, onMs);
                break;
            case ToneDialPair when onMs >= 1000:
                Emit(CallProgressKind.DialTone, 0.85f, CallProgressReason.SteadyTone, _burstStart, 0f, onMs);
                break;
            case Tone425 when onMs >= 3000:
                Emit(CallProgressKind.DialTone, 0.7f, CallProgressReason.SteadyTone, _burstStart, 0f, onMs);
                break;
        }
    }

    private void UpdateBeep(bool isBeep, float hz)
    {
        if (isBeep)
        {
            if (_beepRun > 0 && MathF.Abs(hz - _beepHz) <= 50f)
            {
                _beepRun += 1 + _beepGap;
            }
            else
            {
                if (_beepRun > 0)
                {
                    EndBeep();
                }
                _beepRun = 1;
                _beepHz = hz;
                _beepStart = _blockIndex;
            }
            _beepGap = 0;
        }
        else if (_beepRun > 0 && ++_beepGap > 1)
        {
            EndBeep();
        }
    }

    private void EndBeep()
    {
        int blocks = _beepRun;
        float hz = _beepHz;
        long start = _beepStart;
        _beepRun = 0;
        _beepGap = 0;
        int durationMs = blocks * BlockMs;
        if (blocks is < 3 or > 50)
        {
            _sitStage = 0;
            return;
        }
        float confidence = durationMs is >= 200 and <= 1000 ? 0.85f : 0.6f;
        Emit(CallProgressKind.Beep, confidence, CallProgressReason.SteadyTone, start, hz, durationMs);
        CheckSit(hz, blocks, start);
        if (_epSpeechBlocks * BlockMs >= 3000 && start - _lastSpeechBlock <= 100)
        {
            Emit(CallProgressKind.MachineGreeting, 0.9f, CallProgressReason.BeepAfterGreeting, _epStart, hz, _epSpeechBlocks * BlockMs);
        }
    }

    // SIT: three tones in ascending bands, each 270-400 ms, with no gap (the last block of one is mixed with the next).
    private void CheckSit(float hz, int blocks, long start)
    {
        bool segment = blocks is >= 4 and <= 14;
        bool adjacent = start - _sitLastEnd <= 3;
        if (segment && hz is >= 850f and <= 1050f)
        {
            _sitStage = 1;
            _sitStart = start;
        }
        else if (segment && hz is >= 1300f and <= 1500f && _sitStage == 1 && adjacent)
        {
            _sitStage = 2;
        }
        else if (segment && hz is >= 1700f and <= 1850f && _sitStage == 2 && adjacent)
        {
            Emit(CallProgressKind.SitTone, 0.85f, CallProgressReason.ToneSequence, _sitStart, hz, blocks * BlockMs);
            _sitStage = 0;
        }
        else
        {
            _sitStage = 0;
        }
        _sitLastEnd = start + blocks;
    }
}
