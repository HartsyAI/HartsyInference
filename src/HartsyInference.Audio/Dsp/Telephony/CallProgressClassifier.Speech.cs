namespace HartsyInference.Audio.Dsp.Telephony;

public sealed partial class CallProgressClassifier
{
    private const int TurnTakingWindowBlocks = 125;
    private const int TurnTakingMaxBlocks = 200;
    private const int ShortUtteranceMaxBlocks = 100;

    private bool _epOpen;
    private int _epSpeechBlocks;
    private int _epGap;
    private long _epStart;
    private long _lastSpeechBlock;
    private bool _epAfterLocal;
    private bool _localSpoke;
    private long _localEnd;
    private int _soundRun;
    private int _soundGap;
    private int _silentRun;

    private void ResetState()
    {
        Array.Fill(_lastBlock, -ReemitBlocks);
        Array.Clear(_lastConfidence);
        ResetTones();
        _epOpen = false;
        _epSpeechBlocks = 0;
        _epGap = 0;
        _epStart = 0;
        _lastSpeechBlock = 0;
        _epAfterLocal = false;
        _localSpoke = false;
        _localEnd = 0;
        _soundRun = 0;
        _soundGap = 0;
        _silentRun = 0;
    }

    private void UpdateSpeech(BlockKind kind, bool local)
    {
        if (local)
        {
            // The local side has the floor: whatever the far end says now is echo or interruption, and the greeting
            // clock starts over once it stops.
            _epOpen = false;
            _epSpeechBlocks = 0;
            _epGap = 0;
            _localSpoke = true;
            _localEnd = _blockIndex;
            return;
        }
        if (kind == BlockKind.Speech)
        {
            if (_epOpen && _epGap >= _gapBlocks)
            {
                FinishEpisode();
            }
            if (!_epOpen)
            {
                _epOpen = true;
                _epSpeechBlocks = 0;
                _epStart = _blockIndex;
                _epAfterLocal = _localSpoke && _blockIndex - _localEnd <= TurnTakingWindowBlocks;
            }
            _epGap = 0;
            _lastSpeechBlock = _blockIndex;
            _epSpeechBlocks++;
            if (_epSpeechBlocks == _longGreetingBlocks)
            {
                Emit(CallProgressKind.MachineGreeting, 0.55f, CallProgressReason.LongUninterruptedSpeech, _epStart, 0f, _epSpeechBlocks * BlockMs);
            }
            else if (_epSpeechBlocks == 2 * _longGreetingBlocks)
            {
                Emit(CallProgressKind.MachineGreeting, 0.7f, CallProgressReason.LongUninterruptedSpeech, _epStart, 0f, _epSpeechBlocks * BlockMs);
            }
        }
        else if (_epOpen && ++_epGap == _gapBlocks)
        {
            FinishEpisode();
        }
    }

    // The utterance is over once the far end has been quiet long enough for a reply to be due.
    private void FinishEpisode()
    {
        _epOpen = false;
        int blocks = _epSpeechBlocks;
        if (blocks * BlockMs < 300)
        {
            return;
        }
        if (_epAfterLocal && blocks <= TurnTakingMaxBlocks)
        {
            Emit(CallProgressKind.HumanSpeech, 0.7f, CallProgressReason.TurnTaking, _epStart, 0f, blocks * BlockMs);
        }
        else if (!_epAfterLocal && blocks <= ShortUtteranceMaxBlocks)
        {
            Emit(CallProgressKind.HumanSpeech, 0.55f, CallProgressReason.ShortUtteranceThenSilence, _epStart, 0f, blocks * BlockMs);
        }
    }

    private void UpdateSoundAndSilence(BlockKind kind, bool local)
    {
        if (local)
        {
            _soundRun = 0;
            _soundGap = 0;
            _silentRun = 0;
            return;
        }
        if (kind == BlockKind.Sound)
        {
            _soundGap = 0;
            if (++_soundRun == _holdBlocks)
            {
                Emit(CallProgressKind.HoldMusic, 0.5f, CallProgressReason.SustainedNonSpeech, _blockIndex - _soundRun + 1, 0f, _soundRun * BlockMs);
            }
        }
        else if (kind == BlockKind.Silent && _soundRun > 0 && ++_soundGap <= 8)
        {
            // Music has quiet moments; a short one does not restart the count.
        }
        else
        {
            _soundRun = 0;
            _soundGap = 0;
        }
        // The off part of a tone cadence (the 4 s between rings) is silence by design, not a quiet line.
        if (kind == BlockKind.Silent && _burstTone == 0 && _lastTone == 0)
        {
            if (++_silentRun == _silenceBlocks)
            {
                Emit(CallProgressKind.PromptSilence, 0.9f, CallProgressReason.SustainedSilence, _blockIndex - _silentRun + 1, 0f, _silentRun * BlockMs);
            }
        }
        else
        {
            _silentRun = 0;
        }
    }
}
