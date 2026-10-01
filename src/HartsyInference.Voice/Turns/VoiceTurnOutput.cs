using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Runtime;

namespace HartsyInference.Voice.Turns;

/// <summary>One turn's path from synthesized chunks to the outbound queue: fixed 20 ms frames through a
/// <see cref="StreamingResampler"/> when the synthesizer's rate differs from the outbound rate, then
/// <see cref="VoiceOutbound.WriteAsync"/> under the turn's flush epoch and turn mark.</summary>
/// <remarks>The resampler carries its filter context across sentences, so a reply is one continuous signal; it is
/// built per turn, so one turn's tail never bleeds into the next. Its output trails its input by one frame, so
/// <see cref="CompleteAsync"/> pads the last partial frame and pushes one frame of silence to emit the reply's final
/// samples.</remarks>
internal sealed class VoiceTurnOutput
{
    private const int FramesPerSecond = 50;

    private readonly VoiceOutbound _outbound;
    private readonly int _turnId;
    private readonly StreamingResampler? _resampler;
    private readonly float[] _input;
    private readonly float[] _output;
    private int _inputFill;
    private long _epoch;
    private bool _started;
    private bool _superseded;

    public VoiceTurnOutput(VoiceOutbound outbound, int turnId, int sourceRate, int outputRate)
    {
        ArgumentNullException.ThrowIfNull(outbound);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(turnId);
        _outbound = outbound;
        _turnId = turnId;
        _resampler = CreateResampler(sourceRate, outputRate);
        _input = _resampler is null ? [] : new float[_resampler.InputFrameSize];
        _output = _resampler is null ? [] : new float[_resampler.OutputFrameSize];
    }

    /// <summary>When the first samples were queued, in monotonic nanoseconds; 0 before.</summary>
    public long FirstWriteNs { get; private set; }

    /// <summary>Whether a flush ended this turn's output.</summary>
    public bool Superseded => _superseded;

    /// <summary>Throws when <paramref name="sourceRate"/> cannot be converted to <paramref name="outputRate"/> in whole
    /// 20 ms frames.</summary>
    public static void Validate(int sourceRate, int outputRate) => CreateResampler(sourceRate, outputRate);

    /// <summary>Queues one chunk of synthesized audio; false once a flush superseded the turn.</summary>
    public async ValueTask<bool> WriteAsync(float[] samples, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (!await StartAsync(cancel).ConfigureAwait(false))
        {
            return false;
        }
        if (_resampler is null)
        {
            return await PutAsync(samples, samples.Length, cancel).ConfigureAwait(false);
        }
        int offset = 0;
        while (offset < samples.Length)
        {
            int take = Math.Min(_input.Length - _inputFill, samples.Length - offset);
            Array.Copy(samples, offset, _input, _inputFill, take);
            _inputFill += take;
            offset += take;
            if (_inputFill < _input.Length)
            {
                break;
            }
            _inputFill = 0;
            _resampler.Process(_input, _output);
            if (!await PutAsync(_output, _output.Length, cancel).ConfigureAwait(false))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>Emits what the resampler still holds and returns the outbound position just past this turn's audio.</summary>
    public async ValueTask<long> CompleteAsync(CancellationToken cancel)
    {
        if (_resampler is not null && _started && !_superseded)
        {
            // Two frames: the zero-padded partial frame, then one of silence to push the one-frame lag out.
            for (int frame = 0; frame < 2; frame++)
            {
                Array.Clear(_input, _inputFill, _input.Length - _inputFill);
                _inputFill = 0;
                _resampler.Process(_input, _output);
                if (!await PutAsync(_output, _output.Length, cancel).ConfigureAwait(false))
                {
                    break;
                }
            }
        }
        return _outbound.Written;
    }

    private async ValueTask<bool> StartAsync(CancellationToken cancel)
    {
        if (!_started)
        {
            _epoch = await _outbound.BeginTurnAsync(_turnId, cancel).ConfigureAwait(false);
            _started = true;
        }
        return !_superseded;
    }

    private async ValueTask<bool> PutAsync(float[] samples, int count, CancellationToken cancel)
    {
        if (!await _outbound.WriteAsync(samples, 0, count, _epoch, cancel).ConfigureAwait(false))
        {
            _superseded = true;
            return false;
        }
        if (FirstWriteNs == 0)
        {
            FirstWriteNs = MonotonicClock.NowNs();
        }
        return true;
    }

    private static StreamingResampler? CreateResampler(int sourceRate, int outputRate)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(sourceRate, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(outputRate, 0);
        if (sourceRate == outputRate)
        {
            return null;
        }
        if (sourceRate % FramesPerSecond != 0)
        {
            throw new ArgumentException($"A {sourceRate} Hz synthesizer does not divide into 20 ms frames.", nameof(sourceRate));
        }
        return new StreamingResampler(sourceRate, outputRate, sourceRate / FramesPerSecond);
    }
}
