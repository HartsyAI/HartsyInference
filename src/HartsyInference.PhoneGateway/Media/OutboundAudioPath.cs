using HartsyInference.Audio.Streaming;
using HartsyInference.PhoneLink;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>The producer side of the outbound ring: host audio at the negotiated rate is resampled to 8 kHz in fixed
/// 20 ms frames and queued for the tick thread; prompts, already at 8 kHz, go straight in.</summary>
/// <remarks>The ring allows one producer at a time. Host audio arrives on the link reader thread and prompts on
/// whichever thread plays them, so the two are serialized by <c>_producerLock</c>; that is legitimate because neither
/// is the tick thread, which never takes a lock. <see cref="StreamingResampler"/> needs whole input frames and lags
/// one frame, so partial host frames are staged, and <see cref="EndTurn"/> pads the tail and pushes one silent frame
/// to flush the lag. Which turn a frame belongs to is decided before it reaches here (the link drops stale turns), so
/// this path only needs to reset its own state on a flush.</remarks>
public sealed class OutboundAudioPath
{
    private const int FramesPerSecond = 1000 / 20;

    private readonly ClockedAudioSource _source;
    private readonly CallRecorder? _recorder;
    private readonly object _producerLock = new();
    private StreamingResampler? _resampler;
    private short[] _staging = [];
    private float[] _inFloat = [];
    private float[] _outFloat = [];
    private readonly short[] _outPcm = new short[ClockedAudioSource.FrameSamples];
    private int _staged;
    private int _inputFrame;
    private uint _hostRate;
    private long _framesQueued;

    public OutboundAudioPath(ClockedAudioSource source, CallRecorder? recorder = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        _source = source;
        _recorder = recorder;
    }

    /// <summary>The host rate the path is configured for; zero before <see cref="Configure"/>.</summary>
    public uint HostRate => Volatile.Read(ref _hostRate);

    /// <summary>8 kHz frames handed to the ring from host audio.</summary>
    public long FramesQueued => Volatile.Read(ref _framesQueued);

    /// <summary>Samples the ring refused because it was full.</summary>
    public long DroppedSamples => _source.DroppedSamples;

    /// <summary>Prepares the resampler for host audio at <paramref name="hostRate"/> (one of the protocol's rates).</summary>
    public void Configure(uint hostRate)
    {
        if (!LinkProtocol.IsOutboundSampleRate(hostRate))
        {
            throw new ArgumentOutOfRangeException(nameof(hostRate), hostRate, "Not a PhoneLink outbound rate.");
        }
        lock (_producerLock)
        {
            _hostRate = hostRate;
            _staged = 0;
            _inputFrame = (int)hostRate / FramesPerSecond;
            if (hostRate == ClockedAudioSource.SampleRate)
            {
                _resampler = null;
                return;
            }
            _resampler = new StreamingResampler((int)hostRate, ClockedAudioSource.SampleRate, _inputFrame);
            if (_staging.Length < _inputFrame)
            {
                _staging = new short[_inputFrame];
                _inFloat = new float[_inputFrame];
                _outFloat = new float[ClockedAudioSource.FrameSamples];
            }
        }
    }

    /// <summary>Queues host audio for the current turn. Called on the link reader thread.</summary>
    public void Write(ReadOnlySpan<short> pcm)
    {
        lock (_producerLock)
        {
            if (_hostRate == 0)
            {
                throw new InvalidOperationException("OutboundAudioPath.Configure must run before host audio arrives.");
            }
            if (_resampler is null)
            {
                _recorder?.WriteOutbound(pcm);
                _source.WriteOutbound(pcm);
                Volatile.Write(ref _framesQueued, _framesQueued + pcm.Length / ClockedAudioSource.FrameSamples);
                return;
            }
            while (pcm.Length > 0)
            {
                int take = Math.Min(pcm.Length, _inputFrame - _staged);
                pcm.Slice(0, take).CopyTo(_staging.AsSpan(_staged));
                _staged += take;
                pcm = pcm.Slice(take);
                if (_staged == _inputFrame)
                {
                    ConvertStagedFrame();
                }
            }
        }
    }

    /// <summary>Flushes the staged tail of a turn: pads it with silence and pushes the resampler's lag out.</summary>
    public void EndTurn()
    {
        lock (_producerLock)
        {
            if (_resampler is null)
            {
                return;
            }
            if (_staged > 0)
            {
                _staging.AsSpan(_staged).Clear();
                _staged = _inputFrame;
                ConvertStagedFrame();
            }
            _staging.AsSpan().Clear();
            _staged = _inputFrame;
            ConvertStagedFrame();
            _resampler.Reset();
        }
    }

    /// <summary>Drops staged audio and asks the tick thread to empty the ring; returns the milliseconds discarded.</summary>
    public uint Flush()
    {
        lock (_producerLock)
        {
            int stagedMs = _hostRate == 0 ? 0 : _staged * 1000 / (int)_hostRate;
            _staged = 0;
            _resampler?.Reset();
            int queued = _source.Flush();
            return (uint)(queued * 1000 / ClockedAudioSource.SampleRate + stagedMs);
        }
    }

    /// <summary>Queues 8 kHz audio that bypasses the resampler (a prompt).</summary>
    public void WritePrompt(ReadOnlySpan<short> pcm8k)
    {
        lock (_producerLock)
        {
            _recorder?.WriteOutbound(pcm8k);
            _source.WriteOutbound(pcm8k);
        }
    }

    private void ConvertStagedFrame()
    {
        for (int i = 0; i < _inputFrame; i++)
        {
            _inFloat[i] = _staging[i] * (1f / 32768f);
        }
        _resampler!.Process(_inFloat.AsSpan(0, _inputFrame), _outFloat);
        for (int i = 0; i < _outPcm.Length; i++)
        {
            float v = _outFloat[i] * 32767f;
            _outPcm[i] = (short)Math.Clamp(v, short.MinValue, short.MaxValue);
        }
        _recorder?.WriteOutbound(_outPcm);
        _source.WriteOutbound(_outPcm);
        _staged = 0;
        Volatile.Write(ref _framesQueued, _framesQueued + 1);
    }
}
