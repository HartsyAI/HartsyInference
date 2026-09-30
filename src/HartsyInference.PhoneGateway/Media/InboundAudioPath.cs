using System.Net;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneGateway.Transport;
using HartsyInference.PhoneLink;
using SIPSorcery.Net;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>The receive side of a call: the RTP packet handler (sipsorcery's receive thread) pushes G.711 frames into
/// the jitter buffer, and a pump thread pops one every 20 ms, decodes it, resamples 8 kHz to 16 kHz and queues the
/// frame on the link's audio lane.</summary>
/// <remarks>The pump runs on the default scheduler, phased 10 ms after the tick thread's deadlines so the two never
/// wake together, and it never blocks: the link lane drops rather than waits, and a link that is down is counted
/// in <see cref="DroppedByLink"/>. The packet handler does one copy and returns; decode and resample happen here,
/// never on the IO thread and never anywhere the models run.</remarks>
public sealed class InboundAudioPath : IDisposable
{
    private const long PeriodNs = ClockedAudioSource.PeriodNs;
    private const long PhaseNs = PeriodNs / 2;
    private const int OutputSamples = LinkProtocol.InboundFrameSamples;

    private readonly RtpJitterBuffer _jitter;
    private readonly EngineLink _link;
    private readonly uint _callId;
    private readonly CallRecorder? _recorder;
    private readonly StreamingResampler _resampler = new(ClockedAudioSource.SampleRate, LinkProtocol.InboundSampleRate, ClockedAudioSource.FrameSamples);
    private readonly byte[] _encoded = new byte[RtpJitterBuffer.FrameBytes];
    private readonly short[] _pcm8k = new short[ClockedAudioSource.FrameSamples];
    private readonly float[] _in = new float[ClockedAudioSource.FrameSamples];
    private readonly float[] _out = new float[OutputSamples];
    private readonly short[] _pcm16k = new short[OutputSamples];
    private Thread? _thread;
    private volatile bool _running;
    private long _framesPumped;
    private long _framesConcealed;
    private long _framesSilence;
    private long _droppedByLink;
    private long _packetsIgnored;

    public InboundAudioPath(RtpJitterBuffer jitter, EngineLink link, uint callId, CallRecorder? recorder)
    {
        ArgumentNullException.ThrowIfNull(jitter);
        ArgumentNullException.ThrowIfNull(link);
        _jitter = jitter;
        _link = link;
        _callId = callId;
        _recorder = recorder;
    }

    /// <summary>Frames handed to the link (audio, concealed or silence).</summary>
    public long FramesPumped => Volatile.Read(ref _framesPumped);

    public long FramesConcealed => Volatile.Read(ref _framesConcealed);

    /// <summary>Frames sent as silence because no stream was established.</summary>
    public long FramesSilence => Volatile.Read(ref _framesSilence);

    /// <summary>Frames the link refused because no connection was up.</summary>
    public long DroppedByLink => Volatile.Read(ref _droppedByLink);

    /// <summary>RTP packets that were not G.711 audio (telephone events, other payload types).</summary>
    public long PacketsIgnored => Volatile.Read(ref _packetsIgnored);

    /// <summary>Handler for <c>RTPSession.OnRtpPacketReceived</c>: one copy into the jitter buffer, nothing else.</summary>
    public void HandleRtpPacket(IPEndPoint remote, SDPMediaTypesEnum mediaType, RTPPacket packet)
    {
        if (mediaType != SDPMediaTypesEnum.audio || !G711Formats.TryGetLaw(packet.Header.PayloadType, out _))
        {
            Interlocked.Increment(ref _packetsIgnored);
            return;
        }
        RTPHeader header = packet.Header;
        _jitter.Push(header.SequenceNumber, header.Timestamp, header.MarkerBit == 1, (byte)header.PayloadType, packet.Payload);
    }

    public void Start()
    {
        if (_thread is not null)
        {
            throw new InvalidOperationException("InboundAudioPath is already started.");
        }
        _running = true;
        _thread = new Thread(PumpMain) { Name = "phone-rtp-pump", IsBackground = true };
        _thread.Start();
    }

    public void Stop()
    {
        Thread? thread = _thread;
        _thread = null;
        _running = false;
        if (thread is not null && thread != Thread.CurrentThread)
        {
            thread.Join();
        }
    }

    private void PumpMain()
    {
        try
        {
            long t0 = MonotonicClock.NowNs() + PhaseNs;
            long n = 1;
            while (_running)
            {
                long deadline = t0 + n * PeriodNs;
                MonotonicClock.SleepUntil(deadline);
                if (!_running)
                {
                    break;
                }
                Pump();
                n++;
                long now = MonotonicClock.NowNs();
                if (now - deadline > 5 * PeriodNs)
                {
                    // Far behind (a stall, or the process was suspended): re-base rather than burst.
                    t0 = now + PhaseNs;
                    n = 1;
                }
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[PhoneGateway] RTP pump thread faulted; inbound audio stops for this call", ex);
        }
    }

    private void Pump()
    {
        JitterPopResult result = _jitter.Pop(_encoded, out byte payloadType);
        bool concealed = false;
        switch (result)
        {
            case JitterPopResult.Frame:
            case JitterPopResult.Concealed:
                G711.Decode(_encoded, _pcm8k, payloadType == G711Formats.PcmaPayloadType ? G711Law.ALaw : G711Law.MuLaw);
                if (result == JitterPopResult.Concealed)
                {
                    concealed = true;
                    Volatile.Write(ref _framesConcealed, _framesConcealed + 1);
                }
                break;
            default:
                Array.Clear(_pcm8k);
                Volatile.Write(ref _framesSilence, _framesSilence + 1);
                break;
        }
        for (int i = 0; i < _pcm8k.Length; i++)
        {
            _in[i] = _pcm8k[i] * (1f / 32768f);
        }
        _resampler.Process(_in, _out);
        for (int i = 0; i < _pcm16k.Length; i++)
        {
            _pcm16k[i] = (short)Math.Clamp(_out[i] * 32767f, short.MinValue, short.MaxValue);
        }
        _recorder?.WriteInbound(_pcm16k);
        if (!_link.TryEnqueueInboundAudio(_callId, _pcm16k, concealed))
        {
            Volatile.Write(ref _droppedByLink, _droppedByLink + 1);
        }
        Volatile.Write(ref _framesPumped, _framesPumped + 1);
    }

    public void Dispose() => Stop();
}
