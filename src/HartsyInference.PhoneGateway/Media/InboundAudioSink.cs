using SIPSorceryMedia.Abstractions;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>Format-negotiation shell on the receive side. sipsorcery needs an <see cref="IAudioSink"/> to offer receive
/// formats and to be told the negotiated one; the audio itself is not taken from <see cref="GotEncodedMediaFrame"/>
/// (that path carries no sequence number or timestamp) but from <c>RTPSession.OnRtpPacketReceived</c>, which
/// <see cref="InboundAudioPath"/> hooks into the <see cref="RtpJitterBuffer"/>.</summary>
public sealed class InboundAudioSink(AudioCodecPreference codec) : IAudioSink
{
    private Func<AudioFormat, bool>? _formatFilter;
    private AudioFormat _format = AudioFormat.Empty;

    /// <summary>Never raised: this sink does nothing that can fail.</summary>
    public event SourceErrorDelegate? OnAudioSinkError
    {
        add { }
        remove { }
    }

    /// <summary>The negotiated receive format, or <see cref="AudioFormat.Empty"/> before negotiation.</summary>
    public AudioFormat Format => _format;

    public List<AudioFormat> GetAudioSinkFormats()
    {
        List<AudioFormat> formats = G711Formats.Offer(codec);
        Func<AudioFormat, bool>? filter = _formatFilter;
        return filter is null ? formats : formats.Where(filter).ToList();
    }

    public void RestrictFormats(Func<AudioFormat, bool> filter) => _formatFilter = filter;

    public void SetAudioSinkFormat(AudioFormat audioFormat) => _format = audioFormat;

    /// <summary>Ignored: frames are taken from the RTP packet event with their sequence numbers.</summary>
    public void GotEncodedMediaFrame(EncodedAudioFrame encodedMediaFrame)
    {
    }

    /// <summary>Ignored (obsolete in sipsorcery); see <see cref="GotEncodedMediaFrame"/>.</summary>
    public void GotAudioRtp(System.Net.IPEndPoint remoteEndPoint, uint ssrc, uint seqnum, uint timestamp, int payloadID, bool marker, byte[] payload)
    {
    }

    public Task StartAudioSink() => Task.CompletedTask;

    public Task PauseAudioSink() => Task.CompletedTask;

    public Task ResumeAudioSink() => Task.CompletedTask;

    public Task CloseAudioSink() => Task.CompletedTask;
}
