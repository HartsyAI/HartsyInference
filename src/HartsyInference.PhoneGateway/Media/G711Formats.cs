using HartsyInference.Audio.Io;
using SIPSorceryMedia.Abstractions;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>The two static RTP payload types the gateway speaks, mapped to <see cref="G711Law"/> and sipsorcery's
/// <see cref="AudioFormat"/>.</summary>
internal static class G711Formats
{
    /// <summary>RFC 3551 static payload type for PCMU.</summary>
    public const int PcmuPayloadType = 0;

    /// <summary>RFC 3551 static payload type for PCMA.</summary>
    public const int PcmaPayloadType = 8;

    public static AudioFormat Pcmu => new(SDPWellKnownMediaFormatsEnum.PCMU);

    public static AudioFormat Pcma => new(SDPWellKnownMediaFormatsEnum.PCMA);

    /// <summary>The formats to offer for <paramref name="preference"/>, most preferred first.</summary>
    public static List<AudioFormat> Offer(AudioCodecPreference preference) => preference switch
    {
        AudioCodecPreference.Pcmu => [Pcmu],
        AudioCodecPreference.Pcma => [Pcma],
        _ => [Pcmu, Pcma],
    };

    /// <summary>True for PCMU or PCMA, with the matching law.</summary>
    public static bool TryGetLaw(int payloadType, out G711Law law)
    {
        switch (payloadType)
        {
            case PcmuPayloadType:
                law = G711Law.MuLaw;
                return true;
            case PcmaPayloadType:
                law = G711Law.ALaw;
                return true;
            default:
                law = default;
                return false;
        }
    }

    /// <summary>True for a G.711 format, with the matching law.</summary>
    public static bool TryGetLaw(AudioFormat format, out G711Law law)
    {
        switch (format.Codec)
        {
            case AudioCodecsEnum.PCMU:
                law = G711Law.MuLaw;
                return true;
            case AudioCodecsEnum.PCMA:
                law = G711Law.ALaw;
                return true;
            default:
                law = default;
                return false;
        }
    }
}
