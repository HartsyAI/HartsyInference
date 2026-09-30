namespace HartsyInference.Audio.Io;

/// <summary>The two ITU-T G.711 companding laws: μ-law (PCMU, North America and Japan) and A-law (PCMA, elsewhere).</summary>
public enum G711Law
{
    /// <summary>μ-law, RTP payload type 0 (PCMU).</summary>
    MuLaw,

    /// <summary>A-law, RTP payload type 8 (PCMA).</summary>
    ALaw,
}
