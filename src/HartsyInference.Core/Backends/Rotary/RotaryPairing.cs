namespace HartsyInference.Core.Backends.Rotary;

/// <summary>Which two elements of a rotated vector form a pair.</summary>
public enum RotaryPairing
{
    /// <summary>Adjacent elements <c>(2i, 2i + 1)</c>: GPT-J, Wan, Oasis, Matrix-Game 3, LTX-0.9, LTX-2.5, DeepSeek.</summary>
    Interleaved,

    /// <summary>Elements <c>(i, i + half)</c> of the rotated width: NeoX / Llama <c>rotate_half</c>, LTX-2 split.</summary>
    RotateHalf,
}
