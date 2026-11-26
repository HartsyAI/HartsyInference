namespace HartsyInference.ModelAssets.Duo;

/// <summary>The CPU geometry of the released Duo v1 video codec mode <c>duo_5_17</c>: how many frames and latent steps a block holds, and how
/// many audio latent steps its duration needs. Mirrors <c>duo/models/realization/modules/codec_spec.py</c> @068e357. Any other mode,
/// including the unreleased <c>old</c>, <c>sana_1_4</c> and <c>duo_2_5</c>, is refused.</summary>
public sealed record DuoCodecSpec
{
    public const int Fps = 24;
    public const int SampleRate = 32000;
    public const int AudioHop = 800;

    /// <summary>The mode name, as upstream spells it.</summary>
    public required string Mode { get; init; }

    /// <summary>Frames per latent step, in order. A block is one chunk of these steps.</summary>
    public required int[] FrameSpans { get; init; }

    /// <summary>Whether the decoder publishes each block one chunk late, with a two-latent lookahead (the <c>duo_5_17</c> policy).</summary>
    public required bool Delayed { get; init; }

    /// <summary>Video frames in one block.</summary>
    public int Frames => FrameSpans.Sum();

    /// <summary>Video latent steps in one block.</summary>
    public int Latents => FrameSpans.Length;

    /// <summary>Audio latent steps for one block's duration, per channel: <c>ceil(frames * 32000 / (24 * 800))</c>.</summary>
    public int AudioLatents
    {
        get
        {
            long numerator = (long)Frames * SampleRate, denominator = (long)Fps * AudioHop;
            return (int)((numerator + denominator - 1) / denominator);
        }
    }

    /// <summary>The spec of the released mode. Only <c>duo_5_17</c> is accepted; unreleased or unknown names are refused.</summary>
    public static DuoCodecSpec Get(string mode)
    {
        if (mode != "duo_5_17")
        {
            throw new NotSupportedException(
                $"Video codec '{mode}' is not released for Duo v1; only 'duo_5_17' is supported.");
        }
        return new DuoCodecSpec { Mode = mode, FrameSpans = [1, 4, 4, 4, 4], Delayed = true };
    }
}
