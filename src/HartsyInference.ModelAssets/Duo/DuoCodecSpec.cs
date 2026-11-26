namespace HartsyInference.ModelAssets.Duo;

/// <summary>The CPU geometry of a Duo video codec mode: how many frames and latent steps a block holds, and how many audio latent steps
/// its duration needs. Mirrors <c>duo/models/realization/modules/codec_spec.py</c> @068e357. Only <c>duo_5_17</c> is the released
/// Duo v1 mode; the others are listed so a checkpoint or request that names them fails with a clear message, not silently.</summary>
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

    /// <summary>The spec for a mode name. An unknown name is an error, never a guess.</summary>
    public static DuoCodecSpec Get(string mode) => mode switch
    {
        "old" => new DuoCodecSpec { Mode = mode, FrameSpans = [4], Delayed = false },
        "sana_1_4" => new DuoCodecSpec { Mode = mode, FrameSpans = [4], Delayed = false },
        "duo_2_5" => new DuoCodecSpec { Mode = mode, FrameSpans = [1, 4], Delayed = false },
        "duo_5_17" => new DuoCodecSpec { Mode = mode, FrameSpans = [1, 4, 4, 4, 4], Delayed = true },
        _ => throw new ArgumentException($"Unknown video codec '{mode}'; choose old, sana_1_4, duo_2_5 or duo_5_17.", nameof(mode)),
    };
}
